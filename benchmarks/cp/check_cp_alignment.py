"""CP 对齐检查：竞品的 hm 语义 vs CPU 标杆，以及多段 part 下导出的边界状态是否正确。

背景：本 AscendC 算子（pre_process_fwd_kernel_merged）的接口允许**一次调用处理多段**
（`B` 维 + 多段 `cu_seqlens`），而竞品的 CP 包装层一次只喂"每个 part 的末段"。本脚本用
两条相互独立的检查回答"两者是否等价、我们的接口语义是否成立"：

  [A] --mode hm （单卡）
      直接调 Triton kernel，三方对比：
        (1) 逐段调用       MULTI_SEQS=False，每段一次，hm 各一份
        (2) 多段一次调用   MULTI_SEQS=True，grid 第三维 = 段号，hm [N,HV,K,V+K]
        (3) CPU 标杆       逐段算（benchmarks/cp/pre_process_h20/reference.py）
      结论项：
        A1  (1) == (3)  → hm 级对齐（同窗口同结果）
        A2  (2) == (1)  → "多段一次算 == 逐段算"，即我们接口语义成立

  [B] --mode cp （torchrun，world_size>=2）
      跑完整 wrapper（build_cp_context + pre_process + merge），与独立 ground truth 比
      returned initial_state。两组输入：
        B1  切点对齐序列边界（对照，ground truth 应全 0）
        B2  一个 part 含多条序列、且切点落在序列内部（目标场景）
      结论项：
        B1  initial_state == 0
        B2  initial_state == "沿真实递推扫到窗口起点"的状态

用法：
    python  -m benchmarks.cp.check_cp_alignment --mode hm
    torchrun --nproc_per_node=2 -m benchmarks.cp.check_cp_alignment --mode cp
    # 也可以只跑 cp 的某个 layout
    torchrun --nproc_per_node=2 -m benchmarks.cp.check_cp_alignment --mode cp --layout zigzag

注意：对齐用 AFFINE_CHAIN_PRECISION="ieee"（与 H20 基线对齐时的口径一致），
否则 triton 会走 tf32，m 半边会与 fp32 标杆差出 1e-2 量级。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import triton

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "pre_process_h20"))

from reference import pre_process_fwd_kernel_merged as cpu_ref      # noqa: E402

import fla.ops.cp.chunk_delta_h as cpdh                             # noqa: E402
from fla.ops.cp import build_cp_context                             # noqa: E402

BF16, F32, I32 = torch.bfloat16, torch.float32, torch.int32
BT_DEFAULT = 64
PREC = "ieee"


# ------------------------------------------------------------------ 输入构造
def make_decay(gen, T, shape, BT, device, decay_per_chunk=0.013, dtype=F32):
    """base-2 的 chunk 内累积对数衰减，形状 [T, *shape]（与 bench_pre_process_h20 一致）。"""
    d = decay_per_chunk / BT
    x = -d * (1.0 + torch.rand(T, *shape, generator=gen, dtype=torch.float32) * 0.5)
    nt = (T + BT - 1) // BT
    pad = nt * BT - T
    if pad:
        x = torch.cat([x, torch.zeros(pad, *shape, dtype=torch.float32)], dim=0)
    x = x.view(nt, BT, *shape).cumsum(dim=1).reshape(-1, *shape)[:T]
    return x.contiguous().to(device=device, dtype=dtype)


def build_inputs(T, HK, HV, K, V, BT, device, seed=0, use_bg=False):
    """token-major 输入：k [T,HK,K] / w [T,HV,K] / v(=u) [T,HV,V] / gk [T,HV,K]。

    与 benchmarks/cp/bench_pre_process_h20.py 同一套构造（k 单位化、w = beta*k），
    否则 k^T w 谱范数过大，递推会在几十个 chunk 内发散。
    """
    gen = torch.Generator().manual_seed(seed)
    k = torch.nn.functional.normalize(torch.randn(T, HK, K, generator=gen), dim=-1)
    idx = torch.arange(HV) // max(1, HV // HK)
    beta = torch.rand(T, HV, 1, generator=gen) * 0.02
    w = beta * k[:, idx, :]
    v = torch.randn(T, HV, V, generator=gen)
    bg = (torch.rand(T, HK, 1, generator=gen) * 0.02) * k if use_bg else None
    gk = make_decay(gen, T, (HV, K), BT, device)
    to_dev = lambda x: x.to(device=device, dtype=BF16).contiguous()   # noqa: E731
    return dict(k=to_dev(k), w=to_dev(w), u=to_dev(v), v=to_dev(v),
                gk=gk, bg=(to_dev(bg) if bg is not None else None),
                HK=HK, HV=HV, K=K, V=V, T=T)


# ------------------------------------------------------------- 直接调 Triton kernel
def run_kernel(t, cu_list, multi_seqs, BT):
    """按 cu_list=[s0,s1,...] 在 GPU 上跑上游 kernel，返回 hm [N,HV,K,V+K] fp32。"""
    HK, HV, K, V, T = t["HK"], t["HV"], t["K"], t["V"], t["T"]
    dev = t["k"].device
    bs = 32 if K <= 64 else 64
    nseg = len(cu_list) - 1
    hm = torch.zeros(nseg, HV, K, V + K, dtype=F32, device=dev)
    common = dict(v=t["v"], w=t["w"], g=None, gk=t["gk"], bg=t["bg"], u=t["u"],
                  T=T, H=HK, HV=HV, K=K, V=V, BT=BT, BLOCK_SIZE=bs,
                  BK1=triton.next_power_of_2(K), AFFINE_CHAIN_PRECISION=PREC)
    grid_x = triton.cdiv(V, bs) + triton.cdiv(K, bs)
    if multi_seqs:
        # 一段一个 program：cu_seqlens 全程给全，i_n 由 grid 第三维承载
        cu = torch.tensor(cu_list, dtype=I32, device=dev)
        cpdh.pre_process_fwd_kernel_merged[(grid_x, HV, nseg)](
            k=t["k"], hm=hm, cu_seqlens=cu, MULTI_SEQS=True, **common)
    else:
        for i in range(nseg):
            cu = torch.tensor([cu_list[i], cu_list[i + 1]], dtype=I32, device=dev)
            cpdh.pre_process_fwd_kernel_merged[(grid_x, HV)](
                k=t["k"], hm=hm[i], cu_seqlens=cu, MULTI_SEQS=False, **common)
    return hm


def run_cpu(t, cu_list, BT):
    """逐段跑 CPU 标杆，返回 hm [N,HV,K,V+K] fp32（放在 CPU）。"""
    out = []
    for i in range(len(cu_list) - 1):
        out.append(cpu_ref(t["k"].cpu(), t["v"].cpu(), t["w"].cpu(), gk=t["gk"].cpu(),
                           bg=None if t["bg"] is None else t["bg"].cpu(),
                           cu_seqlens=(cu_list[i], cu_list[i + 1]), chunk_size=BT))
    return torch.stack(out)


def report(tag, got, want, atol=1e-6, rtol=2e-3):
    d = (got.float() - want.float()).abs()
    denom = want.float().abs().clamp_min(1e-6)
    matched = (d <= atol + rtol * denom).float().mean().item()
    print(f"  {tag:<34} max_abs={d.max().item():.3e}  mean_abs={d.mean().item():.3e}  matched={matched:.6f}")
    return d.max().item(), matched


# ------------------------------------------------------------------- [A] hm 级
HM_CASES = (
    # (tag, T, HK, HV, 段边界)  —— 段长故意含非 64 倍数与尾块
    ("aligned", 512, 4, 4, [0, 64, 128, 256, 512]),
    ("multi-段/非 64 倍数", 512, 4, 4, [0, 96, 256, 320, 512]),
    ("GVA 1:2（HK<HV）", 512, 2, 4, [0, 96, 256, 320, 512]),
)


def mode_hm(args):
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    print("=" * 96)
    print("[A] hm 级：多段一次算 vs 逐段算 vs CPU 标杆（AFFINE_CHAIN_PRECISION=ieee）")
    print("=" * 96)
    for tag, T, HK, HV, segs in HM_CASES:
        t = build_inputs(T, HK, HV, args.kdim, args.vdim, BT_DEFAULT, dev, seed=args.seed)
        hm_multi = run_kernel(t, segs, multi_seqs=True, BT=BT_DEFAULT)
        hm_single = run_kernel(t, segs, multi_seqs=False, BT=BT_DEFAULT)
        hm_cpu = run_cpu(t, segs, BT_DEFAULT)
        print(f"\ncase={tag}  T={T} HK={HK} HV={HV} K={t['K']} V={t['V']} "
              f"segs={segs}（{len(segs) - 1} 段）")
        report("A1 逐段(False) vs CPU", hm_single, hm_cpu)
        report("A2 多段(True)  vs CPU", hm_multi, hm_cpu)
        report("A3 多段(True)  vs 逐段", hm_multi, hm_single)
    print("\n判读：A1 通过 = hm 级对齐；A2/A3 通过 = 「多段一次算 == 逐段算」，"
          "即我们接口的多段语义成立。")


# ------------------------------------------------------------------- [B] CP 级
CP_PRESETS = {
    # 对照：切点（part_len 的整数倍）正好落在序列边界上
    "aligned": dict(segs=[0, 128, 256, 384, 512], world_hint=2),
    # 目标：part 含多条序列，且切点落在序列内部（rank1 的窗口 = seq1 的尾巴
    #       + seq2 全段 + seq3 的前半段，末段 ≠ 跨界段）
    "multi": dict(segs=[0, 40, 600, 700, 1024], world_hint=2),
}


def cp_ranges(layout, part_len, rank, world):
    if layout == "contiguous":
        return [(part_len * rank, part_len * (rank + 1))]
    front = (part_len * rank, part_len * (rank + 1))
    back = (part_len * (2 * world - 1 - rank), part_len * (2 * world - rank))
    return [front, back]


def mode_cp(args):
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(dev)

    preset = CP_PRESETS[args.preset]
    segs = list(args.segs) if args.segs else list(preset["segs"])
    num_parts = world if args.layout == "contiguous" else 2 * world
    if segs[-1] % num_parts != 0:
        raise SystemExit(f"segs[-1]={segs[-1]} 必须能被 num_parts={num_parts}（{args.layout}）整除")

    T = segs[-1]
    t = build_inputs(T, args.hk, args.hv, args.kdim, args.vdim, args.bt, dev, seed=args.seed)
    ctx = build_cp_context(torch.tensor(segs, dtype=I32), dist.group.WORLD, layout=args.layout)
    part_len = ctx.part_len
    ranges = cp_ranges(args.layout, part_len, rank, world)

    def local_of(x):
        return torch.cat([x[lo:hi] for lo, hi in ranges], dim=0).contiguous()

    out = cpdh.chunk_gated_delta_rule_fwd_h_pre_process(
        k=local_of(t["k"]), w=local_of(t["w"]), u=local_of(t["v"]), v=None,
        gk=local_of(t["gk"]), cu_seqlens=ctx.cu_seqlens, context=ctx)
    init = out[0] if isinstance(out, (tuple, list)) else out
    init = init.float()

    cu_local = [int(v) for v in ctx.cu_seqlens.cpu().tolist()]
    n_loc = len(cu_local) - 1

    def to_global(i):
        base = 0
        for lo, hi in ranges:
            if i < base + (hi - lo):
                return lo + (i - base)
            base += hi - lo
        raise IndexError(i)

    gt = torch.zeros_like(init)
    for n in range(n_loc):
        g_lo = to_global(cu_local[n])
        gs = max(s for s in segs if s <= g_lo)
        if g_lo > gs:                      # 该段的第一条 token 不是序列起点 → 需要携带状态
            hm = cpu_ref(t["k"].cpu(), t["v"].cpu(), t["w"].cpu(), gk=t["gk"].cpu(),
                         cu_seqlens=(gs, g_lo), chunk_size=args.bt)
            gt[n] = hm[:, :, :args.vdim].to(dev)

    d = (init - gt).abs()
    nz = int((gt.abs().sum(dim=(1, 2, 3)) > 0).sum().item())
    print(f"[rank {rank}] layout={args.layout} part_len={part_len} ranges={ranges} "
          f"local_segs={cu_local} 需要携带的段数={nz}  max_abs={d.max().item():.3e}")
    if nz:
        for n in range(n_loc):
            if gt[n].abs().sum() > 0:
                print(f"           seg#{n}: max_abs={(init[n] - gt[n]).abs().max().item():.3e} "
                      f"|gt|max={gt[n].abs().max().item():.3e}")
    dist.barrier()
    worst = torch.tensor([d.max().item()], device=dev)
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"\n[preset={args.preset}] 全局 max_abs = {worst.item():.3e}")
        print("判读：对照组（aligned）应为 0；目标组（multi）若显著 >0，说明竞品"
              "「每 part 只喂末段」在多段 part 下与真实递推不一致。")
    dist.destroy_process_group()


# ----------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="pre_process_fwd_kernel_merged 的 CP 对齐检查",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=("hm", "cp"), default="hm")
    ap.add_argument("--preset", choices=sorted(CP_PRESETS), default="multi")
    ap.add_argument("--layout", choices=("contiguous", "zigzag"), default="contiguous")
    ap.add_argument("--segs", type=int, nargs="*", default=None,
                    help="全局打包序列边界，如 --segs 0 40 600 700 1024")
    ap.add_argument("--hv", type=int, default=32, help="value head 数")
    ap.add_argument("--hk", type=int, default=32, help="key head 数")
    ap.add_argument("--k", type=int, default=128, dest="kdim")
    ap.add_argument("--v", type=int, default=128, dest="vdim")
    ap.add_argument("--bt", type=int, default=BT_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.mode == "hm":
        mode_hm(args)
    else:
        mode_cp(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
