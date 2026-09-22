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
      returned initial_state。三个 preset：
        aligned  切点全部对齐序列边界 —— **纯对照**：两个 rank 都 is_first & is_last，
                 kernel 与 merge 都不发，所以它过不过与被测语义无关（只看有没有多余状态）
        cut      一条序列被从中间切开（最经典的 CP 情形）
        multi    一个 part 含多条序列、且切点落在序列内部（目标场景，末段 != 跨界段）
      结论项：
        需要携带状态的段上  initial_state == "沿真实递推扫到窗口起点"的状态
        其余段上            initial_state == 0

      每个"需要携带状态"的段会打三行数：
        |gt|max                                      ground truth 的量级
        噪声基准A = |单调用<cp-precision> - 单调用ieee|  同一支 kernel 换精度档位的抖动
        噪声基准B = |同精度拆块复合 - 同精度单调用|     同一支 kernel、同一精度，只把前缀按
                                                       part 边界拆成多段再前缀复合（= 竞品
                                                       merge 的语义）后与一次算完的差
        CP 路径  = |init - 单调用ieee|                  竞品 chained-merge 的结果与真值之差
      判据：CP 路径 ≈ 噪声基准B → 语义一致（差异只是"分块口径不同"的 bf16 量化点差异）；
      差出几个量级 → 竞品在多段 part 下不一致。之所以要有 B 这一项：窗口短的时候
      |tf32x3 - ieee| 经常正好是 0（tf32x3 精度接近 fp32），拿它当基准判不了事。
      注意 wrapper 不接受 ieee（H20 上只有 tf32 / tf32x3 两条路），所以 CP 路径的精度
      由 --cp-precision 控制，默认 tf32x3（最接近真值）。

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
import os
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
def run_kernel(t, cu_list, multi_seqs, BT, prec=PREC):
    """按 cu_list=[s0,s1,...] 在 GPU 上跑上游 kernel，返回 hm [N,HV,K,V+K] fp32。

    prec 直接接到 kernel 的 AFFINE_CHAIN_PRECISION（只影响 m 那侧的 FP32 链式乘）：
    "ieee" / "tf32x3" / None（None = triton 默认，NVIDIA 上是 tf32）。
    """
    HK, HV, K, V, T = t["HK"], t["HV"], t["K"], t["V"], t["T"]
    dev = t["k"].device
    bs = 32 if K <= 64 else 64
    nseg = len(cu_list) - 1
    hm = torch.zeros(nseg, HV, K, V + K, dtype=F32, device=dev)
    common = dict(v=t["v"], w=t["w"], g=None, gk=t["gk"], bg=t["bg"], u=t["u"],
                  T=T, H=HK, HV=HV, K=K, V=V, BT=BT, BLOCK_SIZE=bs,
                  BK1=triton.next_power_of_2(K), AFFINE_CHAIN_PRECISION=prec)
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
    # CPU 标杆在 CPU 上，GPU 结果在 device 上：统一搬到 CPU 再比（张量都很小）
    got = got.detach().float().cpu()
    want = want.detach().float().cpu()
    d = (got - want).abs()
    denom = want.abs().clamp_min(1e-6)
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
    # 最经典的 CP 情形：一条序列被从中间切开（rank1 的窗口起点落在 seq0 内部）
    "cut": dict(segs=[0, 600, 1024], world_hint=2),
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


def _kernel_prec(cp_precision: str):
    """--cp-precision → kernel 的 AFFINE_CHAIN_PRECISION（只作用于 m 那侧的 FP32 链）。"""
    return {"tf32x3": "tf32x3", "tf32": None}[cp_precision]


def mode_cp(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dev = torch.device("cuda", local_rank % torch.cuda.device_count())
    torch.cuda.set_device(dev)
    try:                       # 显式给 device_id，避免 barrier/init 的设备警告
        dist.init_process_group("nccl", device_id=dev)
    except TypeError:          # 老版本 torch 没有 device_id
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2:
        raise SystemExit(f"CP 检查需要 world_size >= 2，当前 world_size={world}（请用 torchrun 起）")

    preset = CP_PRESETS[args.preset]
    segs = list(args.segs) if args.segs else list(preset["segs"])
    if segs[0] != 0 or any(b <= a for a, b in zip(segs, segs[1:])):
        raise SystemExit(f"--segs 必须是从 0 开始的严格递增边界，当前 {segs}")
    num_parts = world if args.layout == "contiguous" else 2 * world
    if segs[-1] % num_parts != 0:
        raise SystemExit(f"segs[-1]={segs[-1]} 必须能被 num_parts={num_parts}（{args.layout}）整除")

    T = segs[-1]
    t = build_inputs(T, args.hk, args.hv, args.kdim, args.vdim, args.bt, dev, seed=args.seed)
    use_tf32x3 = args.cp_precision == "tf32x3"
    kernel_prec = _kernel_prec(args.cp_precision)
    # 注意：build_cp_context 是从传进来的 cu_seqlens **推导设备**的
    # （`local.to(device=cu_seqlens.device)`），所以这里必须给 device 张量，
    # 否则 context.cu_seqlens 会留在 CPU 上，kernel 启动时报
    # "Pointer argument (at 8) cannot be accessed from Triton (cpu tensor?)"。
    ctx = build_cp_context(torch.tensor(segs, dtype=I32, device=dev), dist.group.WORLD,
                           layout=args.layout, use_tf32x3_affine_chain=use_tf32x3)
    assert ctx.cu_seqlens.is_cuda, f"context.cu_seqlens 还在 {ctx.cu_seqlens.device} 上"
    # 注意：FLACPContext.part_len 只在 zigzag 分支里被赋值，contiguous 下是 None，
    # 所以这里自己按 num_parts 算一遍。
    part_len = T // num_parts
    ranges = cp_ranges(args.layout, part_len, rank, world)

    def local_of(x):
        # wrapper 要求 4 维 [B, T, H, D]：补一个 B=1 维
        return torch.cat([x[lo:hi] for lo, hi in ranges], dim=0).unsqueeze(0).contiguous()

    out = cpdh.chunk_gated_delta_rule_fwd_h_pre_process(
        k=local_of(t["k"]), w=local_of(t["w"]), u=local_of(t["v"]), v=None,
        chunk_size=args.bt, gk=local_of(t["gk"]),
        cu_seqlens=ctx.cu_seqlens, context=ctx)
    init = (out[0] if isinstance(out, (tuple, list)) else out).float()

    cu_local = [int(v) for v in ctx.cu_seqlens.cpu().tolist()]
    n_loc = len(cu_local) - 1

    def to_global(i):
        # 局部下标 → 全局 token 下标，用于"段的左端点"。zigzag 下各窗口在全局空间里不连续，
        # 所以正好落在窗口右端点的下标要归到"下一个窗口的左端点"。
        # i 可以取到 T_local，此时归到最后一段的右端点。
        base = 0
        for j, (lo, hi) in enumerate(ranges):
            if i < base + (hi - lo) or j == len(ranges) - 1:
                return lo + (i - base)
            base += hi - lo
        raise IndexError(i)

    def to_global_end(i):
        # 局部下标 → 全局 token 下标，用于"段的右端点"：正好落在窗口边界时取该窗口的右端。
        base = 0
        for j, (lo, hi) in enumerate(ranges):
            if i <= base + (hi - lo) or j == len(ranges) - 1:
                return lo + (i - base)
            base += hi - lo
        raise IndexError(i)

    def seq_start(g):
        return max(s for s in segs if s <= g)

    # 拆块分解基准：同一段前缀，让"同一支 kernel + 同一精度"按 part 边界拆成多个窗口再复合，
    # 与"一次算完"比。竞品的 merge 就是"前缀复合"，所以这个量才是 CP 路径真正的噪声基准
    # （|单调用 tf32x3 - 单调用 ieee| 在窗口较短时经常退化成 0，判不了事）。
    split_points = sorted({k * part_len for k in range(1, T // part_len)})

    def chain_of(lo, hi):
        hm = run_kernel(t, [lo, hi], False, args.bt, prec=kernel_prec)[0]
        return hm[:, :, :V].clone(), hm[:, :, V:].clone()

    def compose_prefix(lo, hi):
        """把 [lo, hi) 按 part 边界拆开、逐段求链再前缀复合（= 竞品 merge 的语义）。"""
        bounds = [lo] + [s for s in split_points if lo < s < hi] + [hi]
        h = m = None
        for a, b in zip(bounds, bounds[1:]):
            hh, mm = chain_of(a, b)
            h = hh if h is None else hh + torch.einsum("hkj,hjv->hkv", mm, h)
            m = mm if m is None else torch.einsum("hkj,hjl->hkl", mm, m)
        return h

    V = args.vdim
    # 用 1 元素张量（不用 0 维）做 all_reduce，避免后端对空/标量张量的限制
    worst_err = torch.zeros(1, device=dev)
    worst_noise = torch.zeros(1, device=dev)
    worst_zero = torch.zeros(1, device=dev)
    n_carry = 0

    print(f"[rank {rank}] layout={args.layout} part_len={part_len} ranges={ranges} "
          f"local_segs={cu_local} cp_precision={args.cp_precision} "
          f"ctx_cu_seqlens.device={ctx.cu_seqlens.device}")
    # 把 wrapper 的判定也打出来：contiguous 下 is_last_rank 决定"发不发 kernel"，
    # is_first_rank 决定"做不做 merge"；两个都为 True 就是纯对照（本 rank 什么都不用算）。
    if args.layout == "contiguous":
        print(f"          is_first_rank={ctx.is_first_rank} is_last_rank={ctx.is_last_rank} "
              f"pre_num_ranks={ctx.pre_num_ranks} post_num_ranks={ctx.post_num_ranks}")
        if ctx.is_first_rank and ctx.is_last_rank:
            print("          （该 rank 既不用发 kernel 也不用 merge：切点全落在序列边界上，纯对照）")
    else:
        print(f"          is_first_by_part={ctx.is_first_by_part} "
              f"is_last_by_part={ctx.is_last_by_part} "
              f"pre_by_part={ctx.pre_num_ranks_by_part} post_by_part={ctx.post_num_ranks_by_part}")
    for n in range(n_loc):
        g_lo, g_hi = to_global(cu_local[n]), to_global_end(cu_local[n + 1])
        gs = seq_start(g_lo)
        if g_lo == gs:
            z = init[n].abs().max().reshape(1)
            worst_zero = torch.maximum(worst_zero, z)
            print(f"          seg#{n}: 全局[{g_lo},{g_hi}) 起点即序列起点({gs}) → 期望 0，"
                  f"实测 |init|max={z.item():.3e}")
            continue
        n_carry += 1
        # ground truth = 沿该序列的真实递推，从序列起点扫到本 rank 窗口起点；三种口径：
        #   CPU 标杆（fp32 主机，定义上的真值）
        #   单调用 kernel + ieee（GPU 上不做 merge 的真值）
        #   单调用 kernel + 与 CP 路径相同精度（本机精度噪声基准）
        gt_cpu = cpu_ref(t["k"].cpu(), t["v"].cpu(), t["w"].cpu(), gk=t["gk"].cpu(),
                         cu_seqlens=(gs, g_lo), chunk_size=args.bt)[:, :, :V].to(dev)
        gt_ieee = run_kernel(t, [gs, g_lo], False, args.bt, prec="ieee")[0][:, :, :V]
        gt_same = run_kernel(t, [gs, g_lo], False, args.bt, prec=kernel_prec)[0][:, :, :V]
        gt_split = compose_prefix(gs, g_lo)

        err = (init[n] - gt_ieee).abs().max().reshape(1)
        noise_prec = (gt_same - gt_ieee).abs().max().reshape(1)
        noise_split = (gt_split - gt_same).abs().max().reshape(1)
        err_cpu = (init[n] - gt_cpu).abs().max().reshape(1)
        mag = gt_ieee.abs().max().reshape(1)
        noise = torch.maximum(noise_prec, noise_split)
        thr = torch.maximum(torch.maximum(10 * noise, 0.05 * mag),
                            torch.tensor([1e-2], device=dev))
        worst_err = torch.maximum(worst_err, err)
        worst_noise = torch.maximum(worst_noise, noise)
        print(f"          seg#{n}: 全局[{g_lo},{g_hi}) 需要携带状态 ← 从 {gs} 扫到 {g_lo} 共 "
              f"{g_lo - gs} tokens  |gt|max={mag.item():.3e}")
        print(f"                 噪声基准A 精度档位 |单调用{args.cp_precision} - 单调用ieee| = "
              f"{noise_prec.item():.3e}")
        print(f"                 噪声基准B 拆块分解 |同精度拆块复合 - 同精度单调用| = "
              f"{noise_split.item():.3e}  （拆点 {[s for s in split_points if gs < s < g_lo]}）")
        print(f"                 CP 路径 |init - 单调用ieee| = {err.item():.3e}"
              f"   (vs CPU 标杆 {err_cpu.item():.3e})"
              f"   rel={err.item() / max(mag.item(), 1e-12):.3e}"
              f"   阈值 {thr.item():.3e} → {'一致' if err <= thr else '不一致（需人工确认）'}")
    dist.barrier()
    for x in (worst_err, worst_noise, worst_zero):
        dist.all_reduce(x, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"\n[preset={args.preset} layout={args.layout} cp_precision={args.cp_precision} "
              f"world={world}]")
        print(f"  本 rank 需要携带状态的段数={n_carry}  全局 max|init-gt_ieee|={worst_err.item():.3e}"
              f"  噪声基准={worst_noise.item():.3e}  期望为 0 的段上 |init|max={worst_zero.item():.3e}")
        print("  判读：对照组 aligned（没有段需要携带状态）应全 0。若 init 与 ieee 的差只到噪声量级"
              " → 竞品「每 part 只喂末段」在多段 part 下与真实递推一致；若差出几个量级 → 不一致。")
    dist.destroy_process_group()


# ----------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="pre_process_fwd_kernel_merged 的 CP 对齐检查",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=("hm", "cp"), default="hm")
    ap.add_argument("--preset", choices=sorted(CP_PRESETS), default="multi")
    ap.add_argument("--layout", choices=("contiguous", "zigzag"), default="contiguous")
    ap.add_argument("--cp-precision", choices=("tf32x3", "tf32"), default="tf32x3",
                    help="CP 路径上 m 那侧 FP32 链式乘的精度（经 use_tf32x3_affine_chain 透传）")
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
