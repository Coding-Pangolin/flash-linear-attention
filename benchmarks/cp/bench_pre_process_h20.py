# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""pre_process_fwd_kernel_merged —— H20 基线采集脚本

采集上游 `fla/ops/cp/chunk_delta_h.py::pre_process_fwd_kernel_merged` 在 NVIDIA H20 上的
**纯 device kernel 耗时**, 作为 Ascend NPU 侧同名算子的对标基线。

用法
----
    python -m benchmarks.cp.bench_pre_process_h20 --list
    python -m benchmarks.cp.bench_pre_process_h20 --case model-g
    python -m benchmarks.cp.bench_pre_process_h20 --case model-gk
    python -m benchmarks.cp.bench_pre_process_h20 --case model-dplr
    python -m benchmarks.cp.bench_pre_process_h20 --case model-gk --t 2816   # 自定义窗口长度
    python -m benchmarks.cp.bench_pre_process_h20 --case model-gk --torch-profiler
    python -m benchmarks.cp.bench_pre_process_h20 --case model-gk --save-io ./case_gk

口径说明
--------
* `--warmup/--repeat` 的 CUDA event 数给出的是**流上事件间隔**(含 host launch 空隙),
  用作交叉验证; 定基线请用 nsys 或 `--torch-profiler`(CUPTI) 的 device 时长。
* `--nvtx`(默认开) 会在测量区间外套 NVTX range, 便于 nsys 滤掉 warmup 与 autotune。

语义约定(与上游 kernel 一致)
----------------------------
* 张量按 token-major 布局 [T, H, D]
* gate 是 base-2 的 chunk 内累积对数衰减, kernel 内部用 exp2
* 一次调用 = 一个窗口(一个 part); hm 形状 [HV, K, V+K], 左 V 列是 h, 右 K 列是 m
* hm 为 FP32; h 进 dot 前降到 BF16, v_new 也降 BF16, m 的链在 FP32 内累加
* cu_seqlens 传 [0, T] 与上游 CP 路径一致(该路径恒走 varlen 分支), 传 None 走定长分支

输入按 delta 规则构造: k 沿 head 维单位化, w = beta*k(beta ~ U(0, --beta-scale)),
bg = gamma*k。若直接用独立随机的 w, k^T w 的谱范数远大于 1, 递推会在几十个 chunk 内
发散成 NaN(计时不受影响, 但参考输出作废)。可用 --beta-scale / --decay-per-chunk /
--bg-scale 调节, 脚本会在输出非有限时给出警告。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

try:
    import triton
except Exception as exc:  # pragma: no cover
    print(f"[FATAL] 需要 triton(NVIDIA 版): {exc}")
    raise SystemExit(2)


PRECISIONS = ("default", "tf32x3", "ieee")

PRESETS = {
    "model-g": dict(variant="g", t=11264, hk=16, hv=32, k=128, v=128, bt=64),
    "model-gk": dict(variant="gk", t=11264, hk=32, hv=32, k=128, v=128, bt=64),
    "model-dplr": dict(variant="dplr", t=11264, hk=32, hv=32, k=128, v=128, bt=64),
    "tiny-gk": dict(variant="gk", t=256, hk=4, hv=4, k=128, v=128, bt=64),
}

DEST_OF = {"variant": "variant", "t": "t", "hk": "hk", "hv": "hv",
           "k": "kdim", "v": "vdim", "bt": "bt"}


# --------------------------------------------------------------------- 上游入口
def load_kernel():
    try:
        from fla.ops.cp.chunk_delta_h import pre_process_fwd_kernel_merged
    except Exception as exc:
        print("[FATAL] 无法导入 fla.ops.cp.chunk_delta_h:", exc)
        print("        请在 flash-linear-attention 仓库根目录执行(或先 pip install -e .)。")
        raise SystemExit(2)
    return pre_process_fwd_kernel_merged


def tf32_supported() -> bool:
    try:
        from fla.utils import IS_TF32_SUPPORTED
        return bool(IS_TF32_SUPPORTED)
    except Exception:
        return False


# ------------------------------------------------------------------ 输入构造
def make_decay(gen, T, shape, BT, device, decay_per_chunk, dtype=torch.float32):
    """base-2 的 chunk 内累积对数衰减, 形状 [T, *shape]。

    每个 chunk 的总衰减约 exp2(-decay_per_chunk), 逐 token 有小幅随机波动且严格单调递减。
    """
    d = decay_per_chunk / BT
    x = -d * (1.0 + torch.rand(T, *shape, generator=gen, dtype=torch.float32) * 0.5)
    nt = (T + BT - 1) // BT
    pad = nt * BT - T
    if pad:
        x = torch.cat([x, torch.zeros(pad, *shape, dtype=torch.float32)], dim=0)
    x = x.view(nt, BT, *shape).cumsum(dim=1).reshape(-1, *shape)[:T]
    return x.contiguous().to(device=device, dtype=dtype)


def build_case(a, device):
    gen = torch.Generator().manual_seed(a.seed)
    bf16, f32 = torch.bfloat16, torch.float32
    T, K, V, BT, HV = a.t, a.kdim, a.vdim, a.bt, a.hv

    use_bg = a.variant == "dplr"
    HK = HV if a.variant in ("gk", "dplr") else a.hk
    if a.variant in ("gk", "dplr") and a.hk != a.hv:
        print(f"[warn] {a.variant} 路径要求 HK==HV, 已把 HK 从 {a.hk} 改为 {a.hv}")

    def to_dev(x, dt=bf16):
        return x.to(device=device, dtype=dt).contiguous()

    # k 沿 head 维单位化, w = beta * k: 这样 delta 规则的状态更新是收缩的。
    # 直接用独立随机 w 会让 k^T w 的谱范数远大于 1, 递推在几十个 chunk 内就发散成 NaN。
    k = torch.nn.functional.normalize(torch.randn(T, HK, K, generator=gen), dim=-1)
    idx = torch.arange(HV) // max(1, HV // HK)

    if use_bg:
        beta = torch.rand(T, HK, 1, generator=gen) * a.beta_scale
        w = beta * k
        bg = (torch.rand(T, HK, 1, generator=gen) * a.bg_scale) * k
        u = to_dev(torch.randn(T, HV, V, generator=gen))
        v = to_dev(torch.randn(T, HV, V, generator=gen))
    else:
        beta = torch.rand(T, HV, 1, generator=gen) * a.beta_scale
        w = beta * k[:, idx, :]
        bg = None
        u = to_dev(torch.randn(T, HV, V, generator=gen))
        v = u                                  # GDN/KDA 下 v 复用 u

    g = make_decay(gen, T, (HV,), BT, device, a.decay_per_chunk) \
        if a.variant == "g" else None
    gk = None if a.variant == "g" \
        else make_decay(gen, T, (HV, K), BT, device, a.decay_per_chunk)
    hm = torch.zeros(HV, K, V + K, dtype=f32, device=device)
    cu = torch.tensor([0, T], dtype=torch.int32, device=device) if a.varlen else None

    return dict(k=to_dev(k), v=v, w=to_dev(w), g=g, gk=gk,
                bg=(to_dev(bg) if bg is not None else None),
                u=u, hm=hm, cu_seqlens=cu,
                H=HK, HV=HV, K=K, V=V, T=T, BT=BT), use_bg


# -------------------------------------------------------------------- MAC 估算
def estimate_macs(T, HV, K, V, BT, use_bg):
    """按上游列块分流的实际计算量估 MAC(含 m 半边 M_c 的重复计算)。"""
    bs = 32 if K <= 64 else 64
    nt = (T + BT - 1) // BT
    n_h = (V + bs - 1) // bs
    n_m = (K + bs - 1) // bs
    per_head = n_h * (BT * K * bs + K * BT * bs) + n_m * (K * BT * K) + K * K * V
    if use_bg:
        per_head += n_h * (K * BT * bs)
    return per_head * nt * HV


# --------------------------------------------------------------------- 计时
def bench_events(launch, warmup, repeat):
    """CUDA event: 单次 launch 的事件间隔 + 背靠背吞吐。"""
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    lat = []
    for _ in range(repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        launch()
        e.record()
        torch.cuda.synchronize()
        lat.append(s.elapsed_time(e))

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeat):
        launch()
    e.record()
    torch.cuda.synchronize()
    return lat, s.elapsed_time(e) / repeat


def _device_us(evt):
    for attr in ("device_time_total", "cuda_time_total",
                 "self_device_time_total", "self_cuda_time_total"):
        val = getattr(evt, attr, None)
        if val:
            return float(val)
    return 0.0


def bench_cupti(launch, warmup, repeat):
    """CUPTI(torch.profiler): 每个 kernel 的平均 device 时长, 纯 kernel 口径。"""
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(repeat):
            launch()
        torch.cuda.synchronize()

    rows = []
    for evt in prof.key_averages():
        total = _device_us(evt)
        if total > 0:
            rows.append((evt.key, int(evt.count), total / max(1, int(evt.count))))
    rows.sort(key=lambda r: -r[2])
    return rows


def pct(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round((len(vals) - 1) * p)))]


# ----------------------------------------------------------------------- main
def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="采集 H20 上 pre_process_fwd_kernel_merged 的 kernel 耗时",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--case", default="model-gk", choices=sorted(PRESETS))
    ap.add_argument("--variant", choices=("g", "gk", "dplr"), default=None)
    ap.add_argument("--t", type=int, default=None, help="窗口 token 数(不是整条序列)")
    ap.add_argument("--hk", type=int, default=None, help="key 侧 head 数")
    ap.add_argument("--hv", type=int, default=None, help="value 侧 head 数")
    ap.add_argument("--k", type=int, default=None, dest="kdim", help="key 维 K")
    ap.add_argument("--v", type=int, default=None, dest="vdim", help="value 维 V")
    ap.add_argument("--bt", type=int, default=None, help="chunk_size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta-scale", type=float, default=0.02,
                    help="w = beta*k 的 beta 上界; 直接决定 delta 规则的收缩程度")
    ap.add_argument("--decay-per-chunk", type=float, default=0.013,
                    help="每个 chunk 的 log2 总衰减量, chunk 内衰减率 = exp2(-该值)")
    ap.add_argument("--bg-scale", type=float, default=0.02,
                    help="仅 DPLR: bg = gamma*k 的 gamma 上界")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--precision", default="all", choices=("all",) + PRECISIONS,
                    help="只测某一种 AFFINE_CHAIN_PRECISION, 默认全测")
    ap.add_argument("--no-cu-seqlens", dest="varlen", action="store_false",
                    help="不传 cu_seqlens(走定长分支); 默认传 [0,T] 与上游 CP 路径一致")
    ap.add_argument("--nvtx", action="store_true", default=True,
                    help="在测量区间外套 NVTX range, 便于 nsys 过滤 warmup/autotune")
    ap.add_argument("--no-nvtx", dest="nvtx", action="store_false")
    ap.add_argument("--torch-profiler", action="store_true",
                    help="额外用 CUPTI 采一遍, 输出每个 kernel 的 device 时长")
    ap.add_argument("--save-io", default=None, metavar="DIR",
                    help="把本 case 的输入与参考输出落盘, 供 NPU 侧复现同一 case")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)

    for key, val in PRESETS[a.case].items():
        dest = DEST_OF[key]
        if getattr(a, dest) is None:
            setattr(a, dest, val)
    return a


def main(argv=None):
    a = parse_args(argv if argv is not None else sys.argv[1:])

    if a.list:
        print("内置 case:")
        for name, p in PRESETS.items():
            print(f"  {name:<12} variant={p['variant']:<5} T={p['t']:<6} "
                  f"HK={p['hk']:<4} HV={p['hv']:<4} K={p['k']} V={p['v']} BT={p['bt']}")
        print("\n命令里显式给的 --t/--hk/--hv/--k/--v/--bt 会覆盖 case 预设。")
        return 0

    if not torch.cuda.is_available():
        print("[FATAL] 当前环境没有可用的 CUDA 设备。")
        return 2
    torch.cuda.set_device(a.device)
    device = torch.device("cuda", a.device)
    prop = torch.cuda.get_device_properties(a.device)

    kernel = load_kernel()
    tensors, use_bg = build_case(a, device)

    K, V, BT = tensors["K"], tensors["V"], tensors["BT"]
    HV, T, HK = tensors["HV"], tensors["T"], tensors["H"]
    BK1 = triton.next_power_of_2(K)
    BS = 32 if K <= 64 else 64
    grid = (triton.cdiv(V, BS) + triton.cdiv(K, BS), HV)

    print("=" * 78)
    print("H20 baseline: fla-org pre_process_fwd_kernel_merged")
    print("=" * 78)
    print(f"device       : {prop.name}  SMs={prop.multi_processor_count}")
    print(f"torch/triton : {torch.__version__} / {triton.__version__}")
    print(f"case         : {a.case}  variant={a.variant}  varlen={a.varlen}")
    print(f"shape        : T={T} HK={HK} HV={HV} K={K} V={V} BT={BT} seed={a.seed}")
    print(f"grid         : {grid}   BLOCK_SIZE={BS}  BK1={BK1}  "
          f"IS_TF32_SUPPORTED={tf32_supported()}")
    macs = estimate_macs(T, HV, K, V, BT, use_bg)
    print(f"MAC/call     : {macs/1e6:.2f} M  ({macs*2/1e9:.2f} GFLOP, 含上游列块冗余)")
    print(f"hm           : {tuple(tensors['hm'].shape)} {tensors['hm'].dtype}")
    print()

    precisions = PRECISIONS if a.precision == "all" else (a.precision,)
    results = {}

    for prec in precisions:
        arg_prec = None if prec == "default" else prec

        def launch():
            kernel[grid](
                k=tensors["k"], v=tensors["v"], w=tensors["w"],
                g=tensors["g"], gk=tensors["gk"], bg=tensors["bg"], u=tensors["u"],
                hm=tensors["hm"], cu_seqlens=tensors["cu_seqlens"], T=T,
                H=HK, HV=HV, K=K, V=V, BT=BT, BK1=BK1, BLOCK_SIZE=BS,
                MULTI_SEQS=False, AFFINE_CHAIN_PRECISION=arg_prec,
            )

        try:
            launch()                       # 触发 autotune + 首次编译
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"[{prec}] 启动失败: {type(exc).__name__}: {exc}")
            continue

        if a.nvtx:
            torch.cuda.nvtx.range_push(f"bench:cp_pre_process:{prec}")
        lat, throttled = bench_events(launch, a.warmup, a.repeat)

        cupti = None
        if a.torch_profiler:
            cupti = bench_cupti(launch, a.warmup, a.repeat)
        if a.nvtx:
            torch.cuda.nvtx.range_pop()

        p50 = statistics.median(lat)
        p90 = pct(lat, 0.90)
        tf = macs * 2 / (p50 * 1e-3) / 1e12
        results[prec] = dict(min=min(lat), p50=p50, p90=p90, max=max(lat),
                             back_to_back=throttled, tflops=tf)
        print(f"[{prec}] warmup={a.warmup} repeat={a.repeat}")
        print(f"        event latency(ms): min={min(lat):.4f}  p50={p50:.4f}  "
              f"p90={p90:.4f}  max={max(lat):.4f}")
        print(f"        back-to-back     : {throttled:.4f} ms/call   有效算力≈{tf:.1f} TFLOPS")
        if cupti:
            print("        CUPTI device time(us, avg per call):")
            for key, count, us in cupti[:6]:
                mark = "  <== kernel" if "pre_process" in key else ""
                print(f"          {us:10.2f}  x{count:<6} {key[:70]}{mark}")
            results[prec]["cupti_top"] = [
                dict(key=k, count=c, avg_us=u) for k, c, u in cupti[:6]]

    hm = tensors["hm"]
    finite = bool(torch.isfinite(hm).all().item())
    print()
    print(f"输出检查     : finite={finite}  "
          f"|h|max={hm[:, :, :V].abs().max().item():.4e}  "
          f"|m|max={hm[:, :, V:].abs().max().item():.4e}")
    if not finite:
        print("  [WARN] 输出含非有限值: 该 case 的递推发散了。")
        print("         计时结果仍然可用(本 kernel 无数据相关分支, NaN 不改变吞吐),")
        print("         但 hm 不能作为参考输出。请调小 --beta-scale 或调大")
        print("         --decay-per-chunk 后重跑。")

    if a.save_io and results and finite:
        os.makedirs(a.save_io, exist_ok=True)
        payload = {name: (t.cpu() if torch.is_tensor(t) else t)
                   for name, t in tensors.items()}
        torch.save(payload, os.path.join(a.save_io, "case.pt"))
        meta = dict(case=a.case, variant=a.variant, varlen=a.varlen, seed=a.seed,
                    T=T, HK=HK, HV=HV, K=K, V=V, BT=BT,
                    grid=list(grid), block_size=BS, bk1=BK1, macs=macs,
                    device=prop.name, torch=torch.__version__, triton=triton.__version__,
                    results=results)
        with open(os.path.join(a.save_io, "case.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        print(f"\n已导出输入与参考输出 -> {a.save_io}/case.pt  (元数据: case.json)")

    if results:
        print()
        print("=" * 78)
        print("summary(可直接回贴)")
        print("=" * 78)
        for prec, r in results.items():
            print(f"case={a.case} variant={a.variant} T={T} HK={HK} HV={HV} "
                  f"K={K} V={V} BT={BT} prec={prec} "
                  f"p50={r['p50']*1000:.1f}us p90={r['p90']*1000:.1f}us "
                  f"min={r['min']*1000:.1f}us")
        print(f"device={prop.name} torch={torch.__version__} triton={triton.__version__}")
        print()
        print("nsys 采样(取纯 kernel 时长):")
        print("  nsys profile -o prof --force-overwrite=true --trace=cuda,nvtx \\")
        print(f"      python -m benchmarks.cp.bench_pre_process_h20 --case {a.case} \\")
        print(f"      --warmup {a.warmup} --repeat {a.repeat}")
        print("  nsys stats --report nvtx_gpu_proj_sum,cuda_gpu_kern_sum prof.nsys-rep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
