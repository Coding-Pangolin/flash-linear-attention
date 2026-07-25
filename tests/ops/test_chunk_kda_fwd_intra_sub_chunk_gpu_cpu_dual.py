#!/usr/bin/env python3
"""GPU Triton vs CPU golden for chunk_kda_fwd_kernel_intra_sub_chunk.

Run on CUDA with this repo on PYTHONPATH (or editable install)::

    pip install -e .   # recommended, so fla matches this checkout
    pip install ct
    python tests/ops/test_chunk_kda_fwd_intra_sub_chunk_gpu_cpu_dual.py --smoke

Layout:
  - GPU / Triton: BSND  q/k/g [B, T, H, K], beta [B, T, H]
  - CPU golden:   BNSD  q/k/g [B, H, T, K], beta [B, H, T]

Outputs are compared in BNSD after converting GPU results; use ct.viz for plots.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
import triton

_TEST_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TEST_DIR.parents[1]
# Prefer this checkout's fla over a site-packages install (older wheels lack HV).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from chunk_kda_fwd_intra_sub_chunk_ref import (  # noqa: E402
    BC,
    chunk_kda_fwd_intra_sub_chunk_ref,
    prepare_chunk_indices,
)

try:
    import ct
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pip install ct  # required for ct.viz") from exc


def bsnd_to_bnsd(x: torch.Tensor) -> torch.Tensor:
    """[B, T, H, ...] -> [B, H, T, ...]"""
    return x.transpose(1, 2).contiguous()


def bnsd_to_bsnd(x: torch.Tensor) -> torch.Tensor:
    """[B, H, T, ...] -> [B, T, H, ...]"""
    return x.transpose(1, 2).contiguous()


def _make_gate_bnsd(
    B: int, H: int, T: int, K: int, dtype: torch.dtype, mode: str
) -> torch.Tensor:
    if mode == "lin_strong":
        g = -torch.linspace(0, 30, T).view(1, 1, T, 1).expand(B, H, T, K)
    elif mode == "lin_mild":
        g = -torch.linspace(0, 8, T).view(1, 1, T, 1).expand(B, H, T, K)
    elif mode == "rand":
        g = torch.randn(B, H, T, K)
    else:
        raise ValueError(mode)
    return g.to(dtype).contiguous()


def _kernel_arg_names(kernel) -> set[str]:
    """Unwrap Autotune/Heuristics wrappers and collect JIT arg names."""
    cur = kernel
    names: set[str] = set()
    for _ in range(6):
        if hasattr(cur, "arg_names") and cur.arg_names:
            names.update(cur.arg_names)
        if hasattr(cur, "fn"):
            cur = cur.fn
            continue
        break
    return names


def run_gpu_sub_chunk(
    q_bsnd: torch.Tensor,
    k_bsnd: torch.Tensor,
    g_bsnd: torch.Tensor,
    beta_bsnd: torch.Tensor,
    scale: float,
    chunk_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch Triton chunk_kda_fwd_kernel_intra_sub_chunk (safe_gate diagonal path)."""
    import fla
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_kernel_intra_sub_chunk
    from fla.ops.utils import prepare_chunk_indices as fla_prepare_chunk_indices
    from fla.utils import IS_GATHER_SUPPORTED

    print(f"  [fla] using {fla.__file__}", flush=True)

    B, T, H, K = k_bsnd.shape
    HV = g_bsnd.shape[2]
    if HV < H or HV % H != 0:
        raise ValueError(f"illegal GVA: H={H} HV={HV}")
    BT = chunk_size
    if BT not in (32, 64):
        raise ValueError(f"GPU kernel only supports chunk_size 32/64, got {BT}")

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = fla_prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    BK = triton.next_power_of_2(K)

    # Zero-init so unwritten tiles match CPU golden (production path uses empty).
    Aqk = torch.zeros(B, T, HV, BT, device=k_bsnd.device, dtype=k_bsnd.dtype)
    Akkd = torch.zeros(B, T, HV, BC, device=k_bsnd.device, dtype=torch.float32)

    grid = (NT, NC, B * HV)
    kwargs = dict(
        q=q_bsnd,
        k=k_bsnd,
        g=g_bsnd,
        beta=beta_bsnd,
        Aqk=Aqk,
        Akk=Akkd,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        USE_GATHER=IS_GATHER_SUPPORTED,
    )
    # Post-GVA kernels take HV; older site-packages installs only have H.
    arg_names = _kernel_arg_names(chunk_kda_fwd_kernel_intra_sub_chunk)
    if "HV" in arg_names:
        kwargs["HV"] = HV
    elif H != HV:
        raise RuntimeError(
            f"imported fla kernel has no HV arg (file={fla.__file__}) but H={H} != HV={HV}; "
            "install/use this repo's fla (pip install -e .)"
        )

    chunk_kda_fwd_kernel_intra_sub_chunk[grid](**kwargs)
    torch.cuda.synchronize()
    return Aqk, Akkd


def _print_stats(name: str, gpu: torch.Tensor, cpu: torch.Tensor) -> dict[str, float]:
    diff = (gpu.float() - cpu.float()).abs()
    rel = diff / cpu.float().abs().clamp_min(1.0)
    stats = {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": rel.max().item(),
        "gpu_finite": bool(torch.isfinite(gpu).all()),
        "cpu_finite": bool(torch.isfinite(cpu).all()),
    }
    print(
        f"  [{name}] max_abs={stats['max_abs']:.6g} mean_abs={stats['mean_abs']:.6g} "
        f"max_rel={stats['max_rel']:.6g} finite(gpu/cpu)={stats['gpu_finite']}/{stats['cpu_finite']}",
        flush=True,
    )
    return stats


def _diagnose_akkd_block(
    q_bnsd: torch.Tensor,
    k_bnsd: torch.Tensor,
    g_bnsd: torch.Tensor,
    beta_bnsd: torch.Tensor,
    scale: float,
    akkd_gpu: torch.Tensor,
    akkd_cpu: torch.Tensor,
    *,
    cpu_dtype: torch.dtype,
) -> None:
    """Locate worst Akkd block and check residual M @ (I+L) - I (logic vs numerics)."""
    from chunk_kda_fwd_intra_sub_chunk_ref import _forward_sub_inv

    diff = (akkd_gpu.float() - akkd_cpu.float()).abs()
    rel = diff / akkd_cpu.float().abs().clamp_min(1.0)
    # worst over [B,H,T,BC] → pick token row, then its BC-aligned block
    idx = rel.view(-1).argmax().item()
    B, H, T, _ = akkd_cpu.shape
    tmp = idx
    bc_i = tmp % BC
    tmp //= BC
    t = tmp % T
    tmp //= T
    h = tmp % H
    b = tmp // H
    i_ti = (t // BC) * BC
    valid = min(BC, T - i_ti)
    rows = list(range(i_ti, i_ti + valid))
    mid = i_ti + min(BC // 2, T - i_ti - 1)

    q_r = q_bnsd.to(cpu_dtype)
    k_r = k_bnsd.to(cpu_dtype)
    g_r = g_bnsd.to(cpu_dtype)
    beta_r = beta_bnsd.to(cpu_dtype)
    gm = g_r[b, h, rows, :] - g_r[b, h, mid, :]
    gq, gk = torch.exp2(gm), torch.exp2(-gm)
    k_pos = k_r[b, h, rows, :] * gq
    k_neg = k_r[b, h, rows, :] * gk
    beta_row = beta_r[b, h, rows]
    akk = (k_pos @ k_neg.transpose(-1, -2)) * beta_row[:, None]
    strict = torch.tril(torch.ones(valid, valid, dtype=torch.bool), diagonal=-1)
    L = torch.where(strict, akk, torch.zeros_like(akk))
    M_ref = _forward_sub_inv(L)
    eye = torch.eye(valid, dtype=cpu_dtype)

    Mg = akkd_gpu[b, h, rows, :valid].to(cpu_dtype)
    Mc = akkd_cpu[b, h, rows, :valid].to(cpu_dtype)
    resid_g = (Mg @ (eye + L) - eye).abs().max().item()
    resid_c = (Mc @ (eye + L) - eye).abs().max().item()
    resid_g_self = (Mg @ torch.linalg.inv(Mg) - eye).abs().max().item() if torch.isfinite(Mg).all() else float("nan")

    print(
        f"  [diagnose Akkd] worst @ b={b} h={h} t={t} (col={bc_i}) block_t0={i_ti} "
        f"rel={rel[b, h, t, bc_i].item():.6g}",
        flush=True,
    )
    print(
        f"    |L|max={L.abs().max().item():.6g} |M_cpu|max={Mc.abs().max().item():.6g} "
        f"|M_gpu|max={Mg.abs().max().item():.6g}",
        flush=True,
    )
    print(
        f"    residual M@(I+L)-I: cpu={resid_c:.6g} gpu_vs_cpu_L={resid_g:.6g}",
        flush=True,
    )
    print(
        f"    diag cpu[:4]={Mc.diag()[:4].tolist()} gpu[:4]={Mg.diag()[:4].tolist()}",
        flush=True,
    )
    if resid_c > 1.0:
        print(
            "    => CPU residual already large: |L| too big for stable fp32 forward-sub "
            "(raw randn k without L2 norm). Comparison is not meaningful; "
            "re-run with default --l2norm (or smaller k).",
            flush=True,
        )
    elif resid_c < 1e-2 and resid_g > 1.0:
        print(
            "    => GPU Akkd is NOT (I+L_cpu)^{-1}: L from GPU GEMM differs (bf16/TF32) "
            "or inv diverged; try --dtype fp32 to confirm.",
            flush=True,
        )
    elif resid_g < 1e-1:
        print("    => GPU Akkd ≈ (I+L_cpu)^{-1}: logic aligned; error is mostly elsewhere.", flush=True)


def _l2norm_last(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match FLA use_qk_l2norm_in_kernel: normalize over K."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def run_case(
    B: int,
    H: int,
    T: int,
    K: int,
    BT: int,
    *,
    varlen: bool = False,
    cu: Optional[Sequence[int]] = None,
    dtype: torch.dtype = torch.bfloat16,
    gate: str = "lin_mild",
    seed: int = 0,
    device: str = "cuda",
    cpu_dtype: torch.dtype = torch.float64,
    enable_viz: bool = True,
    sample_count: int = 200_000,
    viz_dir: Optional[Path] = None,
    aqk_tol: float = 5e-2,
    akkd_rel_tol: float = 1e-2,
    diagnose: bool = False,
    l2norm: bool = True,
) -> dict:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        # Prefer IEEE fp32 matmul when inputs are fp32; reduces TF32 noise in dual checks.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    scale = 1.0 / math.sqrt(K)
    q_bnsd = torch.randn(B, H, T, K, dtype=dtype)
    k_bnsd = torch.randn(B, H, T, K, dtype=dtype)
    if l2norm:
        # Without this, akk≈k@k.T has diag~K and |L|~O(10..100); fp32 (I+L)^{-1} blows up.
        q_bnsd = _l2norm_last(q_bnsd.float()).to(dtype)
        k_bnsd = _l2norm_last(k_bnsd.float()).to(dtype)
    g_bnsd = _make_gate_bnsd(B, H, T, K, dtype, gate)
    beta_bnsd = torch.rand(B, H, T, dtype=dtype)

    cu_t = None
    idx_flat = None
    idx_gpu = None
    if varlen or cu is not None:
        assert B == 1
        if cu is None:
            mid = T // 2
            cu = [0, mid, T]
        assert cu[0] == 0 and cu[-1] == T
        cu_t = torch.tensor(list(cu), dtype=torch.long)
        flat = prepare_chunk_indices(cu_t, BT)
        idx_flat = torch.tensor(flat, dtype=torch.long)
        # FLA prepare_chunk_indices returns [NT, 2]; kernel indexes as contiguous pairs.
        idx_gpu = idx_flat.view(-1, 2).contiguous()

    t0 = time.time()
    aqk_cpu, akkd_cpu = chunk_kda_fwd_intra_sub_chunk_ref(
        q_bnsd,
        k_bnsd,
        g_bnsd,
        beta_bnsd,
        scale,
        BT,
        cu_t,
        idx_flat,
        dtype=cpu_dtype,
    )
    t_cpu = time.time() - t0

    q_g = bnsd_to_bsnd(q_bnsd).to(device)
    k_g = bnsd_to_bsnd(k_bnsd).to(device)
    g_g = bnsd_to_bsnd(g_bnsd).to(device)
    beta_g = beta_bnsd.transpose(1, 2).contiguous().to(device)  # [B,H,T] -> [B,T,H]
    cu_g = None if cu_t is None else cu_t.to(device)
    idx_g = None if idx_gpu is None else idx_gpu.to(device)

    t1 = time.time()
    aqk_gpu_bsnd, akkd_gpu_bsnd = run_gpu_sub_chunk(
        q_g, k_g, g_g, beta_g, scale, BT, cu_g, idx_g
    )
    t_gpu = time.time() - t1

    aqk_gpu = bsnd_to_bnsd(aqk_gpu_bsnd.detach().cpu())
    akkd_gpu = bsnd_to_bnsd(akkd_gpu_bsnd.detach().cpu())

    tag = f"cu={list(cu)}" if cu is not None else "dense"
    print(
        f"\n[case] B={B} H={H} T={T} K={K} BT={BT} {tag} gate={gate} dtype={dtype} "
        f"cpu_dtype={cpu_dtype} l2norm={l2norm} t_cpu={t_cpu:.2f}s t_gpu={t_gpu:.2f}s",
        flush=True,
    )
    print(
        f"  shapes gpu_bsnd Aqk={tuple(aqk_gpu_bsnd.shape)} Akkd={tuple(akkd_gpu_bsnd.shape)} "
        f"| cpu_bnsd Aqk={tuple(aqk_cpu.shape)} Akkd={tuple(akkd_cpu.shape)}",
        flush=True,
    )

    aqk_stats = _print_stats("Aqk", aqk_gpu, aqk_cpu)
    akkd_stats = _print_stats("Akkd", akkd_gpu, akkd_cpu)
    if diagnose or akkd_stats["max_rel"] >= akkd_rel_tol:
        _diagnose_akkd_block(
            q_bnsd,
            k_bnsd,
            g_bnsd,
            beta_bnsd,
            scale,
            akkd_gpu,
            akkd_cpu,
            cpu_dtype=cpu_dtype,
        )

    if enable_viz:
        out = Path(viz_dir) if viz_dir is not None else Path("./viz_chunk_kda_fwd_intra_sub_chunk")
        case_dir = out / (
            f"B{B}_H{H}_T{T}_K{K}_BT{BT}_{tag.replace('=', '').replace(',', '-')}_{gate}"
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        viz_kwargs = {"out_dir": str(case_dir), "sample_count": int(sample_count)}
        print(f"  [viz] writing to {case_dir}", flush=True)
        ct.viz(aqk_gpu.float(), aqk_cpu.float(), name="Aqk", **viz_kwargs)
        ct.viz(akkd_gpu.float(), akkd_cpu.float(), name="Akkd", **viz_kwargs)

    assert aqk_stats["gpu_finite"] and aqk_stats["cpu_finite"]
    assert akkd_stats["gpu_finite"] and akkd_stats["cpu_finite"]
    assert aqk_stats["max_abs"] < aqk_tol, aqk_stats
    assert akkd_stats["max_rel"] < akkd_rel_tol, akkd_stats

    return {
        "aqk": aqk_stats,
        "akkd": akkd_stats,
        "t_cpu": t_cpu,
        "t_gpu": t_gpu,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPU Triton vs CPU golden for chunk_kda_fwd_intra_sub_chunk"
    )
    p.add_argument(
        "--device",
        default=os.environ.get("FLA_DUAL_DEVICE", "cuda:0"),
        help="cuda device, e.g. cuda:0 or 0",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate", default="lin_mild", choices=["lin_mild", "lin_strong", "rand"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument(
        "--cpu-dtype",
        default="fp64",
        choices=["fp64", "fp32"],
        help="golden compute dtype (fp32 closer to GPU/NPU accum)",
    )
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("-sc", "--sample-count", type=int, default=200_000)
    p.add_argument("--viz-dir", type=Path, default=None)
    p.add_argument("--smoke", action="store_true", help="only small dense/varlen cases")
    p.add_argument("--model", action="store_true", help="also H=32,T=2048/4096")
    p.add_argument(
        "--diagnose",
        action="store_true",
        help="print worst Akkd block residual M@(I+L)-I (auto-on when Akkd assert fails)",
    )
    p.add_argument(
        "--no-l2norm",
        action="store_true",
        help="disable Q/K L2 norm (will make |L|~O(K) and fp32 Akkd unstable)",
    )
    p.add_argument("--B", type=int, default=None)
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--BT", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run the GPU Triton path")

    device = args.device
    if device.isdigit():
        device = f"cuda:{device}"
    torch.cuda.set_device(device)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    cpu_map = {"fp64": torch.float64, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    cpu_dtype = cpu_map[args.cpu_dtype]
    viz = not args.no_viz
    common = dict(
        dtype=dtype,
        gate=args.gate,
        seed=args.seed,
        device=device,
        cpu_dtype=cpu_dtype,
        enable_viz=viz,
        sample_count=args.sample_count,
        viz_dir=args.viz_dir,
        diagnose=args.diagnose,
        l2norm=not args.no_l2norm,
    )

    if all(v is not None for v in (args.B, args.H, args.T, args.K, args.BT)):
        run_case(args.B, args.H, args.T, args.K, args.BT, **common)
        print("custom case passed")
        return

    # GPU kernel supports BT in {32, 64} only
    run_case(1, 2, 64, 128, 64, **common)
    run_case(1, 2, 64, 128, 32, **common)
    run_case(1, 2, 96, 128, 64, varlen=True, **common)
    run_case(1, 2, 96, 128, 64, cu=[0, 40, 96], **common)
    run_case(2, 2, 64, 128, 64, **common)
    run_case(1, 2, 48, 128, 64, **common)
    run_case(1, 4, 128, 128, 64, **{**common, "gate": "rand"})

    if not args.smoke:
        # Large T: more blocks → worse max_rel under bf16/TF32; loosen Akkd check.
        run_case(1, 8, 512, 128, 64, **{**common, "akkd_rel_tol": 5e-2})
        if args.model:
            run_case(1, 16, 2048, 128, 64, **{**common, "akkd_rel_tol": 5e-1})
            run_case(1, 32, 4096, 128, 64, **{**common, "akkd_rel_tol": 5e-1})

    print("all cases passed")


if __name__ == "__main__":
    main()
