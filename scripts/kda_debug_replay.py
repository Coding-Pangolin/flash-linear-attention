"""Replay KDA chunk_kda from prod debug dump (``kda_debug_input_tensors_rank0.pt``).

Dump is a flat dict, e.g. from model forward hook::

    torch.load("kda_debug_input_tensors_rank0.pt")
    {
        "step", "layer", "mode", "safe_gate", "lower_bound",
        "q", "k", "v", "g", "beta", "A_log", "dt_bias",
        "initial_state", "cu_seqlens",
    }

Quick use::

    from scripts.kda_debug_replay import run_chunk_kda_from_debug_dump

    o, final_state = run_chunk_kda_from_debug_dump("kda_debug_input_tensors_rank0.pt")
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Union

# Prefer repo root when running: python3 scripts/kda_debug_replay.py ...
_SCRIPT_DIR = Path(__file__).resolve().parent
_GPU_ROOT = _SCRIPT_DIR.parent


def _bootstrap_local_fla(gpu_root: Path, *, force_triton: bool = False) -> None:
    """Force imports from this repo's ``fla/``, not site-packages.

    ``pip install -e .`` is optional; ``PYTHONPATH=<repo>`` or this bootstrap
    is enough when you run the script from the checkout.
    """
    gpu_root = gpu_root.resolve()
    chunk_fwd = gpu_root / "fla" / "ops" / "kda" / "chunk_fwd.py"
    if not chunk_fwd.is_file():
        raise RuntimeError(
            f"expected {chunk_fwd} — you are not on feat/kda-gpu-dump (or equivalent).\n"
            "  git fetch coding-pangolin feat/kda-gpu-dump && git checkout feat/kda-gpu-dump"
        )

    if force_triton:
        os.environ["FLA_FLASH_KDA"] = "0"
        os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "1"

    root_s = str(gpu_root)
    if root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)

    # Drop any previously imported fla (e.g. from site-packages in REPL / notebook).
    for name in list(sys.modules):
        if name == "fla" or name.startswith("fla."):
            del sys.modules[name]


def _print_fla_import_paths() -> None:
    import fla
    import fla.ops.kda.chunk as kda_chunk_mod
    import fla.ops.kda.chunk_fwd as kda_chunk_fwd_mod

    from fla.ops.backends import BackendRegistry

    BackendRegistry.ensure_initialized("kda")
    reg = BackendRegistry._registries.get("kda")
    active = reg.get_active() if reg is not None else None

    print("[kda_debug_replay] === fla import paths ===", flush=True)
    print(f"  fla.__file__              = {fla.__file__}", flush=True)
    print(f"  fla.ops.kda.chunk         = {kda_chunk_mod.__file__}", flush=True)
    print(f"  fla.ops.kda.chunk_fwd     = {kda_chunk_fwd_mod.__file__}", flush=True)
    print(f"  chunk_kda                 = {kda_chunk_mod.chunk_kda}", flush=True)
    print(
        f"  FLA_FLASH_KDA             = {os.environ.get('FLA_FLASH_KDA', '(default)')}",
        flush=True,
    )
    print(
        f"  FLA_DISABLE_BACKEND_DISPATCH = "
        f"{os.environ.get('FLA_DISABLE_BACKEND_DISPATCH', '0')}",
        flush=True,
    )
    print(
        f"  active kda backend        = "
        f"{None if active is None else active.backend_type}",
        flush=True,
    )
    print("[kda_debug_replay] ========================", flush=True)

    repo_root = _GPU_ROOT.resolve()
    for label, path in (
        ("fla", fla.__file__),
        ("chunk", kda_chunk_mod.__file__),
        ("chunk_fwd", kda_chunk_fwd_mod.__file__),
    ):
        if path is None or not Path(path).resolve().is_relative_to(repo_root):
            raise RuntimeError(
                f"{label} loaded from {path!r}, not under repo {repo_root}.\n"
                "Your prints in the checkout are ignored. Use one of:\n"
                f"  PYTHONPATH={repo_root} python3 scripts/kda_debug_replay.py ...\n"
                f"  python3 scripts/kda_debug_replay.py --fla-root {repo_root} ...\n"
                "Or fix editable install: pip uninstall flash-linear-attention fla -y; "
                f"pip install -e {repo_root}"
            )


import torch

DumpLike = Union[str, Path, Mapping[str, Any]]


def _kda_fwd_debug_g_enabled() -> bool:
    return os.environ.get("KDA_FWD_DEBUG_G", "0").strip().lower() in ("1", "true", "yes", "on")


def _print_tensor_stats(name: str, t: torch.Tensor | None) -> None:
    if t is None:
        print(f"[KDA_FWD_DEBUG_G] {name}: None", flush=True)
        return
    x = t.detach().float().reshape(-1).cpu()
    finite = torch.isfinite(x)
    n_finite = int(finite.sum())
    n_total = x.numel()
    if n_finite == 0:
        print(
            f"[KDA_FWD_DEBUG_G] {name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"finite=0/{n_total}",
            flush=True,
        )
        return
    xf = x[finite]
    print(
        f"[KDA_FWD_DEBUG_G] {name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"finite={n_finite}/{n_total} min={xf.min().item():.6g} max={xf.max().item():.6g} "
        f"mean={xf.mean().item():.6g}",
        flush=True,
    )


def _maybe_log_g_from_dump(
    *,
    g: torch.Tensor,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    lower_bound: float,
    chunk_size: int,
    scale: float,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> None:
    if not _kda_fwd_debug_g_enabled():
        return

    import fla.ops.kda.chunk as kda_chunk_mod

    print(
        f"[KDA_FWD_DEBUG_G] run_chunk_kda_from_debug_dump: "
        f"KDA_FWD_DEBUG_G={os.environ.get('KDA_FWD_DEBUG_G')} "
        f"FLA_FLASH_KDA={os.environ.get('FLA_FLASH_KDA', '(default)')} "
        f"fla.ops.kda.chunk={kda_chunk_mod.__file__}",
        flush=True,
    )
    _print_tensor_stats("g_raw (dump g, before cumsum)", g)

    try:
        from fla.ops.kda.debug_g import log_kda_g_before_intra

        log_kda_g_before_intra(
            path="kda_debug_replay",
            g=g,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            chunk_size=chunk_size,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
        )
        return
    except ImportError:
        pass

    if not use_gate_in_kernel:
        print("[KDA_FWD_DEBUG_G] skip gk recompute (use_gate_in_kernel=False)", flush=True)
        return

    from fla.ops.kda.gate import kda_gate_chunk_cumsum
    from fla.ops.utils.constant import RCP_LN2

    gk = kda_gate_chunk_cumsum(
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=RCP_LN2,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
    )
    _print_tensor_stats("gk (cumsum log2, intra input)", gk)


def _finite_stats(t: torch.Tensor) -> str:
    x = t.detach().float().view(-1)
    return (
        f"finite {int(torch.isfinite(x).sum())}/{x.numel()} "
        f"(nan={int(torch.isnan(x).sum())}, inf={int(torch.isinf(x).sum())})"
    )


def _as_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    return dev


def _guess_beta_sigmoid_in_kernel(beta: torch.Tensor) -> bool:
    """Dump from NPU fused path usually stores sigmoid(beta_raw); GPU needs raw only if flag True."""
    x = beta.detach().float().view(-1)
    if x.numel() == 0:
        return True
    lo = float(x.min())
    hi = float(x.max())
    # sigmoid(beta_raw) for typical randn raw stays in (0, 1) with little mass at exact 0/1
    if lo >= -0.05 and hi <= 1.05:
        return False
    return True


def load_kda_debug_dump(path: str | Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"expected dict in dump, got {type(data)}")
    for key in ("q", "k", "v", "g", "beta"):
        if key not in data:
            raise KeyError(f"dump missing required key {key!r}")
    return data


def run_chunk_kda_from_debug_dump(
    dump: DumpLike,
    *,
    device: str | torch.device = "cuda:0",
    chunk_size: int = 64,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool | None = None,
    output_final_state: bool = True,
    verbose: bool = True,
    fla_root: Path | None = None,
    force_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run GPU ``chunk_kda`` with tensors from a prod debug dump.

    Matches NPU fused model path when dump q/k are already L2-normalized and
    ``beta`` is already ``sigmoid(beta_raw)`` (default for debug dumps).

    Parameters
    ----------
    dump:
        Path to ``.pt`` file or dict from ``torch.load``.
    use_qk_l2norm_in_kernel:
        Default ``False`` — dump q/k are pre-normalized like ``test_npu_chunk_kda`` model case.
    use_beta_sigmoid_in_kernel:
        Default auto: ``False`` if beta looks like sigmoid output, else ``True``.
    """
    gpu_root = (fla_root or _GPU_ROOT).resolve()
    _bootstrap_local_fla(gpu_root, force_triton=force_triton)
    _print_fla_import_paths()

    from fla.ops.kda import chunk_kda

    if isinstance(dump, (str, Path)):
        data = load_kda_debug_dump(dump)
    else:
        data = dict(dump)

    dev = _as_device(device)

    q = data["q"].contiguous().to(dev)
    k = data["k"].contiguous().to(dev)
    v = data["v"].contiguous().to(dev)
    g = data["g"].contiguous().to(dev)
    beta = data["beta"].contiguous().to(dev)

    safe_gate = bool(data.get("safe_gate", True))
    lower_bound = float(data.get("lower_bound", -5.0))

    A_log = data.get("A_log")
    dt_bias = data.get("dt_bias")
    if A_log is not None:
        A_log = A_log.contiguous().to(dev).float()
    if dt_bias is not None:
        dt_bias = dt_bias.contiguous().to(dev).float()

    initial_state = data.get("initial_state")
    if initial_state is not None:
        initial_state = initial_state.contiguous().to(dev).float()

    cu = data.get("cu_seqlens")
    cu_seqlens = None
    if cu is not None:
        cu_seqlens = cu.detach().to(device=dev, dtype=torch.int64).contiguous()

    if scale is None:
        scale = float(data.get("scale", q.shape[-1] ** -0.5))

    if use_beta_sigmoid_in_kernel is None:
        use_beta_sigmoid_in_kernel = _guess_beta_sigmoid_in_kernel(beta)

    if verbose:
        print(
            f"[kda_debug_replay] B={q.shape[0]} T={q.shape[1]} Hk={q.shape[2]} Hv={v.shape[2]} "
            f"K={q.shape[3]} V={v.shape[3]} cs={chunk_size} dtype={q.dtype} "
            f"varlen={cu_seqlens is not None} safe_gate={safe_gate} "
            f"l2norm_in_kernel={use_qk_l2norm_in_kernel} beta_sigmoid_in_kernel={use_beta_sigmoid_in_kernel}",
            flush=True,
        )

    call_kw: dict[str, Any] = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        chunk_size=chunk_size,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )
    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("dump has use_gate_in_kernel but missing A_log")
        call_kw["A_log"] = A_log
        if dt_bias is not None:
            call_kw["dt_bias"] = dt_bias
    if initial_state is not None:
        call_kw["initial_state"] = initial_state
    if cu_seqlens is not None:
        call_kw["cu_seqlens"] = cu_seqlens

    _maybe_log_g_from_dump(
        g=g,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        scale=float(scale),
        A_log=A_log if use_gate_in_kernel else None,
        dt_bias=dt_bias if use_gate_in_kernel else None,
        cu_seqlens=cu_seqlens,
    )

    with torch.inference_mode():
        o, final_state = chunk_kda(**call_kw)

    if dev.type == "cuda":
        torch.cuda.synchronize()

    if verbose:
        print(f"[kda_debug_replay] o {_finite_stats(o)} shape={tuple(o.shape)}", flush=True)
        if final_state is not None and final_state.numel() > 0:
            print(
                f"[kda_debug_replay] final_state {_finite_stats(final_state)} "
                f"shape={tuple(final_state.shape)}",
                flush=True,
            )

    return o, final_state


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Run chunk_kda from kda_debug_input_tensors dump")
    p.add_argument("pt_path", nargs="?", type=Path, help="e.g. kda_debug_input_tensors_rank0.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument(
        "--fla-root",
        type=Path,
        default=None,
        help=f"repo root containing fla/ (default: {_GPU_ROOT})",
    )
    p.add_argument(
        "--force-triton",
        action="store_true",
        help="set FLA_FLASH_KDA=0 and FLA_DISABLE_BACKEND_DISPATCH=1 (skip FlashKDA)",
    )
    p.add_argument(
        "--check-imports",
        action="store_true",
        help="only print which fla/ files Python loads, then exit",
    )
    args = p.parse_args()

    try:
        if args.check_imports:
            gpu_root = (args.fla_root or _GPU_ROOT).resolve()
            _bootstrap_local_fla(gpu_root, force_triton=args.force_triton)
            _print_fla_import_paths()
            raise SystemExit(0)

        if args.pt_path is None:
            p.error("pt_path is required unless --check-imports is set")

        run_chunk_kda_from_debug_dump(
            args.pt_path,
            device=args.device,
            chunk_size=args.chunk_size,
            fla_root=args.fla_root,
            force_triton=args.force_triton,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("OK", flush=True)
