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
from pathlib import Path
from typing import Any, Mapping, Union

import torch

DumpLike = Union[str, Path, Mapping[str, Any]]


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
    from fla.ops.kda import chunk_kda
    from fla.ops.kda import chunk_fwd as _chunk_fwd_mod
    from fla.ops.kda.debug_g import kda_fwd_debug_g_enabled

    if kda_fwd_debug_g_enabled():
        print(
            f"[KDA_FWD_DEBUG_G] run_chunk_kda_from_debug_dump: "
            f"KDA_FWD_DEBUG_G={os.environ.get('KDA_FWD_DEBUG_G')} "
            f"FLA_FLASH_KDA={os.environ.get('FLA_FLASH_KDA', '(default)')} "
            f"chunk_fwd={_chunk_fwd_mod.__file__}",
            flush=True,
        )

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
    p.add_argument("pt_path", type=Path, help="e.g. kda_debug_input_tensors_rank0.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--chunk-size", type=int, default=64)
    args = p.parse_args()

    try:
        run_chunk_kda_from_debug_dump(
            args.pt_path,
            device=args.device,
            chunk_size=args.chunk_size,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("OK", flush=True)
