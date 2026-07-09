"""Optional g/gk stats logging for KDA forward (env: KDA_FWD_DEBUG_G=1)."""
from __future__ import annotations

import os

import torch

from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2


def kda_fwd_debug_g_enabled() -> bool:
    return os.environ.get("KDA_FWD_DEBUG_G", "0").strip().lower() in ("1", "true", "yes", "on")


def print_tensor_stats(name: str, t: torch.Tensor | None) -> None:
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
            f"finite=0/{n_total} (all non-finite)",
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


def log_kda_g_before_intra(
    *,
    path: str,
    g: torch.Tensor,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    lower_bound: float | None,
    chunk_size: int,
    scale: float,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
) -> None:
    """Log g_raw and gk stats. Pass precomputed ``gk`` when already cumsum'd."""
    if not kda_fwd_debug_g_enabled():
        return

    print(
        f"[KDA_FWD_DEBUG_G] path={path} use_gate_in_kernel={use_gate_in_kernel} "
        f"safe_gate={safe_gate} lower_bound={lower_bound} chunk_size={chunk_size} scale={scale}",
        flush=True,
    )

    if use_gate_in_kernel:
        print_tensor_stats("g_raw (before cumsum)", g)
        if gk is None:
            gk = kda_gate_chunk_cumsum(
                g=g,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=RCP_LN2,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=lower_bound,
            )
    else:
        if gk is None:
            gk = chunk_local_cumsum(
                g=g,
                scale=RCP_LN2,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
    print_tensor_stats("gk (cumsum log2, intra input)", gk)
