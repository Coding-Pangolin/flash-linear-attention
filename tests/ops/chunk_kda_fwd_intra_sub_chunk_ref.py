#!/usr/bin/env python3
# Copyright (c) 2026 Tianjin University, Ltd.
"""Golden for ChunkKdaFwdIntraSubChunk (BNSD, MHA only).

Default compute dtype is float64. For large model shapes use dtype=torch.float32 —
NPU accumulates in fp32, and fp32↔fp64 already differs by ~1e-2 on Akkd at scale.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


BC = 16


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> list[int]:
    indices: list[int] = []
    for seq in range(int(cu_seqlens.numel()) - 1):
        length = int(cu_seqlens[seq + 1] - cu_seqlens[seq])
        n_chunks = (length + chunk_size - 1) // chunk_size
        for local in range(n_chunks):
            indices.extend([seq, local])
    return indices


def _forward_sub_inv(L: torch.Tensor) -> torch.Tensor:
    """(I-L)^{-1} via Triton-style forward substitution. L is [..., valid, valid]."""
    valid = L.shape[-1]
    M = -L.clone()
    for i in range(2, valid):
        a = M[..., i, :].clone()
        a[..., i:] = 0
        a = a + (a.unsqueeze(-2) @ M).squeeze(-2)
        a[..., i:] = 0
        M[..., i, :] = a
    eye = torch.eye(valid, dtype=M.dtype, device=M.device)
    return M + eye


def chunk_kda_fwd_intra_sub_chunk_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference matching Triton safe-gate diagonal path. Inputs are BNSD."""
    assert q.ndim == 4 and q.shape == k.shape == g.shape
    B, H, T, K = q.shape
    assert beta.shape == (B, H, T)
    assert chunk_size in (32, 64, 128)
    BT = chunk_size
    NC = BT // BC

    q_r = q.to(dtype)
    k_r = k.to(dtype)
    g_r = g.to(dtype)
    beta_r = beta.to(dtype)

    aqk = torch.zeros(B, H, T, BT, dtype=dtype, device=q.device)
    akkd = torch.zeros(B, H, T, BC, dtype=dtype, device=q.device)

    if cu_seqlens is None:
        # Dense: batch over heads for speed on large H/T.
        nt = (T + BT - 1) // BT
        for i_b in range(B):
            for i_t in range(nt):
                for i_i in range(NC):
                    i_ti = i_t * BT + i_i * BC
                    if i_ti >= T:
                        continue
                    valid = min(BC, T - i_ti)
                    mid = i_ti + min(BC // 2, T - i_ti - 1)
                    rows = list(range(i_ti, i_ti + valid))

                    g_block = g_r[i_b, :, rows, :]  # [H, valid, K]
                    g_mid = g_r[i_b, :, mid, :]  # [H, K]
                    gm = g_block - g_mid[:, None, :]
                    gq = torch.exp2(gm)
                    gk = torch.exp2(-gm)
                    q_block = q_r[i_b, :, rows, :] * gq
                    k_pos = k_r[i_b, :, rows, :] * gq
                    k_neg = k_r[i_b, :, rows, :] * gk
                    beta_row = beta_r[i_b, :, rows]  # [H, valid]

                    aqk_blk = torch.matmul(q_block, k_neg.transpose(-1, -2)) * scale
                    akk_blk = torch.matmul(k_pos, k_neg.transpose(-1, -2)) * beta_row[:, :, None]

                    tril = torch.tril(torch.ones(valid, valid, dtype=torch.bool, device=q.device))
                    strict = torch.tril(
                        torch.ones(valid, valid, dtype=torch.bool, device=q.device), diagonal=-1
                    )
                    aqk_blk = torch.where(tril, aqk_blk, torch.zeros_like(aqk_blk))
                    L = torch.where(strict, akk_blk, torch.zeros_like(akk_blk))
                    M = _forward_sub_inv(L)

                    aqk[i_b, :, rows, i_i * BC : i_i * BC + valid] = aqk_blk[:, :, :valid]
                    akkd[i_b, :, rows, :valid] = M[:, :, :valid]
                    if valid < BC:
                        akkd[i_b, :, rows, valid:] = 0
        return aqk, akkd

    # Varlen path (B=1): keep per-head loop.
    assert B == 1
    if chunk_indices is None:
        flat = prepare_chunk_indices(cu_seqlens, BT)
        chunk_indices = torch.tensor(flat, dtype=torch.long, device=q.device)
    assert chunk_indices.numel() % 2 == 0

    for nt_i in range(chunk_indices.numel() // 2):
        seq = int(chunk_indices[nt_i * 2])
        local = int(chunk_indices[nt_i * 2 + 1])
        bos = int(cu_seqlens[seq])
        eos = int(cu_seqlens[seq + 1])
        local_t = eos - bos
        for i_i in range(NC):
            i_ti = local * BT + i_i * BC
            if i_ti >= local_t:
                continue
            valid = min(BC, local_t - i_ti)
            mid = i_ti + min(BC // 2, local_t - i_ti - 1)
            rows = [bos + i_ti + r for r in range(valid)]

            g_block = g_r[0, :, rows, :]
            g_mid = g_r[0, :, bos + mid, :]
            gm = g_block - g_mid[:, None, :]
            gq = torch.exp2(gm)
            gk = torch.exp2(-gm)
            q_block = q_r[0, :, rows, :] * gq
            k_pos = k_r[0, :, rows, :] * gq
            k_neg = k_r[0, :, rows, :] * gk
            beta_row = beta_r[0, :, rows]

            aqk_blk = torch.matmul(q_block, k_neg.transpose(-1, -2)) * scale
            akk_blk = torch.matmul(k_pos, k_neg.transpose(-1, -2)) * beta_row[:, :, None]
            tril = torch.tril(torch.ones(valid, valid, dtype=torch.bool, device=q.device))
            strict = torch.tril(
                torch.ones(valid, valid, dtype=torch.bool, device=q.device), diagonal=-1
            )
            aqk_blk = torch.where(tril, aqk_blk, torch.zeros_like(aqk_blk))
            L = torch.where(strict, akk_blk, torch.zeros_like(akk_blk))
            M = _forward_sub_inv(L)

            aqk[0, :, rows, i_i * BC : i_i * BC + valid] = aqk_blk[:, :, :valid]
            akkd[0, :, rows, :valid] = M[:, :, :valid]
            if valid < BC:
                akkd[0, :, rows, valid:] = 0

    return aqk, akkd


def _self_check():
    torch.manual_seed(0)
    B, H, T, K, BT = 1, 2, 64, 32, 64
    q = torch.randn(B, H, T, K)
    k = torch.randn(B, H, T, K)
    g = -torch.linspace(0, 40, T).view(1, 1, T, 1).expand(B, H, T, K).contiguous()
    beta = torch.rand(B, H, T)
    scale = 1.0 / math.sqrt(K)
    aqk, akkd = chunk_kda_fwd_intra_sub_chunk_ref(q, k, g, beta, scale, BT)
    assert aqk.shape == (B, H, T, BT)
    assert akkd.shape == (B, H, T, BC)
    assert torch.isfinite(aqk).all() and torch.isfinite(akkd).all()
    for t0 in range(0, T, BC):
        diag = torch.stack([akkd[0, 0, t0 + i, i] for i in range(BC)])
        assert torch.allclose(diag, torch.ones(BC, dtype=torch.float64), atol=1e-5, rtol=1e-5)
    print("golden self-check passed", aqk.abs().mean().item(), akkd.abs().mean().item())


if __name__ == "__main__":
    _self_check()
