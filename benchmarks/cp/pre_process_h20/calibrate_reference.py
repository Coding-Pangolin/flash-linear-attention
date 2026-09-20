#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""02 阶段值域校准: 为 precision-policy.json 定 max_abs_limit / rtol / atol。

做两件事:
1. 量化"契约精度"的影响: 契约标杆(FP32 累加 + 三处舍入点) vs 纯数学 FP64 —— 即
   "如果实现不遵守契约会差多少"
2. 量化"同为 FP32、只是累加顺序不同"的散布(按 16 个 token 分组累加, 模拟 NPU 的
   Cube tile 累加) —— 这正是**任何**正确实现对标杆的预期偏差量级, 直接决定 atol

用法: python calibrate_reference.py --t 11264 --bt 64 --k 128 --v 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _find_reference_dir() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here, here / "reference", here.parent / "reference"):
        if (cand / "reference.py").is_file():
            return cand
    raise SystemExit(f"[FATAL] 找不到 reference.py; 已尝试 {here}, {here / 'reference'}, "
                     f"{here.parent / 'reference'}")


sys.path.insert(0, str(_find_reference_dir()))
from reference import pre_process_fwd_kernel_merged as ref      # noqa: E402


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def build_case(T, BT, HK, HV, K, V, beta_scale=0.02, decay_per_chunk=0.013,
               bg_scale=0.02, seed=0):
    """与 H20 采集脚本同一套输入构造: k 单位化, w = beta*k, bg = gamma*k。"""
    g0 = torch.Generator().manual_seed(seed)
    k = torch.nn.functional.normalize(torch.randn(T, HK, K, generator=g0), dim=-1)
    idx = torch.arange(HV) // (HV // HK)
    beta = torch.rand(T, HV, 1, generator=g0) * beta_scale
    w = beta * k[:, idx, :]
    bg = (torch.rand(T, HK, 1, generator=g0) * bg_scale) * k
    v = torch.randn(T, HV, V, generator=g0)
    u = torch.randn(T, HV, V, generator=g0)

    d = decay_per_chunk / BT
    nt = cdiv(T, BT)
    pad = nt * BT - T

    def decay(extra):
        x = -d * (1.0 + torch.rand(T, *extra, generator=g0) * 0.5)
        if pad:
            x = torch.cat([x, torch.zeros(pad, *extra)], dim=0)
        return x.view(nt, BT, *extra).cumsum(dim=1).reshape(-1, *extra)[:T]

    return dict(k=k.bfloat16(), w=w.bfloat16(), v=v.bfloat16(), u=u.bfloat16(),
                bg=bg.bfloat16(), g=decay((HV,)), gk=decay((HV, K)))


def tiled(k, v, w, gk, bg, BT, K, V, tile=16, accum=torch.float32):
    """与标杆同语义同精度, 但 h 的 token 累加按 `tile` 分组后再相加。

    用来量化"同为 FP32 累加、只是累加顺序不同"的正常散布 —— 这正是 NPU 的
    Cube tile 累加相对标杆的预期偏差量级。
    """
    T = k.shape[0]
    HV, HK = v.shape[1], k.shape[1]
    idx = torch.arange(HV) // (HV // HK)
    use_bg = bg is not None
    dt = accum
    h = torch.zeros(HV, K, V, dtype=dt)
    m = torch.eye(K, dtype=dt).repeat(HV, 1, 1)
    for c in range(cdiv(T, BT)):
        lo, hi = c * BT, min((c + 1) * BT, T)
        o = torch.arange(lo, hi)
        last = hi - 1
        kc = k[o][:, idx, :].to(dt)
        wc = (w[o][:, idx, :] if use_bg else w[o]).to(dt)
        vc = v[o].to(dt)
        # 与 kernel 一致: 先用"未按本 chunk 衰减"的 h 算 v_decay, 再衰减 h
        v_new = vc - torch.einsum("thk,hkv->thv", wc, h.to(k.dtype).to(dt))
        h = h * torch.exp2(gk[last].to(dt))[:, :, None]
        vd = v_new.to(k.dtype).to(dt)
        acc = None
        for t0 in range(0, o.numel(), tile):
            part = torch.einsum("thk,thv->hkv", kc[t0:t0 + tile], vd[t0:t0 + tile])
            acc = part if acc is None else acc + part
        h = h + acc
        kw = torch.einsum("thk,thj->hkj", kc.to(k.dtype).to(dt), wc.to(k.dtype).to(dt))
        M = torch.diag_embed(torch.exp2(gk[last].to(dt))) - kw
        m = ((M @ m).to(torch.float32)).to(dt)
    return torch.cat([h, m], dim=-1)


def rel(a, b):
    d = (a.double() - b.double()).abs()
    return float(d.max()), float(d.max() / (b.double().abs().max() + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=11264)
    ap.add_argument("--bt", type=int, default=64)
    ap.add_argument("--k", type=int, default=128, dest="kdim")
    ap.add_argument("--v", type=int, default=128, dest="vdim")
    a = ap.parse_args()

    print(f"case: T={a.t} BT={a.bt} K={a.kdim} V={a.vdim} "
          f"NT={cdiv(a.t, a.bt)}  beta=0.02 decay/ck=0.013 bg=0.02")
    print(f"{'variant':<16}{'|h|max':>9}{'|m|max':>9}"
          f"{'契约 vs 纯FP64':>24}")
    for hk, hv in ((32, 32), (16, 32)):
        for variant in ("g", "gk", "dplr"):
            c = build_case(a.t, a.bt, hk, hv, a.kdim, a.vdim)
            use_g, use_gk, use_bg = variant == "g", variant != "g", variant == "dplr"
            gate = dict(g=c["g"] if use_g else None, gk=None if use_g else c["gk"])
            kw = dict(bg=c["bg"] if use_bg else None, u=c["u"] if use_bg else None,
                      chunk_size=a.bt, cu_seqlens=(0, a.t))
            base = ref(c["k"], c["v"], c["w"], **gate, **kw)
            pure = ref(c["k"], c["v"], c["w"], **gate, **kw,
                       accum_dtype=torch.float64,
                       round_h_to_input_dtype=False,
                       round_v_new_to_input_dtype=False,
                       round_affine_chain_to_float32=False)
            r1 = rel(base, pure)
            print(f"HK={hk:<2d} HV={hv:<2d} {variant:<6s}"
                  f"{base[:, :, :a.vdim].abs().max():9.3f}"
                  f"{base[:, :, a.vdim:].abs().max():9.4f}"
                  f"   abs={r1[0]:.3e} rel={r1[1]:.2e}")

    # 同为 FP32、累加顺序不同 —— 决定 atol 的那一项
    print("\n同为 FP32 累加、只是把 token 累加按 tile 分组(模拟 NPU Cube 的 tile 累加),"
          " 与契约标杆的偏差:")
    for hk, hv in ((32, 32), (16, 32)):
        c = build_case(a.t, a.bt, hk, hv, a.kdim, a.vdim)
        base = ref(c["k"], c["v"], c["w"], gk=c["gk"],
                   chunk_size=a.bt, cu_seqlens=(0, a.t))
        for tile in (16, 8):
            x = tiled(c["k"], c["v"], c["w"], c["gk"], None, a.bt, a.kdim, a.vdim,
                      tile=tile)
            rh = rel(x[:, :, :a.vdim], base[:, :, :a.vdim])
            rm = rel(x[:, :, a.vdim:], base[:, :, a.vdim:])
            print(f"  HK={hk:<2d} HV={hv:<2d} gk tile={tile:<2d}"
                  f"  h_half abs={rh[0]:.3e} rel={rh[1]:.2e}   "
                  f"m_half abs={rm[0]:.3e} rel={rm[1]:.2e}")


if __name__ == "__main__":
    main()
