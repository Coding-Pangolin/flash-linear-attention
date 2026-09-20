#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""02 阶段值域校准: 为 precision-policy.json 定 max_abs_limit / rtol。

做两件事:
1. 量化"契约舍入点"的影响: 标杆(复刻三个舍入点, fp64 累加) vs 纯 fp64
2. 与一份**独立实现**(float32 累加、不同代码路径)对齐, 量化"复刻了舍入点但累加顺序
   不同"的实现与标杆之间正常的偏差量级 —— 这正是 NPU 实现对标杆的预期偏差

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


def independent(k, v, w, u, g, gk, bg, BT, K, V, use_g, use_gk, use_bg):
    """独立实现: float32 累加, 与 reference.py 不同的循环/广播写法。"""
    T = k.shape[0]
    HV, HK = v.shape[1], k.shape[1]
    idx = torch.arange(HV) // (HV // HK)
    h = torch.zeros(HV, K, V, dtype=torch.float32)
    m = torch.eye(K, dtype=torch.float32).repeat(HV, 1, 1)
    for c in range(cdiv(T, BT)):
        lo, hi = c * BT, min((c + 1) * BT, T)
        o = torch.arange(lo, hi)
        last = hi - 1
        kc = k[o][:, idx, :].float()
        wc = (w[o][:, idx, :] if use_bg else w[o]).float()
        vc, uc = v[o].float(), u[o].float()
        v_new = (torch.einsum("thk,hkv->thv", wc, h.to(torch.bfloat16).float()) + uc
                 if use_bg else
                 vc - torch.einsum("thk,hkv->thv", wc, h.to(torch.bfloat16).float()))
        if use_g:
            gl, gs = g[last], g[o]
            v_new = v_new * torch.exp2(gl[None, :] - gs).unsqueeze(-1)
            h = h * torch.exp2(gl)[:, None, None]
        if use_gk:
            h = h * torch.exp2(gk[last])[:, :, None]
        h = h + torch.einsum("thk,thv->hkv", kc, v_new.to(torch.bfloat16).float())
        if use_bg:
            h = h + torch.einsum("thk,thv->hkv", bg[o][:, idx, :].float(),
                                 vc.to(torch.bfloat16).float())
        left = bg[o][:, idx, :].float() if use_bg else kc
        if use_g:
            left = left * torch.exp2(g[last][None, :] - g[o]).unsqueeze(-1)
        kw = torch.einsum("thk,thj->hkj", left.to(torch.bfloat16).float(),
                          wc.to(torch.bfloat16).float())
        if use_g:
            diag = torch.eye(K).repeat(HV, 1, 1) * torch.exp2(g[last])[:, None, None]
        else:
            diag = torch.diag_embed(torch.exp2(gk[last]))
        M = (diag + kw) if use_bg else (diag - kw)
        m = M @ m
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
          f"{'contract vs pure fp64':>26}{'independent impl':>26}")
    for hk, hv in ((32, 32), (16, 32)):
        for variant in ("g", "gk", "dplr"):
            c = build_case(a.t, a.bt, hk, hv, a.kdim, a.vdim)
            use_g, use_gk, use_bg = variant == "g", variant != "g", variant == "dplr"
            gate = dict(g=c["g"] if use_g else None, gk=None if use_g else c["gk"])
            kw = dict(bg=c["bg"] if use_bg else None, u=c["u"] if use_bg else None,
                      chunk_size=a.bt, cu_seqlens=(0, a.t))
            base = ref(c["k"], c["v"], c["w"], **gate, **kw)
            pure = ref(c["k"], c["v"], c["w"], **gate, **kw,
                       round_h_to_input_dtype=False,
                       round_v_new_to_input_dtype=False,
                       round_affine_chain_to_float32=False)
            ind = independent(c["k"], c["v"], c["w"], c["u"], c["g"], c["gk"], c["bg"],
                              a.bt, a.kdim, a.vdim, use_g, use_gk, use_bg)
            r1, r2 = rel(base, pure), rel(base, ind)
            print(f"HK={hk:<2d} HV={hv:<2d} {variant:<6s}"
                  f"{base[:, :, :a.vdim].abs().max():9.3f}"
                  f"{base[:, :, a.vdim:].abs().max():9.4f}"
                  f"   abs={r1[0]:.3e} rel={r1[1]:.2e}"
                  f"    abs={r2[0]:.3e} rel={r2[1]:.2e}")


if __name__ == "__main__":
    main()
