#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 H20 采集的 case.pt 与 CPU 标杆对齐。

用途: 02 阶段"标杆对齐"。`benchmarks/cp/bench_pre_process_h20.py --save-io` 导出的
`case.pt` 里同时含输入和 H20 上算出的 `hm`; 本脚本在 CPU 上用唯一标杆源码对同一输入
重算, 再按工作流的精度策略比较。

用法
----
    python scripts/compare_with_gpu.py \
        --case  /path/to/case_gk/case.pt \
        --policy reference/precision-policy.json \
        --skill-dir /path/to/cannbot-skills/ops/catlass-linear-attention-workflow \
        --precision ieee

关于 --precision
----------------
它填 GPU 侧采集时用的 AFFINE_CHAIN_PRECISION, 只影响对 m 半边偏差的**预期**:
* `ieee`    : m 的链式乘按真 FP32, 与标杆一致 -> 可以按策略严格比较
* `tf32x3`  : 近似 FP32, 偏差较小
* `default` : NVIDIA 上是 TF32(尾数 10 位), m 的链乘 176 次会累积到 ~1e-1 相对误差,
              必然超出策略 -> 这时只有 h 半边的结论有效
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _find_reference_dir() -> Path:
    """兼容两种布局: 与 reference.py 同目录, 或 reference/ 子目录。"""
    here = Path(__file__).resolve().parent
    for cand in (here, here / "reference", here.parent / "reference"):
        if (cand / "reference.py").is_file():
            return cand
    raise SystemExit(
        f"[FATAL] 找不到 reference.py; 已尝试 {here}, {here / 'reference'}, "
        f"{here.parent / 'reference'}"
    )


def _variant_of(case) -> str:
    if case.get("g") is not None:
        return "g"
    if case.get("bg") is not None:
        return "dplr"
    return "gk"


def _masks(hv: int, k: int, v: int):
    cols = np.arange(k + v)[None, None, :]
    heads = np.arange(hv)[:, None, None]
    return {
        "h_half": np.broadcast_to(cols < v, (hv, k, k + v)).copy(),
        "m_half": np.broadcast_to(cols >= v, (hv, k, k + v)).copy(),
        "head0": np.broadcast_to(heads == 0, (hv, k, k + v)).copy(),
        "head_last": np.broadcast_to(heads == hv - 1, (hv, k, k + v)).copy(),
    }


def _metrics(actual: np.ndarray, golden: np.ndarray, atol: float, rtol: float) -> dict:
    a = actual.astype(np.float64)
    g = golden.astype(np.float64)
    diff = np.abs(a - g)
    close = diff <= (atol + rtol * np.abs(g))
    rel = diff / (np.abs(g) + 1e-12)
    return dict(elements=int(diff.size),
                matched_ratio=float(close.mean()),
                error_count=int((~close).sum()),
                max_abs=float(diff.max()),
                mare=float(rel.max()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--skill-dir", type=Path, default=None,
                    help="cannbot-skills 的 ops/catlass-linear-attention-workflow 目录")
    ap.add_argument("--precision", default="default",
                    choices=("default", "tf32x3", "ieee"))
    ap.add_argument("--device", default="cpu",
                    help="标杆执行设备; 正式验收用 cpu")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    case = torch.load(args.case, map_location="cpu", weights_only=True)
    missing = [name for name in ("k", "v", "w", "hm") if case.get(name) is None]
    if missing:
        print(f"[FATAL] case.pt 缺少 {missing}; 请用 --save-io 重新导出")
        return 2

    # case.json 是 --save-io 一起写出的元数据; 用它的 variant/half 而不是靠输入猜测
    meta = None
    meta_path = args.case.with_name("case.json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        half = meta.get("half", "both")
        if half != "both":
            print(f"[FATAL] 该 case 是用 --half {half} 采集的, 输出布局不完整, 无法对齐。")
            print("        请在 H20 上用默认(--half both)重采后加 --save-io。")
            return 2

    hm_gpu = case["hm"]
    if not torch.isfinite(hm_gpu).all():
        print("[FATAL] case.pt 里的 hm 含非有限值(该 case 发散), 无法对齐。")
        print("        请在 H20 上用修好的脚本重跑并加 --save-io。")
        return 2

    variant = (meta or {}).get("variant") or _variant_of(case)
    hv, k, v_plus_k = hm_gpu.shape
    v = case["v"].shape[2]
    k_dim = case["k"].shape[2]
    if k_dim != k or v + k_dim != v_plus_k:
        print(f"[FATAL] 形状不自洽: hm={tuple(hm_gpu.shape)} k.K={k_dim} v.V={v}")
        return 2

    sys.path.insert(0, str(_find_reference_dir()))
    from reference import pre_process_fwd_kernel_merged as ref      # noqa: E402

    cu = case.get("cu_seqlens")
    dev = torch.device(args.device)
    hm_ref = ref(
        case["k"].to(dev), case["v"].to(dev), case["w"].to(dev),
        g=None if case.get("g") is None else case["g"].to(dev),
        gk=None if case.get("gk") is None else case["gk"].to(dev),
        bg=None if case.get("bg") is None else case["bg"].to(dev),
        u=None if case.get("u") is None else case["u"].to(dev),
        chunk_size=int(case.get("BT", 64) or 64),
        cu_seqlens=None if cu is None else (int(cu[0]), int(cu[1])),
    )

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    cfg = policy["dtype"]["float32"]
    print(f"标杆执行设备: {args.device}")
    masks = _masks(hv, k_dim, v)
    ones = np.ones((hv, k_dim, v_plus_k), dtype=bool)
    golden_np = hm_ref.detach().cpu().numpy()
    actual_np = hm_gpu.detach().cpu().to(torch.float32).numpy()

    print("=" * 78)
    print(f"标杆对齐: {args.case}")
    print(f"variant={variant}  hv={hv} K={k_dim} V={v}  T={case['k'].shape[0]}  "
          f"gpu_precision={args.precision}")
    print(f"policy: atol={cfg['atol']} rtol={cfg['rtol']} "
          f"max_abs_limit={cfg['max_abs_limit']}")
    print("=" * 78)

    print(f"\n{'region':<12}{'elements':>10}{'matched':>10}{'err':>8}"
          f"{'max_abs':>12}{'MARE':>12}")
    overall = _metrics(actual_np, golden_np, cfg["atol"], cfg["rtol"])
    for name, mask in (("ALL", ones), ("h_half", masks["h_half"]),
                       ("m_half", masks["m_half"]), ("head0", masks["head0"]),
                       ("head_last", masks["head_last"])):
        sel = mask & ones
        m = _metrics(actual_np[sel], golden_np[sel], cfg["atol"], cfg["rtol"])
        print(f"{name:<12}{m['elements']:>10}{m['matched_ratio']:>10.6f}"
              f"{m['error_count']:>8}{m['max_abs']:>12.3e}{m['mare']:>12.3e}")

    ok = (overall["matched_ratio"] >= policy["global_matched_ratio"]
          and overall["max_abs"] <= cfg["max_abs_limit"])
    print(f"\n概览: matched_ratio={overall['matched_ratio']:.6f} "
          f"(要求 >= {policy['global_matched_ratio']})  "
          f"max_abs={overall['max_abs']:.3e} (限 {cfg['max_abs_limit']})  "
          f"-> {'PASS' if ok else 'FAIL'}")

    if args.precision != "ieee" and not ok:
        print("\n[提示] GPU 侧用的是 "
              f"{'TF32(default)' if args.precision == 'default' else 'tf32x3'}, "
              "m 半边是 176 次链式乘, 低精度会累积成显著偏差。")
        print("       若 h_half 通过而 m_half 不通过, 属于预期现象; "
              "要做严格对齐请用 --precision ieee 重采。")

    if args.skill_dir is not None:
        sys.path.insert(0, str(args.skill_dir / "scripts"))
        from compare_precision import compare_case                    # noqa: E402

        report = compare_case(
            {"hm": actual_np}, {"hm": golden_np},
            case=f"gpu_align_{variant}_{args.precision}",
            expected_outputs={"hm": {"shape": [hv, k_dim, v_plus_k],
                                     "dtype": "float32",
                                     "required_regions": ["h_half", "m_half",
                                                          "head0", "head_last"]}},
            policy=policy,
            valid_masks={"hm": ones},
            regions={"hm": masks},
        )
        print(f"\ncompare_case: status={report['status']}")
        if args.report is not None:
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
            print(f"报告已写入 {args.report}")
        return 0 if report["pass"] else 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
