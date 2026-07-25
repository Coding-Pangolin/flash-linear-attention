#!/usr/bin/env python3
"""GPU / CPU seed dual with shared CPU-RNG inputs (BTHD).

Seed rule (must match NPU): seed_i = base_seed + case_index * 9973

CPU golden options:
  default          run CPU + GPU
  --no-cpu         GPU only
  --cpu-only       CPU only (no CUDA required)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
GPU_ROOT = SCRIPT_DIR.parent
TEST_DIR = GPU_ROOT / "tests" / "ops"
for p in (str(GPU_ROOT), str(SCRIPT_DIR), str(TEST_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from intra_sub_chunk_case_utils import (  # noqa: E402
    build_intra_sub_chunk_inputs,
    case_seed,
    filter_cases,
    filter_gpu_dump_cases,
    load_cases,
)
from run_intra_sub_chunk_dump_cases import (  # noqa: E402
    run_cpu_intra_sub_chunk,
    run_gpu_intra_sub_chunk,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="intra_sub_chunk seed dual (CPU RNG). Same seeds as NPU seed-dual."
    )
    p.add_argument("--cases-file", type=Path, default=GPU_ROOT / "intra_sub_chunk_cases.json")
    p.add_argument("--phase", default="smoke")
    p.add_argument("--names", default="")
    p.add_argument("--include-disabled", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="base seed; case i → seed+i*9973")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-dtype", default="fp32", choices=("fp32", "fp64"))
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--run-cpu",
        dest="run_cpu",
        action="store_true",
        default=True,
        help="run CPU golden (default: on)",
    )
    g.add_argument(
        "--no-cpu",
        dest="run_cpu",
        action="store_false",
        help="skip CPU golden",
    )
    g.add_argument(
        "--cpu-only",
        action="store_true",
        help="only CPU golden (skip GPU; no CUDA needed)",
    )
    p.add_argument(
        "--save-cpu",
        action="store_true",
        help="save aqk_cpu/akkd_cpu .pt under out-dir/<case>/",
    )
    p.add_argument("--out-dir", type=Path, default=Path("./isub_seed_dual_gpu_out"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_cpu = bool(args.run_cpu or args.cpu_only)
    run_gpu = not bool(args.cpu_only)

    cases = load_cases(args.cases_file)
    names = [n.strip() for n in args.names.split(",") if n.strip()] or None
    selected = filter_cases(
        cases, phase=args.phase, names=names, include_disabled=args.include_disabled
    )
    selected, skipped = filter_gpu_dump_cases(selected)
    print(
        f"selected={len(selected)} gpu_skipped={len(skipped)} seed={args.seed} "
        f"rng=CPU run_cpu={run_cpu} run_gpu={run_gpu}"
    )
    if args.dry_run:
        for i, c in enumerate(selected):
            print(f"  [{i}] {c['name']} seed={case_seed(args.seed, i)}")
        return 0

    if run_gpu and not torch.cuda.is_available():
        print("ERROR: CUDA required (or pass --cpu-only)", file=sys.stderr)
        return 2

    device = torch.device(args.device if run_gpu else "cpu")
    cpu_dtype = torch.float64 if args.cpu_dtype == "fp64" else torch.float32
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    ok = fail = 0

    for i, case in enumerate(selected):
        name = str(case["name"])
        seed_i = case_seed(args.seed, i)
        print(f"[{i+1}/{len(selected)}] {name} seed={seed_i}", flush=True)
        try:
            # Always build with CPU RNG; place on CUDA only if running GPU.
            bundle = build_intra_sub_chunk_inputs(
                case, device=device, seed=seed_i, rng_on_cpu=True
            )
            q, k, g, beta = bundle["q"], bundle["k"], bundle["g"], bundle["beta"]
            scale = float(bundle["scale"])
            BT = int(bundle["chunk_size"])
            cu, idx = bundle["cu_seqlens"], bundle["chunk_indices"]

            aqk_c: Optional[torch.Tensor] = None
            akkd_c: Optional[torch.Tensor] = None
            t_cpu = None
            if run_cpu:
                t0 = time.time()
                aqk_c, akkd_c = run_cpu_intra_sub_chunk(
                    q, k, g, beta, scale, BT, cu, idx, cpu_dtype=cpu_dtype
                )
                t_cpu = time.time() - t0
                if args.save_cpu:
                    cdir = args.out_dir / name
                    cdir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "op": "chunk_kda_fwd_intra_sub_chunk_cpu",
                            "seed": seed_i,
                            "layout": "BTHD",
                            "outputs": {"aqk": aqk_c.cpu(), "akkd": akkd_c.cpu()},
                            "meta": bundle["meta"],
                        },
                        cdir / "cpu_golden.pt",
                    )

            aqk_g: Optional[torch.Tensor] = None
            akkd_g: Optional[torch.Tensor] = None
            if run_gpu:
                with torch.cuda.device(device), torch.inference_mode():
                    aqk_g, akkd_g = run_gpu_intra_sub_chunk(
                        q, k, g, beta, scale, BT, cu, idx
                    )
                torch.cuda.synchronize()

            rec: dict[str, Any] = {
                "name": name,
                "status": "ok",
                "seed": seed_i,
                "run_cpu": run_cpu,
                "run_gpu": run_gpu,
            }
            if t_cpu is not None:
                rec["t_cpu_s"] = round(t_cpu, 4)
            if run_cpu and run_gpu and aqk_c is not None and aqk_g is not None:
                diff_a = (aqk_g.float().cpu() - aqk_c.float()).abs().max().item()
                diff_k = (akkd_g.float().cpu() - akkd_c.float()).abs().max().item()
                rec["aqk_max_abs"] = diff_a
                rec["akkd_max_abs"] = diff_k
                print(
                    f"  OK gpu↔cpu aqk_max_abs={diff_a:.6g} akkd_max_abs={diff_k:.6g} "
                    f"t_cpu={t_cpu:.2f}s",
                    flush=True,
                )
            elif run_cpu:
                print(
                    f"  OK cpu-only aqk={tuple(aqk_c.shape)} akkd={tuple(akkd_c.shape)} "
                    f"t_cpu={t_cpu:.2f}s",
                    flush=True,
                )
            else:
                print(
                    f"  OK gpu-only aqk={tuple(aqk_g.shape)} akkd={tuple(akkd_g.shape)}",
                    flush=True,
                )

            results.append(rec)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            results.append({
                "name": name,
                "status": "fail",
                "error": str(exc),
                "seed": seed_i,
            })
            fail += 1

    report = {
        "mode": "seed_dual_gpu",
        "rng": "cpu",
        "layout": "BTHD",
        "base_seed": args.seed,
        "seed_rule": "base + index * 9973",
        "run_cpu": run_cpu,
        "run_gpu": run_gpu,
        "cpu_dtype": args.cpu_dtype,
        "ok": ok,
        "fail": fail,
        "results": results,
    }
    report_path = args.out_dir / "seed_dual_gpu_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report → {report_path} ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
