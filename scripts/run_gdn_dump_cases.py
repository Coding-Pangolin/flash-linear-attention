#!/usr/bin/env python3
"""Batch GPU GDN operator I/O dump aligned with gpu/cases.json (NPU GVA matrix)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
GPU_ROOT = SCRIPT_DIR.parent
for p in (str(GPU_ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gdn_case_utils import (  # noqa: E402
    build_gdn_inputs,
    case_dump_done,
    filter_cases,
    filter_gpu_dump_cases,
    gpu_dump_skip_reason,
    load_cases,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch dump GPU GDN sub-operator I/O from cases.json")
    p.add_argument(
        "--cases-file",
        type=Path,
        default=GPU_ROOT / "cases.json",
        help="case matrix JSON (default: gpu/cases.json)",
    )
    p.add_argument(
        "--dump-dir",
        type=Path,
        required=True,
        help="root directory for GDN_DUMP_DIR output",
    )
    p.add_argument(
        "--phase",
        default="all",
        help="filter: all | 1 (phase_1_*) | 2/gva (gva_*) | legacy (fix/var_hk_*) | prefix:<name>",
    )
    p.add_argument(
        "--names",
        default="",
        help="comma-separated case names (overrides --phase)",
    )
    p.add_argument(
        "--include-disabled",
        action="store_true",
        help="include cases with enabled=false",
    )
    p.add_argument(
        "--ops",
        default="npu",
        help="GDN_DUMP_OPS: npu (7 NPU ops) | all | comma-separated op names",
    )
    p.add_argument(
        "--dtype-save",
        default="",
        help="GDN_DUMP_DTYPE, e.g. fp32 to unify saved tensors",
    )
    p.add_argument("--seed", type=int, default=0, help="base random seed")
    p.add_argument("--device", default="cuda:0", help="CUDA device")
    p.add_argument("--skip-done", action="store_true", help="skip case if manifest.json exists")
    p.add_argument(
        "--include-unsupported-chunk",
        action="store_true",
        help="do not auto-skip cases whose chunk_size is unsupported on GPU (default: skip)",
    )
    p.add_argument("--dry-run", action="store_true", help="list cases only, no GPU run")
    p.add_argument("--no-bwd", action="store_true", help="forward only (no backward / bwd dumps)")
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="write JSON report (default: <dump-dir>/dump_report.json)",
    )
    return p.parse_args()


def _run_one_case(
    case: dict[str, Any],
    *,
    dump_dir: Path,
    ops: str,
    dtype_save: str,
    device: torch.device,
    seed: int,
    run_bwd: bool,
) -> dict[str, Any]:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.dump import gdn_dump_reset

    t0 = time.time()
    name = str(case["name"])
    gdn_dump_reset()
    os.environ["GDN_DUMP_DIR"] = str(dump_dir)
    os.environ["GDN_DUMP_CASE"] = name
    os.environ["GDN_DUMP_OPS"] = ops
    os.environ["GDN_DUMP_CHUNK_SIZE"] = str(int(case.get("chunk_size", 64)))
    os.environ.pop("GDN_DUMP_EXIT", None)
    if dtype_save:
        os.environ["GDN_DUMP_DTYPE"] = dtype_save
    else:
        os.environ.pop("GDN_DUMP_DTYPE", None)

    bundle = build_gdn_inputs(case, device=device, seed=seed)
    meta = bundle.pop("meta")
    scale = bundle.pop("scale")
    cu_seqlens = bundle.pop("cu_seqlens")

    q = bundle["q"].requires_grad_(True)
    k = bundle["k"].requires_grad_(True)
    v = bundle["v"].requires_grad_(True)
    g = bundle["g"]
    beta = bundle["beta"]

    with torch.cuda.device(device):
        o, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
        )
        if run_bwd:
            o.float().sum().backward()

    out_dir = dump_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "case_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    n_ops = 0
    manifest = out_dir / "manifest.json"
    if manifest.is_file():
        with manifest.open(encoding="utf-8") as f:
            n_ops = len(json.load(f))

    return {
        "name": name,
        "status": "ok",
        "elapsed_s": round(elapsed, 3),
        "n_ops": n_ops,
        "dump_dir": str(out_dir),
    }


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 2

    cases = load_cases(args.cases_file)
    names = [n.strip() for n in args.names.split(",") if n.strip()] or None
    selected = filter_cases(
        cases,
        phase=args.phase,
        names=names,
        include_disabled=args.include_disabled,
    )

    if not selected:
        print("No cases selected.", file=sys.stderr)
        return 1

    gpu_skipped: list[dict[str, str]] = []
    if not args.include_unsupported_chunk:
        selected, gpu_skipped = filter_gpu_dump_cases(selected)

    if not selected:
        print("No GPU-runnable cases after chunk_size filter.", file=sys.stderr)
        for item in gpu_skipped:
            print(f"  SKIP {item['name']}: {item['reason']}", file=sys.stderr)
        return 1

    args.dump_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"cases_file={args.cases_file}")
    print(f"phase={args.phase} selected={len(selected)} dump_dir={args.dump_dir} device={device}")
    if gpu_skipped:
        print(f"gpu_chunk_skip={len(gpu_skipped)} (chunk_size!=64, no GPU dual-benchmark)")
    if args.dry_run:
        for c in selected:
            flag = "enabled" if c.get("enabled", True) else "disabled"
            cs = c.get("chunk_size", 64)
            print(f"  [RUN/{flag}] {c['name']} chunk_size={cs}: {c.get('description', '')[:60]}")
        for item in gpu_skipped:
            print(f"  [SKIP/gpu] {item['name']}: {item['reason']}")
        return 0

    report_path = args.report or (args.dump_dir / "dump_report.json")
    results: list[dict[str, Any]] = []
    ok, skip, fail = 0, 0, 0

    for i, case in enumerate(selected):
        name = str(case["name"])
        case_seed = args.seed + i * 9973

        if args.skip_done and case_dump_done(args.dump_dir, name):
            print(f"[{i+1}/{len(selected)}] SKIP {name} (manifest exists)")
            results.append({"name": name, "status": "skip"})
            skip += 1
            continue

        skip_reason = gpu_dump_skip_reason(case)
        if skip_reason and not args.include_unsupported_chunk:
            print(f"[{i+1}/{len(selected)}] SKIP {name} ({skip_reason})")
            results.append({"name": name, "status": "skip_gpu", "reason": skip_reason})
            skip += 1
            continue

        print(f"[{i+1}/{len(selected)}] RUN  {name} ...", flush=True)
        try:
            rec = _run_one_case(
                case,
                dump_dir=args.dump_dir,
                ops=args.ops,
                dtype_save=args.dtype_save,
                device=device,
                seed=case_seed,
                run_bwd=not args.no_bwd,
            )
            results.append(rec)
            ok += 1
            print(f"         OK  ops={rec['n_ops']} elapsed={rec['elapsed_s']}s")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            msg = "CUDA OOM"
            print(f"         FAIL {msg}")
            results.append({"name": name, "status": "oom", "error": msg})
            fail += 1
        except Exception as e:
            print(f"         FAIL {e}")
            results.append({
                "name": name,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            fail += 1

    summary = {
        "cases_file": str(args.cases_file),
        "phase": args.phase,
        "dump_dir": str(args.dump_dir),
        "total": len(selected) + len(gpu_skipped),
        "gpu_runnable": len(selected),
        "gpu_chunk_skipped": gpu_skipped,
        "ok": ok,
        "skip": skip,
        "fail": fail,
        "results": results,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone: ok={ok} skip={skip} fail={fail} report={report_path}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
