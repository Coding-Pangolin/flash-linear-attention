#!/usr/bin/env python3
"""Batch GPU KDA chunk_kda I/O dump from kda_cases.json."""
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

from kda_case_utils import (  # noqa: E402
    build_kda_inputs,
    case_dump_done,
    filter_cases,
    filter_kda_dump_cases,
    kda_dump_skip_reason,
    load_cases,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch dump GPU chunk_kda I/O from kda_cases.json")
    p.add_argument(
        "--cases-file",
        type=Path,
        default=GPU_ROOT / "kda_cases.json",
        help="case matrix JSON (default: gpu/kda_cases.json)",
    )
    p.add_argument("--dump-dir", type=Path, required=True, help="root directory for KDA_DUMP_DIR output")
    p.add_argument(
        "--phase",
        default="all",
        help="filter: all | smoke (smoke_*) | prefix:<name>",
    )
    p.add_argument("--names", default="", help="comma-separated case names (overrides --phase)")
    p.add_argument("--include-disabled", action="store_true")
    p.add_argument("--ops", default="fwd", help="KDA_DUMP_OPS: fwd (default) | all | comma-separated")
    p.add_argument("--dtype-save", default="", help="KDA_DUMP_DTYPE, e.g. fp32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="write JSON report (default: <dump-dir>/kda_dump_report.json)",
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
) -> dict[str, Any]:
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.dump import kda_dump_reset

    t0 = time.time()
    name = str(case["name"])
    kda_dump_reset()
    os.environ["KDA_DUMP_DIR"] = str(dump_dir)
    os.environ["KDA_DUMP_CASE"] = name
    os.environ["KDA_DUMP_OPS"] = ops
    os.environ.pop("KDA_DUMP_EXIT", None)
    if dtype_save:
        os.environ["KDA_DUMP_DTYPE"] = dtype_save
    else:
        os.environ.pop("KDA_DUMP_DTYPE", None)

    bundle = build_kda_inputs(case, device=device, seed=seed)
    meta = bundle.pop("meta")
    flags = bundle.pop("flags")
    scale = bundle.pop("scale")
    chunk_size = bundle.pop("chunk_size")
    cu_seqlens = bundle.pop("cu_seqlens")
    A_log = bundle.pop("A_log")
    dt_bias = bundle.pop("dt_bias")
    initial_state = bundle.pop("initial_state")

    call_kwargs: dict[str, Any] = {
        **bundle,
        "scale": scale,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "chunk_size": chunk_size,
        **flags,
    }
    if A_log is not None:
        call_kwargs["A_log"] = A_log
    if dt_bias is not None:
        call_kwargs["dt_bias"] = dt_bias

    with torch.cuda.device(device), torch.inference_mode():
        o, final_state = chunk_kda(**call_kwargs)

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
        "o_shape": list(o.shape),
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
    selected, gpu_skipped = filter_kda_dump_cases(selected)

    if not selected:
        print("No KDA-runnable cases selected.", file=sys.stderr)
        for item in gpu_skipped:
            print(f"  SKIP {item['name']}: {item['reason']}", file=sys.stderr)
        return 1

    args.dump_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"cases_file={args.cases_file}")
    print(f"phase={args.phase} selected={len(selected)} dump_dir={args.dump_dir} device={device}")
    if gpu_skipped:
        print(f"skipped={len(gpu_skipped)}")

    if args.dry_run:
        for c in selected:
            print(f"  [RUN] {c['name']}: {c.get('description', '')[:70]}")
        for item in gpu_skipped:
            print(f"  [SKIP] {item['name']}: {item['reason']}")
        return 0

    report_path = args.report or (args.dump_dir / "kda_dump_report.json")
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

        skip_reason = kda_dump_skip_reason(case)
        if skip_reason:
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
            )
            results.append(rec)
            ok += 1
            print(f"         OK  ops={rec['n_ops']} o={rec['o_shape']} elapsed={rec['elapsed_s']}s")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("         FAIL CUDA OOM")
            results.append({"name": name, "status": "oom", "error": "CUDA OOM"})
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
        "ok": ok,
        "skip": skip,
        "fail": fail,
        "skipped": gpu_skipped,
        "results": results,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone: ok={ok} skip={skip} fail={fail} report={report_path}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
