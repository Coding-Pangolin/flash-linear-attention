#!/usr/bin/env python3
"""Batch GPU dump for chunk_kda_fwd_kernel_intra_sub_chunk (GDN isub dual-benchmark)."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import torch
import triton

SCRIPT_DIR = Path(__file__).resolve().parent
GPU_ROOT = SCRIPT_DIR.parent
for p in (str(GPU_ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from intra_sub_chunk_case_utils import (  # noqa: E402
    BC,
    build_intra_sub_chunk_inputs,
    case_dump_done,
    filter_cases,
    filter_gpu_dump_cases,
    gpu_dump_skip_reason,
    load_cases,
)

OP_NAME = "chunk_kda_fwd_intra_sub_chunk"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch dump GPU chunk_kda_fwd_intra_sub_chunk I/O from intra_sub_chunk_cases.json"
    )
    p.add_argument(
        "--cases-file",
        type=Path,
        default=GPU_ROOT / "intra_sub_chunk_cases.json",
    )
    p.add_argument("--dump-dir", type=Path, required=True)
    p.add_argument(
        "--phase",
        default="all",
        help="all | smoke | gdn | varlen | gva | prefix:<name>",
    )
    p.add_argument("--names", default="", help="comma-separated case names (overrides --phase)")
    p.add_argument("--include-disabled", action="store_true")
    p.add_argument(
        "--dtype-save",
        default="",
        help="set fp32 to unify saved floating tensors",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="default: <dump-dir>/intra_sub_chunk_dump_report.json",
    )
    return p.parse_args()


def _kernel_arg_names(kernel) -> set[str]:
    cur = kernel
    names: set[str] = set()
    for _ in range(6):
        if hasattr(cur, "arg_names") and cur.arg_names:
            names.update(cur.arg_names)
        if hasattr(cur, "fn"):
            cur = cur.fn
            continue
        break
    return names


def run_gpu_intra_sub_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch Triton intra_sub_chunk; supports GVA (HV != H)."""
    import fla
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_kernel_intra_sub_chunk
    from fla.ops.utils import prepare_chunk_indices as fla_prepare_chunk_indices
    from fla.utils import IS_GATHER_SUPPORTED

    B, T, H, K = k.shape
    HV = g.shape[2]
    BT = chunk_size
    if BT not in (32, 64):
        raise ValueError(f"GPU kernel only supports chunk_size 32/64, got {BT}")
    if HV < H or HV % H != 0:
        raise ValueError(f"illegal GVA: H={H} HV={HV}")

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = fla_prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    BK = triton.next_power_of_2(K)

    Aqk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akkd = torch.zeros(B, T, HV, BC, device=k.device, dtype=torch.float32)

    grid = (NT, NC, B * HV)
    kwargs = dict(
        q=q,
        k=k,
        g=g,
        beta=beta,
        Aqk=Aqk,
        Akk=Akkd,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        USE_GATHER=IS_GATHER_SUPPORTED,
    )
    arg_names = _kernel_arg_names(chunk_kda_fwd_kernel_intra_sub_chunk)
    if "HV" in arg_names:
        kwargs["HV"] = HV
    elif H != HV:
        raise RuntimeError(
            f"imported fla kernel has no HV (file={fla.__file__}) but H={H} != HV={HV}; "
            "use this checkout: pip install -e ."
        )

    chunk_kda_fwd_kernel_intra_sub_chunk[grid](**kwargs)
    torch.cuda.synchronize()
    return Aqk, Akkd


def _to_cpu(x: Any, save_fp32: bool) -> Any:
    if x is None:
        return None
    if not isinstance(x, torch.Tensor):
        return x
    t = x.detach()
    if save_fp32 and t.is_floating_point():
        t = t.float()
    return t.cpu()


def _save_dump(
    *,
    out_dir: Path,
    case_meta: dict[str, Any],
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    save_fp32: bool,
    fla_file: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "case_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(case_meta, f, indent=2, ensure_ascii=False)

    packed_in = {k: _to_cpu(v, save_fp32) for k, v in inputs.items() if v is not None}
    packed_out = {k: _to_cpu(v, save_fp32) for k, v in outputs.items() if v is not None}
    payload = {
        "op": OP_NAME,
        "step": 1,
        "layout": {
            "storage": "BTHD",
            "note": (
                "q/k [B,T,H,K]; g/aqk [B,T,HV,BT]; akkd [B,T,HV,BC] fp32; "
                "beta [B,T,HV]. NPU: transpose(1,2) → BNSD; beta [B,T,HV]→[B,HV,T]."
            ),
        },
        "inputs": packed_in,
        "outputs": packed_out,
        "meta": {
            **case_meta,
            "fla_file": fla_file,
            "save_fp32": save_fp32,
            "BC": BC,
        },
    }
    pt_name = f"001_{OP_NAME}.pt"
    pt_path = out_dir / pt_name
    torch.save(payload, pt_path)

    manifest = [{"step": 1, "op": OP_NAME, "path": pt_name}]
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return pt_path


def _run_one_case(
    case: dict[str, Any],
    *,
    dump_dir: Path,
    device: torch.device,
    seed: int,
    dtype_save: str,
) -> dict[str, Any]:
    import fla

    t0 = time.time()
    name = str(case["name"])
    save_fp32 = dtype_save.lower() in ("fp32", "float32", "float")

    bundle = build_intra_sub_chunk_inputs(case, device=device, seed=seed)
    meta = bundle["meta"]
    q, k, g, beta = bundle["q"], bundle["k"], bundle["g"], bundle["beta"]
    scale = float(bundle["scale"])
    BT = int(bundle["chunk_size"])
    cu = bundle["cu_seqlens"]
    idx = bundle["chunk_indices"]

    with torch.cuda.device(device), torch.inference_mode():
        aqk, akkd = run_gpu_intra_sub_chunk(
            q, k, g, beta, scale, BT, cu_seqlens=cu, chunk_indices=idx
        )

    out_dir = dump_dir / name
    pt_path = _save_dump(
        out_dir=out_dir,
        case_meta=meta,
        inputs={
            "q": q,
            "k": k,
            "g": g,
            "beta": beta,
            "scale": scale,
            "cu_seqlens": cu,
            "chunk_indices": idx,
            "chunk_size": BT,
        },
        outputs={"aqk": aqk, "akkd": akkd},
        save_fp32=save_fp32,
        fla_file=str(fla.__file__),
    )
    elapsed = time.time() - t0
    return {
        "name": name,
        "status": "ok",
        "elapsed_s": round(elapsed, 3),
        "aqk_shape": list(aqk.shape),
        "akkd_shape": list(akkd.shape),
        "dump_pt": str(pt_path),
        "dump_dir": str(out_dir),
    }


def main() -> int:
    args = _parse_args()

    cases = load_cases(args.cases_file)
    names = [n.strip() for n in args.names.split(",") if n.strip()] or None
    selected = filter_cases(
        cases,
        phase=args.phase,
        names=names,
        include_disabled=args.include_disabled,
    )
    selected, gpu_skipped = filter_gpu_dump_cases(selected)

    print(f"cases_file={args.cases_file}")
    print(
        f"phase={args.phase} selected={len(selected)} "
        f"gpu_skipped={len(gpu_skipped)} dump_dir={args.dump_dir} device={args.device}"
    )

    if args.dry_run:
        for c in selected:
            print(f"  [RUN] {c['name']}: {c.get('description', '')[:80]}")
        for item in gpu_skipped:
            print(f"  [SKIP] {item['name']}: {item['reason']}")
        return 0

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available (use --dry-run to preview cases)", file=sys.stderr)
        return 2

    if not selected:
        print("No GPU-runnable cases selected.", file=sys.stderr)
        for item in gpu_skipped:
            print(f"  SKIP {item['name']}: {item['reason']}", file=sys.stderr)
        return 1

    args.dump_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    report_path = args.report or (args.dump_dir / "intra_sub_chunk_dump_report.json")
    results: list[dict[str, Any]] = []
    ok = skip = fail = 0

    for i, case in enumerate(selected):
        name = str(case["name"])
        case_seed = args.seed + i * 9973

        if args.skip_done and case_dump_done(args.dump_dir, name):
            print(f"[{i+1}/{len(selected)}] SKIP {name} (manifest exists)")
            results.append({"name": name, "status": "skip"})
            skip += 1
            continue

        reason = gpu_dump_skip_reason(case)
        if reason:
            print(f"[{i+1}/{len(selected)}] SKIP {name} ({reason})")
            results.append({"name": name, "status": "skip_gpu", "reason": reason})
            skip += 1
            continue

        print(f"[{i+1}/{len(selected)}] RUN  {name} ...", flush=True)
        try:
            rec = _run_one_case(
                case,
                dump_dir=args.dump_dir,
                device=device,
                seed=case_seed,
                dtype_save=args.dtype_save,
            )
            print(
                f"  OK {name} aqk={rec['aqk_shape']} akkd={rec['akkd_shape']} "
                f"{rec['elapsed_s']}s → {rec['dump_pt']}",
                flush=True,
            )
            results.append(rec)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {name}: {exc}", flush=True)
            traceback.print_exc()
            results.append(
                {
                    "name": name,
                    "status": "fail",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            fail += 1

    for item in gpu_skipped:
        results.append({"name": item["name"], "status": "skip_gpu", "reason": item["reason"]})

    report = {
        "op": OP_NAME,
        "dump_dir": str(args.dump_dir),
        "phase": args.phase,
        "seed": args.seed,
        "ok": ok,
        "skip": skip,
        "fail": fail,
        "gpu_skipped": gpu_skipped,
        "results": results,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"report → {report_path}  ok={ok} skip={skip} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
