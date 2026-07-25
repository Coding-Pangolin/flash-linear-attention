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
TEST_DIR = GPU_ROOT / "tests" / "ops"
for p in (str(GPU_ROOT), str(SCRIPT_DIR), str(TEST_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from intra_sub_chunk_case_utils import (  # noqa: E402
    BC,
    build_intra_sub_chunk_inputs,
    case_dump_done,
    case_seed,
    filter_cases,
    filter_gpu_dump_cases,
    gpu_dump_skip_reason,
    load_cases,
)
from chunk_kda_fwd_intra_sub_chunk_ref import (  # noqa: E402
    chunk_kda_fwd_intra_sub_chunk_ref,
    prepare_chunk_indices,
)

OP_NAME = "chunk_kda_fwd_intra_sub_chunk"
OP_NAME_CPU = "chunk_kda_fwd_intra_sub_chunk_cpu"

_CPU_DTYPE_MAP = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp64": torch.float64,
    "float64": torch.float64,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch dump GPU(+CPU) chunk_kda_fwd_intra_sub_chunk I/O from intra_sub_chunk_cases.json"
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
    p.add_argument(
        "--cpu-dtype",
        default="fp32",
        choices=sorted(_CPU_DTYPE_MAP),
        help="CPU golden compute dtype (default: fp32, aligns with NPU accum)",
    )
    p.add_argument(
        "--no-cpu",
        action="store_true",
        help="skip CPU golden dump (GPU I/O only)",
    )
    p.add_argument("--seed", type=int, default=0, help="base seed; case i uses seed+i*9973")
    p.add_argument(
        "--rng-on-cuda",
        action="store_true",
        help="sample tensors on CUDA (legacy; breaks seed parity with NPU). Default: CPU RNG",
    )
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


def _bsnd_to_bthd(x: torch.Tensor) -> torch.Tensor:
    """[B, H, T, ...] -> [B, T, H, ...]"""
    return x.transpose(1, 2).contiguous()


def _bthd_to_bnsd(x: torch.Tensor) -> torch.Tensor:
    """[B, T, H, ...] -> [B, H, T, ...]"""
    return x.transpose(1, 2).contiguous()


def run_cpu_intra_sub_chunk(
    q_bthd: torch.Tensor,
    k_bthd: torch.Tensor,
    g_bthd: torch.Tensor,
    beta_bthd: torch.Tensor,
    scale: float,
    chunk_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    *,
    cpu_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run CPU golden on the same tensors as GPU (GPU layout BTHD in, BTHD out)."""
    q = _bthd_to_bnsd(q_bthd.detach().cpu())
    k = _bthd_to_bnsd(k_bthd.detach().cpu())
    g = _bthd_to_bnsd(g_bthd.detach().cpu())
    beta = beta_bthd.detach().cpu().transpose(1, 2).contiguous()  # [B,T,HV] -> [B,HV,T]
    cu = None if cu_seqlens is None else cu_seqlens.detach().cpu().long()
    idx_flat = None
    if chunk_indices is not None:
        idx_flat = chunk_indices.detach().cpu().reshape(-1).long()
    elif cu is not None:
        flat = prepare_chunk_indices(cu, chunk_size)
        idx_flat = torch.tensor(flat, dtype=torch.long)

    aqk_bnsd, akkd_bnsd = chunk_kda_fwd_intra_sub_chunk_ref(
        q,
        k,
        g,
        beta,
        scale,
        chunk_size,
        cu,
        idx_flat,
        dtype=cpu_dtype,
    )
    return _bsnd_to_bthd(aqk_bnsd), _bsnd_to_bthd(akkd_bnsd)


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
    cpu_dtype: Optional[torch.dtype] = None,
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
                "q/k [B,T,H,K]; g/aqk/aqk_cpu [B,T,HV,BT]; akkd/akkd_cpu [B,T,HV,BC]; "
                "beta [B,T,HV]. GPU akkd is fp32; CPU outputs use cpu_dtype (default fp32). "
                "NPU: transpose(1,2) → BNSD; beta [B,T,HV]→[B,HV,T]."
            ),
        },
        "inputs": packed_in,
        "outputs": packed_out,
        "meta": {
            **case_meta,
            "fla_file": fla_file,
            "save_fp32": save_fp32,
            "BC": BC,
            "cpu_dtype": None
            if cpu_dtype is None
            else str(cpu_dtype).replace("torch.", ""),
            "has_cpu_golden": "aqk_cpu" in packed_out,
        },
    }
    pt_name = f"001_{OP_NAME}.pt"
    pt_path = out_dir / pt_name
    torch.save(payload, pt_path)

    manifest = [{"step": 1, "op": OP_NAME, "path": pt_name}]
    if "aqk_cpu" in packed_out:
        # Separate CPU-only payload for loaders that expect one backend per file.
        cpu_payload = {
            "op": OP_NAME_CPU,
            "step": 2,
            "layout": payload["layout"],
            "inputs": packed_in,
            "outputs": {
                "aqk": packed_out["aqk_cpu"],
                "akkd": packed_out["akkd_cpu"],
            },
            "meta": {
                **payload["meta"],
                "backend": "cpu_ref",
                "source_pt": pt_name,
            },
        }
        cpu_name = f"002_{OP_NAME_CPU}.pt"
        torch.save(cpu_payload, out_dir / cpu_name)
        manifest.append({"step": 2, "op": OP_NAME_CPU, "path": cpu_name})

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
    cpu_dtype: torch.dtype,
    dump_cpu: bool,
    rng_on_cpu: bool = True,
) -> dict[str, Any]:
    import fla

    t0 = time.time()
    name = str(case["name"])
    save_fp32 = dtype_save.lower() in ("fp32", "float32", "float")

    bundle = build_intra_sub_chunk_inputs(
        case, device=device, seed=seed, rng_on_cpu=rng_on_cpu
    )
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

    outputs: dict[str, Any] = {"aqk": aqk, "akkd": akkd}
    t_cpu = None
    if dump_cpu:
        t_c0 = time.time()
        aqk_cpu, akkd_cpu = run_cpu_intra_sub_chunk(
            q,
            k,
            g,
            beta,
            scale,
            BT,
            cu_seqlens=cu,
            chunk_indices=idx,
            cpu_dtype=cpu_dtype,
        )
        t_cpu = time.time() - t_c0
        outputs["aqk_cpu"] = aqk_cpu
        outputs["akkd_cpu"] = akkd_cpu
        meta = {
            **meta,
            "cpu_dtype": str(cpu_dtype).replace("torch.", ""),
            "cpu_backend": "chunk_kda_fwd_intra_sub_chunk_ref",
        }

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
        outputs=outputs,
        save_fp32=save_fp32,
        fla_file=str(fla.__file__),
        cpu_dtype=cpu_dtype if dump_cpu else None,
    )
    elapsed = time.time() - t0
    rec = {
        "name": name,
        "status": "ok",
        "elapsed_s": round(elapsed, 3),
        "aqk_shape": list(aqk.shape),
        "akkd_shape": list(akkd.shape),
        "dump_pt": str(pt_path),
        "dump_dir": str(out_dir),
        "has_cpu": dump_cpu,
    }
    if t_cpu is not None:
        rec["t_cpu_s"] = round(t_cpu, 3)
        rec["aqk_cpu_shape"] = list(outputs["aqk_cpu"].shape)
        rec["akkd_cpu_shape"] = list(outputs["akkd_cpu"].shape)
    return rec


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

    cpu_dtype = _CPU_DTYPE_MAP[args.cpu_dtype]
    dump_cpu = not args.no_cpu
    print(f"cases_file={args.cases_file}")
    print(
        f"phase={args.phase} selected={len(selected)} "
        f"gpu_skipped={len(gpu_skipped)} dump_dir={args.dump_dir} device={args.device} "
        f"dump_cpu={dump_cpu} cpu_dtype={args.cpu_dtype}"
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
        seed_i = case_seed(args.seed, i)

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

        print(f"[{i+1}/{len(selected)}] RUN  {name} seed={seed_i} ...", flush=True)
        try:
            rec = _run_one_case(
                case,
                dump_dir=args.dump_dir,
                device=device,
                seed=seed_i,
                dtype_save=args.dtype_save,
                cpu_dtype=cpu_dtype,
                dump_cpu=dump_cpu,
                rng_on_cpu=not args.rng_on_cuda,
            )
            extra = ""
            if rec.get("has_cpu"):
                extra = f" cpu={rec.get('t_cpu_s')}s"
            print(
                f"  OK {name} aqk={rec['aqk_shape']} akkd={rec['akkd_shape']} "
                f"{rec['elapsed_s']}s{extra} → {rec['dump_pt']}",
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
