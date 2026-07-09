#!/usr/bin/env python3
"""Replay GPU KDA chunk_kda from saved dump ``001_chunk_kda_fwd.pt``.

Load inputs from a dump case directory (or a single .pt file), run ``chunk_kda``
on CUDA, print finite stats, and optionally compare with golden outputs in the dump.

Usage:
  python3 scripts/run_kda_replay_dump.py /data/kda_dump/model/model_fused_t131072
  python3 scripts/run_kda_replay_dump.py /data/kda_dump/all --phase smoke
  python3 scripts/run_kda_replay_dump.py /path/to/001_chunk_kda_fwd.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
GPU_ROOT = SCRIPT_DIR.parent
if str(GPU_ROOT) not in sys.path:
    sys.path.insert(0, str(GPU_ROOT))

OP_PT = "001_chunk_kda_fwd.pt"


def _load_dump(pt_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    inputs = {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for k, v in (data.get("inputs") or {}).items()
    }
    outputs = {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for k, v in (data.get("outputs") or {}).items()
    }
    meta = dict(data.get("meta") or {})
    return inputs, meta, outputs


def _merge_meta(meta: dict, case_meta: dict) -> dict:
    merged = dict(case_meta)
    merged.update(meta)
    return merged


def _int_list(value) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().tolist()]
    return [int(x) for x in value]


def _cu_list(meta: dict, case_meta: dict) -> list[int] | None:
    for src in (meta, case_meta):
        cu = src.get("cu_seqlens")
        if cu is None:
            continue
        if isinstance(cu, torch.Tensor):
            cu = [int(x) for x in cu.tolist()]
        else:
            cu = [int(x) for x in cu]
        if cu:
            return cu
    return None


def _finite_stats(t: torch.Tensor) -> str:
    x = t.detach().float().view(-1)
    total = x.numel()
    finite = int(torch.isfinite(x).sum())
    nan = int(torch.isnan(x).sum())
    inf = int(torch.isinf(x).sum())
    return f"finite {finite}/{total} (nan={nan}, inf={inf})"


def _resolve_pt(path: Path) -> Path:
    if path.is_file() and path.suffix == ".pt":
        return path
    pt = path / OP_PT
    if pt.is_file():
        return pt
    raise FileNotFoundError(f"no {OP_PT} under {path}")


def _list_case_dirs(root: Path, phase: str) -> list[Path]:
    if (root / OP_PT).is_file() or root.suffix == ".pt":
        return [root]
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / OP_PT).is_file())
    phase = phase.strip().lower()
    if phase in ("", "all"):
        return dirs
    if phase == "smoke":
        return [d for d in dirs if d.name.startswith("smoke_")]
    if phase.startswith("prefix:"):
        prefix = phase.split(":", 1)[1]
        return [d for d in dirs if d.name.startswith(prefix)]
    raise ValueError(f"unknown phase {phase!r}")


def run_replay(
    pt_path: Path,
    *,
    device: torch.device,
    compare: bool = True,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> dict[str, Any]:
    from fla.ops.kda import chunk_kda

    case_dir = pt_path.parent
    case_name = case_dir.name if pt_path.name == OP_PT else pt_path.stem
    inputs, meta, golden = _load_dump(pt_path)

    case_meta_path = case_dir / "case_meta.json"
    case_meta = {}
    if case_meta_path.is_file():
        case_meta = json.loads(case_meta_path.read_text(encoding="utf-8"))
    meta = _merge_meta(meta, case_meta)

    q = inputs["q"].contiguous().to(device)
    k = inputs["k"].contiguous().to(device)
    v = inputs["v"].contiguous().to(device)
    g = inputs["g"].contiguous().to(device)
    beta = inputs["beta"].contiguous().to(device)

    chunk_size = int(meta.get("chunk_size") or 64)
    scale_val = meta.get("scale", inputs.get("scale"))
    if isinstance(scale_val, torch.Tensor):
        scale_val = float(scale_val.item())
    scale = float(scale_val if scale_val is not None else (q.shape[-1] ** -0.5))

    cu_seqlens = _cu_list(meta, case_meta)
    chunk_indices = _int_list(meta.get("chunk_indices") or inputs.get("chunk_indices"))
    cu_tensor = None
    if cu_seqlens is not None:
        cu_tensor = torch.tensor(cu_seqlens, dtype=torch.int64, device=device)

    A_log = inputs.get("A_log")
    dt_bias = inputs.get("dt_bias")
    if A_log is not None:
        A_log = A_log.contiguous().to(device)
    if dt_bias is not None:
        dt_bias = dt_bias.contiguous().to(device)

    initial_state = inputs.get("initial_state")
    if initial_state is not None:
        initial_state = initial_state.contiguous().to(device).float()

    flags = {
        "use_qk_l2norm_in_kernel": bool(meta.get("use_qk_l2norm_in_kernel", True)),
        "use_gate_in_kernel": bool(meta.get("use_gate_in_kernel", True)),
        "use_beta_sigmoid_in_kernel": bool(meta.get("use_beta_sigmoid_in_kernel", True)),
        "allow_neg_eigval": bool(meta.get("allow_neg_eigval", False)),
        "safe_gate": bool(meta.get("safe_gate", True)),
        "lower_bound": float(meta.get("lower_bound", -5.0)),
        "state_v_first": bool(meta.get("state_v_first", False)),
    }

    print(
        f"\n=== {case_name} ===\n"
        f"  pt={pt_path}\n"
        f"  B={q.shape[0]} T={q.shape[1]} Hk={q.shape[2]} Hv={v.shape[2]} "
        f"K={q.shape[3]} V={v.shape[3]} cs={chunk_size} dtype={q.dtype}\n"
        f"  flags={flags} varlen={cu_seqlens is not None}",
        flush=True,
    )

    call_kw: dict[str, Any] = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "scale": scale,
        "chunk_size": chunk_size,
        "output_final_state": True,
        **flags,
    }
    if A_log is not None:
        call_kw["A_log"] = A_log
    if dt_bias is not None:
        call_kw["dt_bias"] = dt_bias
    if initial_state is not None:
        call_kw["initial_state"] = initial_state
    if chunk_indices is not None:
        call_kw["chunk_indices"] = torch.tensor(chunk_indices, dtype=torch.int64, device=device)
    elif cu_tensor is not None:
        call_kw["cu_seqlens"] = cu_tensor

    t0 = time.time()
    with torch.cuda.device(device), torch.inference_mode():
        o, final_state = chunk_kda(**call_kw)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    print(f"  [gpu] o {_finite_stats(o)} min={float(o.float().min()):.4g} max={float(o.float().max()):.4g}", flush=True)
    if final_state is not None and final_state.numel() > 0:
        print(
            f"  [gpu] final_state {_finite_stats(final_state)} "
            f"min={float(final_state.float().min()):.4g} max={float(final_state.float().max()):.4g}",
            flush=True,
        )

    result: dict[str, Any] = {
        "name": case_name,
        "elapsed_s": round(elapsed, 3),
        "o_finite": bool(torch.isfinite(o).all()),
    }

    if compare and golden.get("o") is not None:
        o_gold = golden["o"].float()
        o_run = o.detach().cpu().float()
        diff = (o_run - o_gold).abs()
        max_diff = float(diff.max())
        ok = torch.allclose(o_run, o_gold, rtol=rtol, atol=atol)
        tag = "PASS" if ok else "FAIL"
        print(
            f"  [compare/o] {tag} max_abs_diff={max_diff:.4g} rtol={rtol} atol={atol}",
            flush=True,
        )
        result["o_match_golden"] = ok
        result["o_max_abs_diff"] = max_diff

        fs_gold = golden.get("final_state")
        if fs_gold is not None and final_state is not None and final_state.numel() > 0:
            fs_run = final_state.detach().cpu().float()
            fs_diff = (fs_run - fs_gold.float()).abs()
            fs_ok = torch.allclose(fs_run, fs_gold.float(), rtol=rtol, atol=atol)
            tag = "PASS" if fs_ok else "FAIL"
            print(
                f"  [compare/final_state] {tag} max_abs_diff={float(fs_diff.max()):.4g}",
                flush=True,
            )
            result["final_state_match_golden"] = fs_ok

    if not result["o_finite"]:
        result["status"] = "nan"
    elif compare and golden.get("o") is not None and not result.get("o_match_golden", True):
        result["status"] = "mismatch"
    else:
        result["status"] = "ok"

    print(f"  elapsed={elapsed:.3f}s status={result['status']}", flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="Replay chunk_kda from GPU dump .pt on CUDA")
    p.add_argument("dump_path", type=Path, help="case dir, dump root, or .pt file")
    p.add_argument("--case", default="", help="single case name when dump_path is a root dir")
    p.add_argument("--cases", default="", help="comma-separated case names")
    p.add_argument("--phase", default="all", help="all | smoke | prefix:<name>")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-compare", action="store_true", help="skip compare with golden in .pt")
    p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--atol", type=float, default=1e-2)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 2

    dump_path = args.dump_path.resolve()
    if args.case:
        targets = [dump_path / args.case]
    elif args.cases.strip():
        targets = [dump_path / n.strip() for n in args.cases.split(",") if n.strip()]
    else:
        targets = _list_case_dirs(dump_path, args.phase)

    if not targets:
        print(f"ERROR: no dump cases under {dump_path}", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    stats = {"ok": 0, "nan": 0, "mismatch": 0, "error": 0}

    for target in targets:
        try:
            pt_path = _resolve_pt(target)
            rec = run_replay(
                pt_path,
                device=device,
                compare=not args.no_compare,
                rtol=args.rtol,
                atol=args.atol,
            )
            stats[rec["status"]] = stats.get(rec["status"], 0) + 1
        except Exception as exc:
            print(f"=== {target.name} ERROR ===\n{exc}", flush=True)
            stats["error"] += 1

    print(
        f"\nDone: ok={stats.get('ok', 0)} nan={stats.get('nan', 0)} "
        f"mismatch={stats.get('mismatch', 0)} error={stats.get('error', 0)} "
        f"/ {sum(stats.values())} total",
        flush=True,
    )
    return 1 if stats.get("nan") or stats.get("mismatch") or stats.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
