"""KDA chunk_kda I/O dump for GPU golden collection (NPU dual-benchmark).

Environment variables:
  KDA_DUMP_DIR       dump root (required to enable)
  KDA_DUMP_CASE      subdirectory name, default ``default``
  KDA_DUMP_OPS       comma-separated op names; default ``chunk_kda_fwd``
  KDA_DUMP_EXIT      ``1`` to sys.exit(0) after first matched op
  KDA_DUMP_DTYPE     set ``fp32`` to unify saved floating tensors
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Mapping, Optional

import torch

DEFAULT_DUMP_OPS = ("chunk_kda_fwd",)
DEFAULT_DUMP_OPS_STR = ",".join(DEFAULT_DUMP_OPS)

_ENABLED: Optional[bool] = None
_DIR: str = ""
_CASE: str = "default"
_OPS_FILTER: Optional[set[str]] = None
_SAVE_FP32: bool = False
_STEP: int = 0
_MANIFEST: list[dict[str, Any]] = []


def kda_dump_reset() -> None:
    """Reset module state between batch cases in one process."""
    global _ENABLED, _DIR, _CASE, _OPS_FILTER, _SAVE_FP32, _STEP, _MANIFEST
    _ENABLED = None
    _DIR = ""
    _CASE = "default"
    _OPS_FILTER = None
    _SAVE_FP32 = False
    _STEP = 0
    _MANIFEST = []


def _init() -> bool:
    global _ENABLED, _DIR, _CASE, _OPS_FILTER, _SAVE_FP32
    if _ENABLED is not None:
        return _ENABLED
    _DIR = os.environ.get("KDA_DUMP_DIR", "").strip()
    _CASE = os.environ.get("KDA_DUMP_CASE", "default").strip() or "default"
    raw_ops = os.environ.get("KDA_DUMP_OPS", DEFAULT_DUMP_OPS_STR).strip()
    if raw_ops.lower() in ("all", "*"):
        _OPS_FILTER = None
    elif raw_ops.lower() in ("", "default", "fwd"):
        _OPS_FILTER = set(DEFAULT_DUMP_OPS)
    else:
        _OPS_FILTER = {x.strip() for x in raw_ops.split(",") if x.strip()}
    _SAVE_FP32 = os.environ.get("KDA_DUMP_DTYPE", "").lower() in ("fp32", "float32", "float")
    _ENABLED = bool(_DIR)
    return _ENABLED


def _should_dump(op: str) -> bool:
    if not _init():
        return False
    if _OPS_FILTER is None:
        return True
    return op in _OPS_FILTER


def _to_cpu_tensor(x: Any) -> Any:
    if not isinstance(x, torch.Tensor):
        return x
    t = x.detach()
    if _SAVE_FP32 and t.is_floating_point():
        t = t.float()
    return t.cpu()


def _pack_mapping(m: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not m:
        return {}
    out: dict[str, Any] = {}
    for k, v in m.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            out[k] = _to_cpu_tensor(v)
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float)):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _pack_meta(meta: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not meta:
        return {}
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            out[k] = [int(x) for x in v.detach().cpu().flatten().tolist()]
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = v
    out.setdefault("gpu_layout", "BTHD")
    return out


def kda_dump_op(
    op: str,
    *,
    inputs: Optional[Mapping[str, Any]] = None,
    outputs: Optional[Mapping[str, Any]] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> None:
    if not _should_dump(op):
        return

    global _STEP
    _STEP += 1
    out_dir = os.path.join(_DIR, _CASE)
    os.makedirs(out_dir, exist_ok=True)

    payload: dict[str, Any] = {
        "op": op,
        "step": _STEP,
        "layout": {"storage": "BTHD", "note": "q/k [B,T,H,K]; v/g/o [B,T,HV,V/K]; beta [B,T,HV]"},
        "inputs": _pack_mapping(inputs),
        "outputs": _pack_mapping(outputs),
        "meta": _pack_meta(meta),
    }
    path = os.path.join(out_dir, f"{_STEP:03d}_{op}.pt")
    torch.save(payload, path)
    _MANIFEST.append({"step": _STEP, "op": op, "path": os.path.basename(path)})
    print(f"[KDA_DUMP] saved {path}", flush=True)

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(_MANIFEST, f, indent=2)

    if os.environ.get("KDA_DUMP_EXIT", "0") == "1":
        print(f"[KDA_DUMP] KDA_DUMP_EXIT=1 after op={op}", flush=True)
        sys.exit(0)


def kda_dump_enabled() -> bool:
    return _init()
