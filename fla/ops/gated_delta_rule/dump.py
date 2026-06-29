"""GDN 算子 I/O dump：供 GPU 竞品双标杆采集输入/输出。

环境变量：
  GDN_DUMP_DIR      dump 根目录（必填才启用）
  GDN_DUMP_CASE     子目录名，默认 default
  GDN_DUMP_OPS      逗号分隔算子名；默认 NPU 仓已有算子（见 NPU_DUMP_OPS）
  GDN_DUMP_EXIT     1 时在命中 GDN_DUMP_OPS 后 sys.exit(0)
  GDN_DUMP_DTYPE    保存 dtype，默认与 tensor 一致；设 fp32 可统一转 float
  GDN_DUMP_NPU_LAYOUT  1 时额外写入 inputs_npu/outputs_npu（默认 0，使用时再转置以省空间）
"""
from __future__ import annotations

# 与 NPU 仓单算子一一对应，按 GDN 调用链顺序
NPU_DUMP_OPS = (
    "recompute_wu",
    "fwd_h",
    "fwd_o",
    "bwd_dv_local",
    "bwd_dhu",
    "bwd_dqkwg",
    "prepare_wy_repr_bwd",
)
NPU_DUMP_OPS_DEFAULT = ",".join(NPU_DUMP_OPS)

import json
import os
import sys
from typing import Any, Mapping, Optional

import torch

from fla.ops.gated_delta_rule.npu_layout import chunk_indices_npu_list, to_npu_mapping

_ENABLED: Optional[bool] = None
_DIR: str = ""
_CASE: str = "default"
_OPS_FILTER: Optional[set[str]] = None
_SAVE_FP32: bool = False
_STEP: int = 0
_MANIFEST: list[dict[str, Any]] = []


def _init() -> bool:
    global _ENABLED, _DIR, _CASE, _OPS_FILTER, _SAVE_FP32
    if _ENABLED is not None:
        return _ENABLED
    _DIR = os.environ.get("GDN_DUMP_DIR", "").strip()
    _CASE = os.environ.get("GDN_DUMP_CASE", "default").strip() or "default"
    raw_ops = os.environ.get("GDN_DUMP_OPS", NPU_DUMP_OPS_DEFAULT).strip()
    if raw_ops.lower() in ("all", "*"):
        _OPS_FILTER = None
    elif raw_ops.lower() in ("", "npu", "default"):
        _OPS_FILTER = set(NPU_DUMP_OPS)
    else:
        _OPS_FILTER = {x.strip() for x in raw_ops.split(",") if x.strip()}
    _SAVE_FP32 = os.environ.get("GDN_DUMP_DTYPE", "").lower() in ("fp32", "float32", "float")
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
            if k == "chunk_indices":
                out[k] = chunk_indices_npu_list(v)
            else:
                out[k] = [int(x) for x in v.detach().cpu().flatten().tolist()]
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = v
    out.setdefault("gpu_layout", "BTHD")
    out.setdefault("npu_layout", "BHTD")
    if "chunk_indices" in out and "chunk_indices_npu" not in out:
        out["chunk_indices_npu"] = out["chunk_indices"]
    return out


def gdn_dump_op(
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

    inputs_gpu = _pack_mapping(inputs)
    outputs_gpu = _pack_mapping(outputs)
    meta_packed = _pack_meta(meta)

    payload: dict[str, Any] = {
        "op": op,
        "step": _STEP,
        "layout": {"storage": "BTHD", "npu": "BHTD transpose on load"},
        "inputs": inputs_gpu,
        "outputs": outputs_gpu,
        "meta": meta_packed,
    }
    if os.environ.get("GDN_DUMP_NPU_LAYOUT", "0") == "1":
        payload["inputs_npu"] = to_npu_mapping(op, inputs_gpu)
        payload["outputs_npu"] = to_npu_mapping(op, outputs_gpu)
    path = os.path.join(out_dir, f"{_STEP:03d}_{op}.pt")
    torch.save(payload, path)
    _MANIFEST.append({"step": _STEP, "op": op, "path": os.path.basename(path)})
    print(f"[GDN_DUMP] saved {path}", flush=True)

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(_MANIFEST, f, indent=2)

    if os.environ.get("GDN_DUMP_EXIT", "0") == "1":
        print(f"[GDN_DUMP] GDN_DUMP_EXIT=1 after op={op}", flush=True)
        sys.exit(0)


def gdn_dump_enabled() -> bool:
    return _init()
