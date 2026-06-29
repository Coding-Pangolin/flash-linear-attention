"""GPU dump [B,T,H,*] (BTH) -> NPU aclnn [B,H,T,*] (BHT) layout conversion."""
from __future__ import annotations

from typing import Any, Mapping

import torch

_BTH_TO_BHT_NAMES = frozenset({
    "q", "k", "v", "w", "u", "g", "beta", "do", "dv", "dv2", "du",
    "dq", "dk", "dw", "dg", "db", "dg2", "dk2", "v_new", "o",
    "A",
})
_BNTH_TO_BHNT_NAMES = frozenset({"h", "dh"})
_PASSTHROUGH = frozenset({"initial_state", "final_state", "h0", "dh0", "dht"})


def bth_to_bht(t: torch.Tensor) -> torch.Tensor:
    """[B, T, H, *] -> [B, H, T, *]：交换 T 与 H 维即可。"""
    if t.ndim < 3:
        return t
    return t.transpose(1, 2).contiguous()


def bnth_to_bhnt(t: torch.Tensor) -> torch.Tensor:
    """[B, NT, H, K, V] -> [B, H, NT, K, V]：chunk 维在 NT，不能只 transpose(1,2)。"""
    if t.ndim != 5:
        return t
    return t.permute(0, 2, 1, 3, 4).contiguous()


def to_npu_tensor(op: str, name: str, t: Any, *, beta_fp32: bool = True) -> Any:
    if not isinstance(t, torch.Tensor):
        return t
    if name in _PASSTHROUGH:
        return t.detach().cpu()
    if name in _BNTH_TO_BHNT_NAMES:
        out = bnth_to_bhnt(t)
    elif name in _BTH_TO_BHT_NAMES:
        out = bth_to_bht(t)
    else:
        out = t
    out = out.detach().cpu()
    if beta_fp32 and name == "beta" and out.is_floating_point():
        out = out.float()
    return out


def to_npu_mapping(
    op: str,
    mapping: Mapping[str, Any] | None,
    *,
    beta_fp32: bool = True,
) -> dict[str, Any]:
    if not mapping:
        return {}
    return {
        k: to_npu_tensor(op, k, v, beta_fp32=beta_fp32)
        for k, v in mapping.items()
        if v is not None
    }


def chunk_indices_npu_list(chunk_indices: Any) -> list[int] | None:
    if chunk_indices is None:
        return None
    if isinstance(chunk_indices, torch.Tensor):
        if chunk_indices.ndim == 2 and chunk_indices.shape[-1] == 2:
            return [int(x) for x in chunk_indices.detach().cpu().reshape(-1).tolist()]
        return [int(x) for x in chunk_indices.detach().cpu().flatten().tolist()]
    if isinstance(chunk_indices, (list, tuple)):
        return [int(x) for x in chunk_indices]
    return None


def load_dump_for_npu(
    path: str,
    *,
    device: str | torch.device | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load .pt and convert inputs/outputs to NPU BHT layout (transpose on read)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    op = str(d["op"])
    meta = dict(d.get("meta") or {})
    if "inputs_npu" in d:
        inputs = d["inputs_npu"]
        outputs = d.get("outputs_npu") or {}
    else:
        inputs = to_npu_mapping(op, d.get("inputs") or {})
        outputs = to_npu_mapping(op, d.get("outputs") or {})
    if device is not None:
        dev = torch.device(device)
        inputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        outputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
    return inputs, meta, outputs
