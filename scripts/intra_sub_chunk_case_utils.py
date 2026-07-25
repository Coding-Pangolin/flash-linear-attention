"""Helpers for chunk_kda_fwd_intra_sub_chunk GPU dump cases."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Optional

import torch

_DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

# Triton intra_sub_chunk in this repo: BT∈{32,64} (dual test / kernel).
GPU_CHUNK_SIZES = frozenset({32, 64})
BC = 16


def parse_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"unsupported dtype {name!r}")
    return _DTYPE_MAP[key]


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("cases", data)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list or object with 'cases'")
    return data


def filter_cases(
    cases: list[dict[str, Any]],
    *,
    phase: str = "all",
    names: Optional[list[str]] = None,
    include_disabled: bool = False,
) -> list[dict[str, Any]]:
    if names:
        by_name = {c["name"]: c for c in cases}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(f"unknown case(s): {', '.join(missing)}")
        selected = [by_name[n] for n in names]
        if not include_disabled:
            selected = [c for c in selected if c.get("enabled", True)]
        return selected

    selected = list(cases)
    if not include_disabled:
        selected = [c for c in selected if c.get("enabled", True)]

    phase = phase.strip().lower()
    if phase in ("", "all"):
        return selected
    if phase in ("smoke", "0"):
        return [c for c in selected if str(c["name"]).startswith("smoke_")]
    if phase in ("gdn", "table"):
        return [c for c in selected if str(c["name"]).startswith("BSND_")]
    if phase in ("varlen", "var"):
        return [c for c in selected if bool(c.get("varlen"))]
    if phase in ("gva",):
        return [
            c
            for c in selected
            if int(c["value_head"]) != int(c["query_head"]) or "GVA" in str(c["name"])
        ]
    if phase.startswith("prefix:"):
        prefix = phase.split(":", 1)[1]
        return [c for c in selected if str(c["name"]).startswith(prefix)]
    raise ValueError(f"unknown phase {phase!r}")


def gpu_dump_skip_reason(case: dict[str, Any]) -> Optional[str]:
    bt = int(case.get("chunk_size", 64))
    if bt not in GPU_CHUNK_SIZES:
        return f"chunk_size={bt} not supported on GPU (only {sorted(GPU_CHUNK_SIZES)})"
    k = int(case["Kdim"])
    if k != 128:
        return f"Kdim={k}≠128 (NPU tiling lock)"
    h = int(case["query_head"])
    hv = int(case["value_head"])
    if hv < h or hv % h != 0:
        return f"HV={hv} H={h} not GVA-legal"
    if bool(case.get("varlen")) and int(case["B"]) != 1:
        return "varlen requires B=1"
    return None


def filter_gpu_dump_cases(
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    runnable: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for case in cases:
        reason = gpu_dump_skip_reason(case)
        if reason:
            skipped.append({"name": str(case["name"]), "reason": reason})
        else:
            runnable.append(case)
    return runnable, skipped


def case_dump_done(dump_dir: Path, name: str) -> bool:
    return (dump_dir / name / "manifest.json").is_file()


def make_random_cu_seqlens(
    T: int,
    n_seq: int,
    *,
    seed: int,
    bt: int,
    unaligned: bool = True,
) -> list[int]:
    """Match NPU prec_gdn_isub random unaligned cu construction."""
    assert T >= 1 and n_seq >= 1
    if n_seq == 1:
        return [0, T]
    if n_seq > T:
        n_seq = T
    rng = random.Random(seed)
    weights = [rng.random() + 0.05 for _ in range(n_seq)]
    s = sum(weights)
    lengths = [max(1, int(T * w / s)) for w in weights]
    lengths[0] += T - sum(lengths)
    if lengths[0] < 1:
        need = 1 - lengths[0]
        lengths[0] = 1
        lengths[-1] -= need
    if unaligned:
        for i, L in enumerate(lengths):
            if L % bt == 0 and L > 1:
                lengths[i] = L - 1
                lengths[(i + 1) % n_seq] += 1
                break
        for i, L in enumerate(lengths):
            if L % BC == 0 and L > 1:
                lengths[i] = L - 1
                lengths[(i + 1) % n_seq] += 1
                break
    lengths = [max(1, int(x)) for x in lengths]
    lengths[-1] += T - sum(lengths)
    cu = [0]
    for L in lengths:
        cu.append(cu[-1] + L)
    assert cu[-1] == T
    return cu


def _make_gate_bsnd(
    B: int, HV: int, T: int, K: int, dtype: torch.dtype, mode: str, device: torch.device
) -> torch.Tensor:
    if mode == "lin_strong":
        g = -torch.linspace(0, 30, T, device=device).view(1, T, 1, 1).expand(B, T, HV, K)
    elif mode == "lin_mild":
        g = -torch.linspace(0, 8, T, device=device).view(1, T, 1, 1).expand(B, T, HV, K)
    elif mode == "rand":
        g = torch.randn(B, T, HV, K, device=device)
    else:
        raise ValueError(mode)
    return g.to(dtype).contiguous()


def _l2norm_last(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def prepare_chunk_indices_flat(cu_seqlens: torch.Tensor, chunk_size: int) -> list[int]:
    indices: list[int] = []
    for seq in range(int(cu_seqlens.numel()) - 1):
        length = int(cu_seqlens[seq + 1] - cu_seqlens[seq])
        n_chunks = (length + chunk_size - 1) // chunk_size
        for local in range(n_chunks):
            indices.extend([seq, local])
    return indices


def case_seed(base_seed: int, case_index: int) -> int:
    """Stable per-case seed shared by GPU dump / NPU seed-dual (must stay in sync)."""
    return int(base_seed) + int(case_index) * 9973


def build_intra_sub_chunk_inputs(
    case: dict[str, Any],
    *,
    device: torch.device,
    seed: int,
    rng_on_cpu: bool = True,
) -> dict[str, Any]:
    """Build GPU-layout (BTHD) tensors + meta for one case.

    Default ``rng_on_cpu=True``: sample with CPU RNG then ``.to(device)`` so the
    same ``seed`` reproduces on NPU hosts (no dump transfer of inputs).
    """
    torch.manual_seed(seed)
    gen_dev = torch.device("cpu") if rng_on_cpu else device
    if (not rng_on_cpu) and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    B = int(case["B"])
    T = int(case["T"])
    H = int(case["query_head"])
    HV = int(case["value_head"])
    K = int(case["Kdim"])
    BT = int(case["chunk_size"])
    dtype = parse_dtype(case.get("dtype", "bf16"))
    gate = str(case.get("gate", "lin_mild"))
    l2norm = bool(case.get("l2norm", True))
    varlen = bool(case.get("varlen", False))
    scale = 1.0 / math.sqrt(K)

    q = torch.randn(B, T, H, K, dtype=dtype, device=gen_dev)
    k = torch.randn(B, T, H, K, dtype=dtype, device=gen_dev)
    if l2norm:
        q = _l2norm_last(q.float()).to(dtype)
        k = _l2norm_last(k.float()).to(dtype)
    g = _make_gate_bsnd(B, HV, T, K, dtype, gate, gen_dev)
    beta = torch.rand(B, T, HV, dtype=dtype, device=gen_dev)

    cu_list: Optional[list[int]] = None
    cu_t: Optional[torch.Tensor] = None
    idx_t: Optional[torch.Tensor] = None
    if varlen:
        assert B == 1
        n_seq = int(case.get("n_seq", 2))
        if "cu_seqlens" in case and case["cu_seqlens"] is not None:
            cu_list = [int(x) for x in case["cu_seqlens"]]
        else:
            cu_list = make_random_cu_seqlens(T, n_seq, seed=seed, bt=BT, unaligned=True)
        assert cu_list[0] == 0 and cu_list[-1] == T
        cu_t = torch.tensor(cu_list, dtype=torch.long, device=gen_dev)
        flat = prepare_chunk_indices_flat(cu_t.cpu(), BT)
        idx_t = torch.tensor(flat, dtype=torch.long, device=gen_dev).view(-1, 2).contiguous()

    # Move to target device after RNG (identity if already there).
    if device.type != gen_dev.type or device.index != gen_dev.index:
        q = q.to(device)
        k = k.to(device)
        g = g.to(device)
        beta = beta.to(device)
        if cu_t is not None:
            cu_t = cu_t.to(device)
        if idx_t is not None:
            idx_t = idx_t.to(device)

    meta = {
        "name": str(case["name"]),
        "description": case.get("description", ""),
        "B": B,
        "T": T,
        "H": H,
        "HV": HV,
        "K": K,
        "chunk_size": BT,
        "dtype": str(dtype).replace("torch.", ""),
        "gate": gate,
        "l2norm": l2norm,
        "varlen": varlen,
        "n_seq": case.get("n_seq"),
        "cu_seqlens": cu_list,
        "scale": scale,
        "seed": seed,
        "rng_on_cpu": bool(rng_on_cpu),
        "gpu_layout": "BTHD",
        "npu_layout_hint": "transpose(1,2) → BNSD for q/k/g/aqk/akkd; beta [B,T,HV]→[B,HV,T]",
        "op": "chunk_kda_fwd_intra_sub_chunk",
    }
    return {
        "q": q,
        "k": k,
        "g": g,
        "beta": beta,
        "scale": scale,
        "chunk_size": BT,
        "cu_seqlens": cu_t,
        "chunk_indices": idx_t,
        "meta": meta,
    }
