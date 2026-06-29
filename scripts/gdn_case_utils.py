"""Shared helpers for GPU GDN dump cases (aligned with NPU cases.json)."""
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

_LOW_PRECISION_INPUT_HALF_RANGE_QK = 6.5e-3
_LOW_PRECISION_INPUT_HALF_RANGE_V = 6.5e-3

# GPU 竞品 chunk_gated_delta_rule 当前仅 chunk_size=64 路径可靠（cumsum 等硬编码 64）
GPU_DUMP_CHUNK_SIZES = frozenset({64})


def parse_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"unsupported dtype {name!r}, expected one of {sorted(_DTYPE_MAP)}")
    return _DTYPE_MAP[key]


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("cases", data)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list or object with 'cases' list")
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
    else:
        selected = list(cases)

    if not include_disabled:
        selected = [c for c in selected if c.get("enabled", True)]

    phase = phase.strip().lower()
    if phase in ("", "all"):
        return selected
    if phase in ("1", "phase1", "phase_1"):
        return [c for c in selected if str(c["name"]).startswith("phase_1_")]
    if phase in ("2", "phase2", "phase_2", "gva"):
        return [c for c in selected if str(c["name"]).startswith("gva_")]
    if phase in ("0", "legacy", "smoke"):
        prefixes = ("fix_hk_eq_hv_", "var_hk_eq_hv_")
        return [c for c in selected if str(c["name"]).startswith(prefixes)]
    if phase.startswith("prefix:"):
        prefix = phase.split(":", 1)[1]
        return [c for c in selected if str(c["name"]).startswith(prefix)]
    raise ValueError(
        f"unknown phase {phase!r}; use all|1|2|legacy or prefix:<name_prefix>"
    )


def gpu_dump_skip_reason(case: dict[str, Any]) -> str | None:
    """Return skip reason when case cannot run on GPU dump path; None if supported."""
    chunk_size = int(case.get("chunk_size", 64))
    if chunk_size not in GPU_DUMP_CHUNK_SIZES:
        supported = ", ".join(str(x) for x in sorted(GPU_DUMP_CHUNK_SIZES))
        return f"chunk_size={chunk_size} not supported on GPU (only {supported})"
    return None


def filter_gpu_dump_cases(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Keep GPU-runnable cases; return (runnable, skipped[{name, reason}])."""
    runnable: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for case in cases:
        reason = gpu_dump_skip_reason(case)
        if reason:
            skipped.append({"name": str(case["name"]), "reason": reason})
        else:
            runnable.append(case)
    return runnable, skipped


def generate_cu_seqlens(
    cu_seqlens_len: int,
    total_length: int,
    *,
    seg_min: int = 64,
    seg_max: int = 128,
) -> torch.LongTensor:
    """Match NPU bwd_dhu test: fair split + clamp, sum exactly total_length."""
    batchsize = cu_seqlens_len - 1
    if batchsize <= 0:
        return torch.tensor([0, total_length], dtype=torch.long)

    B, T = batchsize, total_length
    lengths = [(T * (i + 1)) // B - (T * i) // B for i in range(B)]
    for i in range(B):
        lengths[i] = max(seg_min, min(seg_max, lengths[i]))

    diff = T - sum(lengths)
    while diff > 0:
        cand = [i for i in range(B) if lengths[i] < seg_max]
        if not cand:
            break
        i = min(cand, key=lambda j: lengths[j])
        lengths[i] += 1
        diff -= 1
    while diff < 0:
        cand = [i for i in range(B) if lengths[i] > seg_min]
        if not cand:
            break
        i = max(cand, key=lambda j: lengths[j])
        lengths[i] -= 1
        diff += 1

    guard = 0
    while diff != 0 and guard < B * (seg_max - seg_min + 64):
        guard += 1
        if diff > 0:
            i = min(range(B), key=lambda j: lengths[j])
            lengths[i] += 1
            diff -= 1
        else:
            i = max(range(B), key=lambda j: lengths[j])
            lengths[i] -= 1
            diff += 1

    sorted_l = sorted(lengths)
    seq_lengths: list[int] = []
    i, j = 0, len(sorted_l) - 1
    while i <= j:
        if i == j:
            seq_lengths.append(sorted_l[i])
        else:
            seq_lengths.append(sorted_l[i])
            seq_lengths.append(sorted_l[j])
        i += 1
        j -= 1

    cu = [0]
    for seg in seq_lengths:
        cu.append(cu[-1] + seg)
    if cu[-1] != total_length:
        raise ValueError(
            f"generate_cu_seqlens: sum={cu[-1]} != T={total_length}, "
            f"cu_seqlens_len={cu_seqlens_len}, seg=[{seg_min},{seg_max}]"
        )
    return torch.tensor(cu, dtype=torch.long)


def _rand_uniform(shape: tuple[int, ...], dtype: torch.dtype, half_range: float, device: torch.device) -> torch.Tensor:
    x = torch.rand(shape, dtype=torch.float32, device=device)
    x = (x * 2.0 - 1.0) * float(half_range)
    return x.to(dtype=dtype)


def _create_gate_g(
    B: int,
    Hv: int,
    T: int,
    gtype: torch.dtype,
    device: torch.device,
    *,
    narrow: bool = True,
    gate_function: str = "negative_linear",
) -> torch.Tensor:
    import torch.nn.functional as F

    if gate_function == "logsigmoid":
        g = F.logsigmoid(torch.randn(B, Hv, T, dtype=gtype, device=device))
        return g
    if gate_function == "zeros":
        return torch.zeros(B, Hv, T, dtype=gtype, device=device)
    # negative_linear (bwd_dhu GVA test / cases default)
    if narrow:
        lo, hi = -1e-2, -1e-6
    else:
        lo, hi = -5e-2, -5e-5
    span = hi - lo
    margin = max(span * 1e-7, 1e-12)
    g_t = torch.linspace(float(hi) - margin, float(lo) + margin, T, dtype=torch.float64)
    return g_t.view(1, 1, T).expand(B, Hv, T).contiguous().to(device=device, dtype=gtype)


def build_gdn_inputs(
    case: dict[str, Any],
    *,
    device: torch.device,
    seed: int = 0,
) -> dict[str, Any]:
    """Build chunk_gated_delta_rule inputs in GPU layout [B, T, H, ...]."""
    B = int(case["B"])
    T = int(case["T"])
    Hk = int(case["query_head"])
    Hv = int(case["value_head"])
    K = int(case["Kdim"])
    V = int(case["Vdim"])
    chunk_size = int(case.get("chunk_size", 64))
    varlen = bool(case.get("varlen", False))
    ktype = parse_dtype(case["dtype"])
    gtype = parse_dtype(case["gtype"])

    if Hv % Hk != 0:
        raise ValueError(f"GVA requires Hv % Hk == 0, got Hk={Hk}, Hv={Hv}")
    if varlen and B != 1:
        raise ValueError(f"varlen case {case['name']} expects B=1, got B={B}")

    torch.manual_seed(seed)
    random.seed(seed)

    low = ktype in (torch.float16, torch.bfloat16)
    hr = _LOW_PRECISION_INPUT_HALF_RANGE_QK if low else 2e-2
    hr_v = _LOW_PRECISION_INPUT_HALF_RANGE_V if low else 2e-2
    gate_function = str(case.get("gate_function", "negative_linear")).strip().lower()

    q = _rand_uniform((B, T, Hk, K), ktype, hr, device)
    k = _rand_uniform((B, T, Hk, K), ktype, hr, device)
    v = _rand_uniform((B, T, Hv, V), ktype, hr_v, device)
    beta = torch.sigmoid(_rand_uniform((B, T, Hv), ktype, 0.5, device))
    g_bh_t = _create_gate_g(B, Hv, T, gtype, device, narrow=low, gate_function=gate_function)
    g = g_bh_t.transpose(1, 2).contiguous()

    cu_seqlens: Optional[torch.LongTensor] = None
    if varlen:
        mean_len = int(case["mean_len"])
        cu_seqlens = generate_cu_seqlens(
            mean_len,
            T,
            seg_min=chunk_size,
            seg_max=min(128, chunk_size * 2),
        ).to(device)

    scale = float(K ** -0.5)

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "scale": scale,
        "cu_seqlens": cu_seqlens,
        "meta": {
            "case_name": case["name"],
            "B": B,
            "T": T,
            "Hk": Hk,
            "Hv": Hv,
            "K": K,
            "V": V,
            "chunk_size": chunk_size,
            "varlen": varlen,
            "dtype": case["dtype"],
            "gtype": case["gtype"],
            "scale": scale,
            "seed": seed,
            "gate_function": gate_function,
            "mean_len": case.get("mean_len"),
            "cu_seqlens": cu_seqlens.detach().cpu().tolist() if cu_seqlens is not None else None,
        },
    }


def case_dump_done(dump_root: Path, case_name: str) -> bool:
    manifest = dump_root / case_name / "manifest.json"
    return manifest.is_file() and manifest.stat().st_size > 2
