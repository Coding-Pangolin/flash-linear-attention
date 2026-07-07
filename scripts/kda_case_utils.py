"""Shared helpers for KDA GPU dump cases (kda_cases.json)."""
from __future__ import annotations

import json
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

_LOW_PRECISION_HALF_RANGE = 6.5e-3
KDA_DUMP_CHUNK_SIZES = frozenset({32, 64})


def parse_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"unsupported dtype {name!r}, expected one of {sorted(_DTYPE_MAP)}")
    return _DTYPE_MAP[key]


def generate_cu_seqlens(
    cu_seqlens_len: int,
    total_length: int,
    *,
    seg_min: int = 64,
    seg_max: int = 128,
) -> torch.LongTensor:
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
        raise ValueError(f"generate_cu_seqlens: sum={cu[-1]} != T={total_length}")
    return torch.tensor(cu, dtype=torch.long)


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
    if phase.startswith("prefix:"):
        prefix = phase.split(":", 1)[1]
        return [c for c in selected if str(c["name"]).startswith(prefix)]
    raise ValueError(f"unknown phase {phase!r}; use all|smoke or prefix:<name_prefix>")


def kda_dump_skip_reason(case: dict[str, Any]) -> str | None:
    chunk_size = int(case.get("chunk_size", 64))
    if chunk_size not in KDA_DUMP_CHUNK_SIZES:
        supported = ", ".join(str(x) for x in sorted(KDA_DUMP_CHUNK_SIZES))
        return f"chunk_size={chunk_size} not supported (only {supported})"
    K = int(case["Kdim"])
    if K > 256:
        return f"Kdim={K} > 256 (KDA limit)"
    return None


def filter_kda_dump_cases(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    runnable: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for case in cases:
        reason = kda_dump_skip_reason(case)
        if reason:
            skipped.append({"name": str(case["name"]), "reason": reason})
        else:
            runnable.append(case)
    return runnable, skipped


def _rand_uniform(shape: tuple[int, ...], dtype: torch.dtype, half_range: float, device: torch.device) -> torch.Tensor:
    x = torch.rand(shape, dtype=torch.float32, device=device)
    x = (x * 2.0 - 1.0) * float(half_range)
    return x.to(dtype=dtype)


def _default_kernel_flags(case: dict[str, Any]) -> dict[str, Any]:
    """Production-like defaults aligned with FlashKDA / benchmark registry."""
    flags = {
        "use_qk_l2norm_in_kernel": bool(case.get("use_qk_l2norm_in_kernel", True)),
        "use_gate_in_kernel": bool(case.get("use_gate_in_kernel", True)),
        "use_beta_sigmoid_in_kernel": bool(case.get("use_beta_sigmoid_in_kernel", True)),
        "allow_neg_eigval": bool(case.get("allow_neg_eigval", False)),
        "safe_gate": bool(case.get("safe_gate", True)),
        "lower_bound": float(case.get("lower_bound", -5.0)),
        "state_v_first": bool(case.get("state_v_first", False)),
        "output_final_state": bool(case.get("output_final_state", True)),
        "disable_recompute": bool(case.get("disable_recompute", False)),
    }
    if flags["safe_gate"] and flags["use_gate_in_kernel"]:
        lb = flags["lower_bound"]
        if not (-5 <= lb < 0):
            raise ValueError(f"lower_bound must be in [-5, 0), got {lb}")
    return flags


def build_kda_inputs(
    case: dict[str, Any],
    *,
    device: torch.device,
    seed: int = 0,
) -> dict[str, Any]:
    """Build chunk_kda inputs in GPU layout [B, T, H/HV, ...]."""
    B = int(case["B"])
    T = int(case["T"])
    Hk = int(case["query_head"])
    Hv = int(case["value_head"])
    K = int(case["Kdim"])
    V = int(case["Vdim"])
    chunk_size = int(case.get("chunk_size", 64))
    varlen = bool(case.get("varlen", False))
    ktype = parse_dtype(case["dtype"])

    if Hv % Hk != 0:
        raise ValueError(f"GVA requires Hv % Hk == 0, got Hk={Hk}, Hv={Hv}")
    if varlen and B != 1:
        raise ValueError(f"varlen case {case['name']} expects B=1, got B={B}")

    torch.manual_seed(seed)
    random.seed(seed)

    low = ktype in (torch.float16, torch.bfloat16)
    hr = _LOW_PRECISION_HALF_RANGE if low else 2e-2

    q = _rand_uniform((B, T, Hk, K), ktype, hr, device)
    k = _rand_uniform((B, T, Hk, K), ktype, hr, device)
    v = _rand_uniform((B, T, Hv, V), ktype, hr, device)

    flags = _default_kernel_flags(case)
    use_gate = flags["use_gate_in_kernel"]
    use_beta_sig = flags["use_beta_sigmoid_in_kernel"]

    if use_gate:
        g = torch.randn(B, T, Hv, K, dtype=ktype, device=device)
        A_log = torch.log(torch.empty(Hv, dtype=torch.float32, device=device).uniform_(1, 16))
        dt_bias = torch.randn(Hv * K, dtype=torch.float32, device=device)
    else:
        import torch.nn.functional as F
        g = F.logsigmoid(torch.randn(B, T, Hv, K, dtype=torch.float32, device=device)).to(ktype)
        A_log = None
        dt_bias = None

    if use_beta_sig:
        beta = torch.randn(B, T, Hv, dtype=ktype, device=device)
    else:
        beta = torch.sigmoid(_rand_uniform((B, T, Hv), ktype, 0.5, device))

    cu_seqlens: Optional[torch.LongTensor] = None
    if varlen:
        mean_len = int(case.get("mean_len", 4))
        cu_seqlens = generate_cu_seqlens(
            mean_len,
            T,
            seg_min=chunk_size,
            seg_max=min(128, chunk_size * 2),
        ).to(device)

    scale = float(case.get("scale", K ** -0.5))
    num_seqs = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    initial_state = torch.randn(num_seqs, Hv, K, V, dtype=torch.float32, device=device)

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": scale,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "chunk_size": chunk_size,
        "flags": flags,
        "meta": {
            "case_name": case["name"],
            "description": case.get("description", ""),
            "B": B,
            "T": T,
            "Hk": Hk,
            "Hv": Hv,
            "K": K,
            "V": V,
            "chunk_size": chunk_size,
            "varlen": varlen,
            "dtype": case["dtype"],
            "scale": scale,
            "seed": seed,
            "mean_len": case.get("mean_len"),
            "cu_seqlens": cu_seqlens.detach().cpu().tolist() if cu_seqlens is not None else None,
            **flags,
        },
    }


def case_dump_done(dump_root: Path, case_name: str) -> bool:
    manifest = dump_root / case_name / "manifest.json"
    return manifest.is_file() and manifest.stat().st_size > 2
