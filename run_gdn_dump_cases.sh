#!/usr/bin/env bash
# Batch GPU GDN dump aligned with gpu/cases.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DUMP_DIR="${GDN_DUMP_DIR:-/tmp/gdn_gpu_dump}"
PHASE="${GDN_DUMP_PHASE:-2}"
OPS="${GDN_DUMP_OPS:-npu}"
DEVICE="${GDN_DUMP_DEVICE:-cuda:0}"
ONLY_ENABLED="${GDN_DUMP_ONLY_ENABLED:-0}"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: run_gdn_dump_cases.sh [options]

Environment:
  GDN_DUMP_DIR      output root (default: /tmp/gdn_gpu_dump)
  GDN_DUMP_PHASE    case filter: all|1|2|legacy (default: 2 = gva_*)
  GDN_DUMP_OPS      dump ops: npu (default) | all | comma-separated list
  GDN_DUMP_DEVICE   CUDA device (default: cuda:0)
  GDN_DUMP_ONLY_ENABLED  set to 1 to respect cases.json enabled=false

Options:
  --phase PHASE     same as GDN_DUMP_PHASE
  --dump-dir DIR
  --names a,b,c     run specific cases
  --skip-done       skip cases with existing manifest.json
  --dry-run         list selected cases only
  --no-bwd          forward only
  --dtype-save fp32 save all tensors as fp32
  -h, --help

Examples:
  # Phase-2 GVA matrix (gva_* cases)
  ./run_gdn_dump_cases.sh --phase 2 --skip-done

  # Phase-1 large-shape matrix
  ./run_gdn_dump_cases.sh --phase 1 --dump-dir /data/gdn_dump/phase1

  # Legacy smoke (fix_hk_eq_hv_*, var_hk_eq_hv_*)
  ./run_gdn_dump_cases.sh --phase legacy

  # Single case
  ./run_gdn_dump_cases.sh --names gva_fix_1 --include-disabled
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --dump-dir) DUMP_DIR="$2"; shift 2 ;;
    --names) EXTRA_ARGS+=(--names "$2"); shift 2 ;;
    --skip-done) EXTRA_ARGS+=(--skip-done); shift ;;
    --dry-run) EXTRA_ARGS+=(--dry-run); shift ;;
    --no-bwd) EXTRA_ARGS+=(--no-bwd); shift ;;
    --include-disabled) EXTRA_ARGS+=(--include-disabled); shift ;;
    --dtype-save) EXTRA_ARGS+=(--dtype-save "$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# phase_1_* / gva_* are mostly enabled=false in cases.json; include them by default.
if [[ "$ONLY_ENABLED" != "1" ]]; then
  EXTRA_ARGS+=(--include-disabled)
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: torch with CUDA required" >&2
  exit 2
fi

python3 "$ROOT/scripts/run_gdn_dump_cases.py" \
  --cases-file "$ROOT/cases.json" \
  --dump-dir "$DUMP_DIR" \
  --phase "$PHASE" \
  --ops "$OPS" \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}"
