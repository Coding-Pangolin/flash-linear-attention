#!/usr/bin/env bash
# Batch GPU KDA chunk_kda dump aligned with gpu/kda_cases.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DUMP_DIR="${KDA_DUMP_DIR:-/tmp/kda_gpu_dump}"
PHASE="${KDA_DUMP_PHASE:-all}"
OPS="${KDA_DUMP_OPS:-fwd}"
DEVICE="${KDA_DUMP_DEVICE:-cuda:0}"
EXTRA_ARGS=()
NAMES_SET=false

usage() {
  cat <<'EOF'
Usage: run_kda_dump_cases.sh [options]

Environment:
  KDA_DUMP_DIR      output root (default: /tmp/kda_gpu_dump)
  KDA_DUMP_PHASE    case filter: all|smoke (default: all)
  KDA_DUMP_OPS      dump ops: fwd (default) | all
  KDA_DUMP_DEVICE   CUDA device (default: cuda:0)

Options:
  --phase PHASE     all | smoke | prefix:<name>
  --dump-dir DIR
  --names a,b,c     run specific cases
  --skip-done       skip cases with existing manifest.json
  --dry-run         list selected cases only
  --dtype-save fp32 save all tensors as fp32
  -h, --help

Examples:
  ./run_kda_dump_cases.sh --dry-run
  ./run_kda_dump_cases.sh --phase smoke --dump-dir /data/kda_dump/smoke --skip-done
  ./run_kda_dump_cases.sh --names smoke_mha_fix,gva_t4096_v256
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --dump-dir) DUMP_DIR="$2"; shift 2 ;;
    --names) EXTRA_ARGS+=(--names "$2"); NAMES_SET=true; shift 2 ;;
    --skip-done) EXTRA_ARGS+=(--skip-done); shift ;;
    --dry-run) EXTRA_ARGS+=(--dry-run); shift ;;
    --include-disabled) EXTRA_ARGS+=(--include-disabled); shift ;;
    --dtype-save) EXTRA_ARGS+=(--dtype-save "$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if $NAMES_SET; then
  PHASE="all"
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: torch with CUDA required" >&2
  exit 2
fi

python3 "$ROOT/scripts/run_kda_dump_cases.py" \
  --cases-file "$ROOT/kda_cases.json" \
  --dump-dir "$DUMP_DIR" \
  --phase "$PHASE" \
  --ops "$OPS" \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}"
