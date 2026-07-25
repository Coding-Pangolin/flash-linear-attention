#!/usr/bin/env bash
# Batch GPU dump for chunk_kda_fwd_intra_sub_chunk (GDN isub dual-benchmark)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DUMP_DIR="${ISUB_DUMP_DIR:-/tmp/intra_sub_chunk_gpu_dump}"
PHASE="${ISUB_DUMP_PHASE:-all}"
DEVICE="${ISUB_DUMP_DEVICE:-cuda:0}"
EXTRA_ARGS=()
NAMES_SET=false

usage() {
  cat <<'EOF'
Usage: run_intra_sub_chunk_dump_cases.sh [options]

Environment:
  ISUB_DUMP_DIR      output root (default: /tmp/intra_sub_chunk_gpu_dump)
  ISUB_DUMP_PHASE    case filter: all|smoke|gdn|varlen|gva (default: all)
  ISUB_DUMP_DEVICE   CUDA device (default: cuda:0)

Options:
  --phase PHASE     all | smoke | gdn | varlen | gva | prefix:<name>
  --dump-dir DIR
  --names a,b,c     run specific cases
  --skip-done       skip cases with existing manifest.json
  --dry-run         list selected cases only
  --include-disabled
  --dtype-save fp32 save all tensors as fp32
  --cpu-dtype D     CPU golden dtype: fp32 (default) | fp64
  --no-cpu          dump GPU I/O only (skip CPU golden)
  -h, --help

Examples:
  ./run_intra_sub_chunk_dump_cases.sh --dry-run
  ./run_intra_sub_chunk_dump_cases.sh --phase smoke --dump-dir /data/isub_dump/smoke --skip-done
  ./run_intra_sub_chunk_dump_cases.sh --names BSND_noGVA_V128_14,BSND_GVA_V256_28
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
    --cpu-dtype) EXTRA_ARGS+=(--cpu-dtype "$2"); shift 2 ;;
    --no-cpu) EXTRA_ARGS+=(--no-cpu); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if $NAMES_SET; then
  PHASE="all"
fi

DRY=false
for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
  [[ "$a" == "--dry-run" ]] && DRY=true
done

if ! $DRY; then
  if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: torch with CUDA required (or pass --dry-run)" >&2
    exit 2
  fi
fi

python3 "$ROOT/scripts/run_intra_sub_chunk_dump_cases.py" \
  --cases-file "$ROOT/intra_sub_chunk_cases.json" \
  --dump-dir "$DUMP_DIR" \
  --phase "$PHASE" \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}"
