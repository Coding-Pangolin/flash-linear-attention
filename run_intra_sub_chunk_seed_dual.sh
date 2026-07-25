#!/usr/bin/env bash
# GPU seed dual: CPU-RNG inputs + optional CPU golden / GPU kernel.
# Same --seed / --phase / --names as NPU ./run_isub_seed_dual.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: run_intra_sub_chunk_seed_dual.sh [options]

  --phase smoke|all|...
  --seed N              base seed (case i → N+i*9973)
  --names a,b
  --run-cpu             run CPU golden (default)
  --no-cpu              skip CPU
  --cpu-only            CPU only (no CUDA)
  --cpu-dtype fp32|fp64
  --save-cpu            write cpu_golden.pt per case
  --dry-run
EOF
}

EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

# cpu-only does not need CUDA
NEED_CUDA=true
for a in "${EXTRA[@]+"${EXTRA[@]}"}"; do
  [[ "$a" == "--cpu-only" || "$a" == "--dry-run" ]] && NEED_CUDA=false
done
if $NEED_CUDA; then
  if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA required (or --cpu-only / --dry-run)" >&2
    exit 2
  fi
fi

exec python3 "$ROOT/scripts/run_intra_sub_chunk_seed_dual.py" "${EXTRA[@]}"
