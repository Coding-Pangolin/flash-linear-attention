#!/usr/bin/env bash
# Replay chunk_kda from saved GPU dump (001_chunk_kda_fwd.pt) on CUDA
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: torch with CUDA required" >&2
  exit 2
fi

python3 "$ROOT/scripts/run_kda_replay_dump.py" "$@"
