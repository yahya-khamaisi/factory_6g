#!/usr/bin/env bash
# Phase 4 equivalence gate: Run B (schedulers) or Run C (estimators), Rayleigh,
# seed 42, on the PyTorch stack (factory6g-torch image). Compared with the
# Phase 1 TensorFlow seeds 42-46 by `python -m factory6g.analysis.equivalence`.
# Usage: scripts/tools/run_phase4_gate.sh b|c   (DATA_DIR defaults to ./data)
set -euo pipefail
cd "$(dirname "$0")/../.."
lane="${1:?usage: $0 b|c}"
DATA_DIR="${DATA_DIR:-$PWD/data}"
RMS=static,round_robin,max_throughput,pf,wmmse,queue_aware,drl,reliability_drl
args=(--config "config/phase4/run_${lane}_seed42_torch.json" --channel rayleigh)
if [[ "$lane" == b ]]; then
  args+=(--resource-managers "$RMS")
else
  args+=(--estimators ls,dft,lmmse,adaptive,neural --stage estimators)
fi
docker run --rm --name "phase4_gate_${lane}" -v "$PWD":/app -v "$DATA_DIR":/app/data:ro -w /app \
  -e PYTHONPATH=/app/src -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
  -e F6G_COMPILE_DECODER="${F6G_COMPILE_DECODER:-1}" factory6g-torch:latest "${args[@]}"
