#!/usr/bin/env bash
# Phase 1 evidence: Rayleigh cells of Run B (schedulers) and Run C (estimators)
# at seeds 42-46 with the per-batch trace on. Everything else equals the locked
# configs (config/config_run_b.json, config/run_c_estimator_matrix.json).
# Usage: scripts/tools/run_phase1_seeds.sh b|c   (one lane per run; two lanes can run side by side)
set -euo pipefail
cd "$(dirname "$0")/../.."
lane="${1:?usage: $0 b|c}"
RMS=static,round_robin,max_throughput,pf,wmmse,queue_aware,drl,reliability_drl
for seed in 42 43 44 45 46; do
  if [[ "$lane" == b ]]; then
    docker compose run --rm simulation --config "config/phase1/run_b_seed${seed}.json" \
      --resource-managers "$RMS" --channel rayleigh
  else
    docker compose run --rm simulation --config "config/phase1/run_c_seed${seed}.json" \
      --estimators ls,dft,lmmse,adaptive,neural --channel rayleigh --stage estimators
  fi
done
