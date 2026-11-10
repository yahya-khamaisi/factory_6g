#!/usr/bin/env bash
# Phase 5: Runs A-D on the PyTorch stack (factory6g-torch image), written to
# results/phase5. Seed 42 on every locked channel, plus Rayleigh at seeds 43-46.
# Runs B and C at seed 42 on Rayleigh are the Phase 4 gate runs and are not repeated.
# Each job is one run x channel x seed; a finished job leaves results/phase5/done/<job>,
# so a lane that is restarted skips what is done.
# Usage: scripts/tools/run_phase5.sh 1|2|smoke   (two lanes run side by side, 4 threads each)
set -euo pipefail
cd "$(dirname "$0")/../.."
lane="${1:?usage: $0 1|2|smoke}"
DATA_DIR="${DATA_DIR:-$PWD/data}"
RMS_ALL=static,round_robin,max_throughput,pf,wmmse,queue_aware,drl,reliability_drl

args_for() {  # run -> CLI arguments besides --config/--channel
  case "$1" in
    a) echo "--estimators neural,ls --resource-managers max_throughput" ;;
    b) echo "--resource-managers $RMS_ALL" ;;
    c) echo "--estimators ls,dft,lmmse,adaptive,neural --stage estimators" ;;
    d) echo "--estimators ls,adaptive,neural,perfect --resource-managers static,max_throughput,pf,wmmse,drl,reliability_drl --stage coupled" ;;
  esac
}

run_job() {  # run channel seed [config]
  local run=$1 ch=$2 seed=$3 cfg=${4:-config/phase5/run_${1}_seed${3}.json}
  local job="${run}_${ch}_s${seed}" done_dir=results/phase5/done
  mkdir -p "$done_dir"
  [[ -f "$done_dir/$job" && "$lane" != smoke ]] && { echo "skip $job"; return; }
  echo "== $job $(date '+%F %T')"
  # shellcheck disable=SC2046
  docker run --rm --name "phase5_${lane}_${job}" -v "$PWD":/app -v "$DATA_DIR":/app/data:ro -w /app \
    -e PYTHONPATH=/app/src -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    -e F6G_COMPILE_DECODER="${F6G_COMPILE_DECODER:-1}" factory6g-torch:latest \
    --config "$cfg" --channel "$ch" $(args_for "$run")
  [[ "$lane" != smoke ]] && date '+%F %T' > "$done_dir/$job"
  return 0
}

case "$lane" in
  smoke)  # one batch at one Eb/N0 point per run and channel, to catch port errors early
    mkdir -p results/phase5/smoke
    for run in a b c d; do
      python3 - "$run" <<'EOF'
import json, sys
run = sys.argv[1]
d = json.load(open(f"config/phase5/run_{run}_seed42.json"))
d["simulation"]["output_dir"] = "results/phase5/smoke"
d["monte_carlo"].update(min_batches=1, max_batches=1, min_error_batches=0, ebno_min=6.0, ebno_max=6.0)
json.dump(d, open(f"results/phase5/smoke/run_{run}.json", "w"), indent=2)
EOF
      for ch in rayleigh rician tr38901 raytraced; do
        run_job "$run" "$ch" 42 "results/phase5/smoke/run_${run}.json"
      done
    done ;;
  1)  # the coupled run is the longest: seed 42 on every channel, then the Rayleigh seeds
    for ch in rayleigh rician tr38901 raytraced; do run_job d "$ch" 42; done
    for s in 43 44 45 46; do run_job d rayleigh "$s"; done ;;
  2)
    for ch in rayleigh raytraced; do run_job a "$ch" 42; done
    for ch in rician tr38901 raytraced; do run_job b "$ch" 42; done
    for ch in rician tr38901 raytraced; do run_job c "$ch" 42; done
    for s in 43 44 45 46; do run_job a rayleigh "$s"; run_job b rayleigh "$s"; run_job c rayleigh "$s"; done ;;
  *) echo "unknown lane $lane" >&2; exit 2 ;;
esac
