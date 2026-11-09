#!/usr/bin/env bash
# Phase 2 Industry 5.0 campaign: four device classes, body blockage and three scripted
# disruptions, every scheduler including the constrained PPO, on the Rayleigh and
# ray-traced channels. DATA_DIR defaults to ./data (the channel bank).
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-$PWD/data}"
RMS=static,round_robin,max_throughput,pf,wmmse,queue_aware,drl,reliability_drl,constrained_drl
for ch in rayleigh raytraced; do
  docker run --rm -v "$PWD":/app -v "$DATA_DIR":/app/data:ro -w /app factory6g:latest \
    --config config/industry50/config_industry50.json --resource-managers "$RMS" --channel "$ch"
done
