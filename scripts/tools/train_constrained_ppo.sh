#!/bin/sh
# Constrained PPO (Phase 2): six 25-iteration segments, each in a fresh container,
# the same protocol and budget as the reliability policy (150 x 256 steps).
# DATA_DIR defaults to ./data; set it when the channel bank lives elsewhere.
cd "$(dirname "$0")/../.." || exit 1
DATA_DIR="${DATA_DIR:-$PWD/data}"
out=models/constrained_drl_ppo_policy
for seg in 1 2 3 4 5 6; do
  docker run --rm --name ppo_constrained_seg$seg \
    -v "$PWD/src:/app/src:ro" -v "$PWD/scripts:/app/scripts:ro" \
    -v "$PWD/config:/app/config:ro" -v "$DATA_DIR:/app/data:ro" -v "$PWD/models:/app/models" \
    --entrypoint python factory6g:latest scripts/tools/train_rl_scheduler.py \
    --objective constrained --output-dir $out --segment-iterations 25 \
    --bler-target 0.01 --dual-step 2.0 \
    --channels rayleigh,rician,tr38901,raytraced || exit 1
done
