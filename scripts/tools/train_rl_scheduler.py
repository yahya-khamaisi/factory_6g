#!/usr/bin/env python3
"""Train a learned scheduler with PPO against the Factory6G simulator.

Example (inside Docker):
    python scripts/tools/train_rl_scheduler.py --objective throughput \\
        --output-dir models/drl_resource_manager_policy
    python scripts/tools/train_rl_scheduler.py --objective reliability \\
        --output-dir models/reliability_drl_resource_manager_policy
"""

from __future__ import annotations

import argparse
import random

from factory6g.sim.env import configure_env

configure_env(force_cpu=True, gpu_num=0)

import numpy as np
import sionna.phy
import tensorflow as tf

from factory6g.rl.ppo import PPOConfig, completed_iterations, train_ppo
from factory6g.rl.scheduler_env import SchedulerEnv
from factory6g.sim.config import load_config
from factory6g.sim.stages.common import resolve_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--objective", choices=["throughput", "reliability"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--steps-per-iteration", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=None, help="Slots per step (default: monte_carlo.batch_size).")
    parser.add_argument("--episode-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None, help="Default: simulation.seed.")
    parser.add_argument(
        "--channels",
        default="rayleigh,rician,tr38901,raytraced",
        help="Comma-separated channel models sampled per episode.",
    )
    parser.add_argument(
        "--segment-iterations",
        type=int,
        default=None,
        help="Run at most this many iterations, then save the full training state and exit; "
        "rerun the same command to continue. Default: run to completion.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.simulation.seed
    # Simulator generators are reseeded per segment from the seed and the number
    # of completed iterations, so a segmented run is reproducible.
    sim_seed = seed + completed_iterations(args.output_dir)
    random.seed(sim_seed)
    np.random.seed(sim_seed)
    tf.random.set_seed(sim_seed)
    sionna.phy.config.seed = sim_seed

    env = SchedulerEnv(
        config.system_runtime_config,
        resolve_kwargs(config.estimators.kwargs, "adaptive"),
        objective=args.objective,
        batch_size=args.batch_size or config.monte_carlo.batch_size,
        episode_length=args.episode_length,
        num_active=config.resource_managers.num_active_users,
        channel_models=tuple(c.strip().lower() for c in args.channels.split(",") if c.strip()),
        # Episodes sample the operating points of the evaluation sweep.
        ebno_grid=tuple(float(v) for v in config.monte_carlo.ebno_db_range),
        rng=np.random.default_rng(seed),
    )
    cfg = PPOConfig(iterations=args.iterations, steps_per_iteration=args.steps_per_iteration, seed=seed)
    result = train_ppo(env, cfg, args.output_dir, max_new_iterations=args.segment_iterations)
    print(f"Completed {result['completed']}/{cfg.iterations} iterations; saved to {args.output_dir}")


if __name__ == "__main__":
    main()
