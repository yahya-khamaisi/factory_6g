"""Greedy evaluation of the trained PPO scheduling policies (Chapter 3, "Training outcome").

For each policy, channel model and operating point, runs greedy episodes in the
training environment (the same state, action and served-rate history as at
evaluation) and reports what the policy learned: the Jain index of the bits
delivered over an episode, the share of full-power decisions, and how often the
selected terminals are the ones with the largest estimated channel energy
(chance level 0.5 for two of four terminals).

Usage (inside Docker):
    python scripts/tools/evaluate_rl_policy.py --output results/rl_eval/policy_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)

from factory6g.rl.ppo import load_actor_critic
from factory6g.sim.rng import seed_all

from factory6g.rl.ppo import greedy_action
from factory6g.rl.scheduler_env import SchedulerEnv
from factory6g.sim.config import load_config
from factory6g.sim.stages.common import resolve_kwargs


def jain(x: np.ndarray) -> float:
    s = float(np.sum(x))
    return s * s / (len(x) * float(np.sum(x * x))) if s > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--channels", default="rayleigh,rician,tr38901,raytraced")
    parser.add_argument("--ebno", default="0,10,20")
    parser.add_argument("--episodes", type=int, default=2, help="Greedy episodes per channel and Eb/N0 point.")
    parser.add_argument("--output", default="results/rl_eval/policy_eval.json")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.simulation.seed
    rm = config.resource_managers
    policies = {
        "drl": ("throughput", rm.drl_model_path),
        "reliability_drl": ("reliability", rm.kwargs.get("reliability_drl", {}).get("model_path")),
    }
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    ebno_points = [float(v) for v in args.ebno.split(",")]

    out: dict[str, dict] = {}
    for name, (objective, path) in policies.items():
        net = load_actor_critic(path)
        out[name] = {"model_path": path, "results": {}}
        for channel in channels:
            for eb in ebno_points:
                env = SchedulerEnv(
                    config.system_runtime_config,
                    resolve_kwargs(config.estimators.kwargs, "adaptive"),
                    objective=objective,
                    channel_models=(channel,),
                    ebno_grid=(eb,),
                    batch_size=config.monte_carlo.batch_size,
                    episode_length=16,
                    num_active=rm.num_active_users,
                    rng=np.random.default_rng(seed),
                )
                seed_all(seed)
                jains, full_power, top_energy, ber_err, ber_bits = [], [], [], 0, 0
                for _ in range(args.episodes):
                    obs, done = env.reset(), False
                    delivered = np.zeros(env.num_ut)
                    while not done:
                        sched, power, _ = net(obs[None])
                        selected, power_high = greedy_action(sched.numpy()[0], power.numpy()[0], env.num_active)
                        energy = obs[:, :-2].mean(axis=1)
                        top = set(np.argsort(-energy)[: env.num_active].tolist())
                        top_energy.append(len(top & set(selected)) / env.num_active)
                        full_power.extend(power_high)
                        obs, _, done, outcome = env.step(selected, power_high)
                        delivered += outcome.delivered
                        ber_err += outcome.bit_errors
                        ber_bits += outcome.total_bits
                    jains.append(jain(delivered))
                key = f"{channel}@{eb:g}dB"
                out[name]["results"][key] = {
                    "jain_index": float(np.nanmean(jains)),
                    "full_power_share": float(np.mean(full_power)),
                    "top_energy_selection_share": float(np.mean(top_energy)),
                    "ber": ber_err / max(ber_bits, 1),
                }
                print(name, key, out[name]["results"][key], flush=True)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
