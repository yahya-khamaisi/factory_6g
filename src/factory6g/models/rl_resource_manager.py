"""Scheduler that runs a PPO-trained policy (see ``factory6g.rl``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from factory6g.models.resource_manager import ResourceDirectives, ResourceManager, _num_ut, _state_key
from factory6g.rl.scheduler_env import HISTORY_ALPHA, POWER_LEVELS, RATE_FLOOR, fairness_debt, policy_state
from factory6g.rl.ppo import RL_CHECKPOINT_FORMAT, greedy_action
from factory6g.sim.types import ResourceManagerFeedback


def is_rl_checkpoint(path: str | Path | None) -> bool:
    if not path:
        return False
    meta = Path(path) / "metadata.json"
    if not meta.exists():
        return False
    return json.loads(meta.read_text(encoding="utf-8")).get("format") == RL_CHECKPOINT_FORMAT


class RLResourceManager(ResourceManager):
    """Deterministic evaluation of a PPO policy.

    State and served-rate history are built exactly as in training
    (``factory6g.rl.scheduler_env``); the history is kept per operating point.
    """

    def __init__(self, model_path: str, num_active: int = 2, **_: Any) -> None:
        from factory6g.rl.ppo import load_actor_critic

        self.needs_channel_feedback = True
        self.num_active = int(num_active)
        self.model_path = str(model_path)
        self.model = load_actor_critic(model_path)
        self.metadata = json.loads((Path(model_path) / "metadata.json").read_text(encoding="utf-8"))
        self._avg_rates_by_state: dict[float, np.ndarray] = {}

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = _num_ut(config)
        key = _state_key(ebno_db)
        avg = self._avg_rates_by_state.get(key)
        if avg is None or avg.shape[0] != num_ut:
            avg = np.full(num_ut, RATE_FLOOR)
        if feedback is None:
            raise ValueError("RLResourceManager needs channel feedback.")
        state = policy_state(feedback.h_hat, float(ebno_db), fairness_debt(avg))
        import torch

        with torch.no_grad():
            sched, power, _ = self.model(state[None])
        selected, power_high = greedy_action(sched.numpy()[0], power.numpy()[0], self.num_active)
        mask = [0] * num_ut
        per_ut_power = [0.0] * num_ut
        for u, hi in zip(selected, power_high):
            mask[u] = 1
            per_ut_power[u] = POWER_LEVELS[1] if hi else POWER_LEVELS[0]
        self._avg_rates_by_state[key] = (1.0 - HISTORY_ALPHA) * avg + HISTORY_ALPHA * np.maximum(
            np.asarray(mask, dtype=float), RATE_FLOOR
        )
        return ResourceDirectives(active_ut_mask=mask, per_ut_power=per_ut_power, pilot_reuse_factor=1)
