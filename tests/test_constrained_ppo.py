"""Constrained PPO: per-Eb/N0 Lagrange multipliers, the constrained reward, and resume."""

from __future__ import annotations

import json

import numpy as np
import pytest

from factory6g.models.resource_manager import create_resource_manager
from factory6g.models.rl_resource_manager import RLResourceManager
from factory6g.rl.ppo import RL_CHECKPOINT_FORMAT, PPOConfig, build_actor_critic, train_ppo
from factory6g.rl.scheduler_env import SchedulerEnv, StepOutcome


def _bare_env(target: float = 0.01) -> SchedulerEnv:
    env = SchedulerEnv.__new__(SchedulerEnv)
    env.objective = "constrained"
    env.batch_size, env._info_bits, env.num_active = 2, 100, 2
    env.bler_target, env.dual_step, env.lagrange_cap = target, 2.0, 10.0
    env.lagrange = {0.0: 0.0, 12.0: 0.0}
    env._blocks_by_ebno = {}
    env._ebno_db = 12.0
    return env


def test_constrained_reward_trades_delivery_against_bler():
    env = _bare_env()
    out = StepOutcome(reward=0.0, bit_errors=5, total_bits=400, delivered=np.array([200.0, 150.0]), tx_power_fraction=1.0, block_errors=1, total_blocks=4)
    assert env.reward(out, np.ones(2)) == pytest.approx(350 / 400)
    env.lagrange[12.0] = 2.0
    assert env.reward(out, np.ones(2)) == pytest.approx(350 / 400 - 2.0 * 0.25)


def test_dual_update_per_operating_point_with_cap():
    env = _bare_env(target=0.01)
    env._blocks_by_ebno = {0.0: [90, 100], 12.0: [0, 100]}
    observed = env.dual_update()
    assert observed == {"0.0": 0.9, "12.0": 0.0}
    assert env.lagrange[0.0] == pytest.approx(2.0 * (0.9 - 0.01))
    assert env.lagrange[12.0] == 0.0  # meets the target: multiplier stays at zero
    for _ in range(20):
        env._blocks_by_ebno = {0.0: [90, 100]}
        env.dual_update()
    assert env.lagrange[0.0] == 10.0  # infeasible point is capped


def test_constrained_drl_routes_to_the_rl_manager(tmp_path):
    build_actor_critic(4, 130).save(tmp_path / "policy.keras")
    (tmp_path / "metadata.json").write_text(json.dumps({"format": RL_CHECKPOINT_FORMAT}))
    mgr = create_resource_manager("constrained_drl", num_ut=4, num_active=2, cnn_model_path=None,
                                  drl_model_path=None, manager_kwargs={"model_path": str(tmp_path)})
    assert isinstance(mgr, RLResourceManager)


@pytest.mark.slow
def test_two_iterations_on_the_real_simulator_resume_the_multipliers(tmp_path):
    from factory6g.sim.config import load_config

    from .conftest import make_tiny_config, write_config

    cfg = load_config(write_config(tmp_path, make_tiny_config(str(tmp_path / "results"))))
    env = SchedulerEnv(cfg.system_runtime_config, {}, objective="constrained", channel_models=("rayleigh",),
                       ebno_grid=(0.0, 6.0), batch_size=1, episode_length=2, num_active=1, rng=np.random.default_rng(0))
    ppo = PPOConfig(iterations=2, steps_per_iteration=4, minibatch_size=2, update_epochs=1, hidden_dim=16)
    train_ppo(env, ppo, tmp_path / "policy", max_new_iterations=1)
    state = json.loads((tmp_path / "policy" / "train_state.json").read_text())
    assert set(state["lagrange"]) == {"0.0", "6.0"}
    history = json.loads((tmp_path / "policy" / "history.json").read_text())
    assert "lagrange" in history[0] and "rollout_bler_by_ebno" in history[0]
    env.lagrange = {0.0: 0.0, 6.0: 0.0}
    train_ppo(env, ppo, tmp_path / "policy")  # resumes iteration 2 and restores the multipliers first
    meta = json.loads((tmp_path / "policy" / "metadata.json").read_text())
    assert meta["objective"] == "constrained" and meta["constraint"]["bler_target"] == 0.01
