"""PPO scheduler: action sampling, checkpoint routing and inference directives."""

import json

import numpy as np
import torch

from factory6g.models.resource_manager import create_resource_manager
from factory6g.models.rl_resource_manager import RLResourceManager
from factory6g.rl.ppo import RL_CHECKPOINT_FORMAT, build_actor_critic, greedy_action, sample_action
from factory6g.rl.scheduler_env import fairness_debt
from factory6g.sim.types import ResourceManagerFeedback


def test_sample_action_selects_distinct_terminals_with_valid_logprob():
    rng = np.random.default_rng(0)
    sel, pw, logp = sample_action(np.array([0.1, 2.0, -1.0, 0.5]), np.zeros(4), 2, rng)
    assert len(set(sel)) == 2 and all(0 <= u < 4 for u in sel)
    assert set(pw) <= {0, 1} and logp < 0.0


def test_greedy_action_takes_top_logits():
    sel, pw = greedy_action(np.array([0.1, 2.0, -1.0, 0.5]), np.array([1.0, -1.0, 0.0, 2.0]), 2)
    assert sel == [1, 3] and pw == [0, 1]


def test_fairness_debt_marks_starved_terminal():
    d = fairness_debt(np.array([1.0, 0.5, 1e-3, 1.0]))
    assert d[2] == 1.0 and d[0] < d[1] < d[2]


def test_rl_checkpoint_routes_to_rl_manager(tmp_path):
    net = build_actor_critic(4, 130)
    torch.save(net.state_dict(), tmp_path / "policy.pt")
    (tmp_path / "metadata.json").write_text(json.dumps({"format": RL_CHECKPOINT_FORMAT}))
    mgr = create_resource_manager(
        "drl", num_ut=4, num_active=2, cnn_model_path=None, drl_model_path=str(tmp_path)
    )
    assert isinstance(mgr, RLResourceManager)
    h = (np.random.default_rng(1).standard_normal((2, 1, 8, 4, 1, 14, 128)) + 0j).astype(np.complex64)
    fb = ResourceManagerFeedback(h_hat=h, err_var=np.zeros_like(h.real))
    d = mgr.get_runtime_directives({"num_ut": 4}, 4.0, feedback=fb)
    assert sum(d.active_ut_mask) == 2
    assert all(p in (0.0, 0.5, 1.0) for p in d.per_ut_power)
