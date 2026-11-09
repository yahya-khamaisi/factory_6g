"""Proximal policy optimisation for the uplink scheduling environment.

Policy
------
The actor keeps the shared-trunk architecture used for the learned schedulers
(layer normalisation, two 1-D convolutions across terminals, a per-terminal
dense layer, a pooled context vector concatenated back onto the per-terminal
features). Its heads output, per terminal, a scheduling logit ``l_u`` and a
power logit ``m_u``; a value head reads the pooled context.

Action distribution
-------------------
The ``A`` active terminals are drawn without replacement from
``softmax(l)`` (a Plackett--Luce draw), and each selected terminal independently
takes the high power level with probability ``sigmoid(m_u)``. The log-probability
of the sampled ordered selection and power choices is used in the PPO ratio.
At evaluation the policy is deterministic: the ``A`` largest ``l_u`` are
selected and the high power level is used where ``sigmoid(m_u) >= 0.5``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

RL_CHECKPOINT_FORMAT = "factory6g_rl_policy_v1"


@dataclass
class PPOConfig:
    iterations: int = 150
    steps_per_iteration: int = 256
    gamma: float = 0.9
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    hidden_dim: int = 128
    seed: int = 42


def build_actor_critic(num_ut: int, state_dim: int, hidden_dim: int = 128):
    """PyTorch actor-critic with the architecture of the TensorFlow version (see module docstring)."""
    import torch
    from torch import nn

    class ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_ut, self.state_dim, self.hidden_dim = int(num_ut), int(state_dim), int(hidden_dim)
            # Keras LayerNormalization uses epsilon 1e-3; kept so converted weights match.
            self.input_norm = nn.LayerNorm(state_dim, eps=1e-3)
            self.conv_1 = nn.Conv1d(state_dim, 64, 3, padding=1)
            self.conv_2 = nn.Conv1d(64, 64, 3, padding=1)
            self.user_dense = nn.Linear(64, hidden_dim)
            self.context_dense = nn.Linear(hidden_dim, hidden_dim)
            self.schedule_td = nn.Linear(2 * hidden_dim, 1)
            self.power_td = nn.Linear(2 * hidden_dim, 1)
            self.value_head = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = torch.as_tensor(x, dtype=torch.float32)
            x = self.input_norm(x)                                   # [B, U, S]
            x = torch.relu(self.conv_1(x.transpose(1, 2)))
            x = torch.relu(self.conv_2(x)).transpose(1, 2)           # [B, U, 64]
            user = torch.relu(self.user_dense(x))                    # [B, U, H]
            context = torch.relu(self.context_dense(user.mean(dim=1)))
            joint = torch.cat([user, context[:, None, :].expand(-1, user.shape[1], -1)], dim=-1)
            return (self.schedule_td(joint)[..., 0], self.power_td(joint)[..., 0], self.value_head(context)[..., 0])

    return ActorCritic()


def load_actor_critic_from_keras(path: str | Path, num_ut: int, state_dim: int, hidden_dim: int = 128):
    """Convert a policy trained on the TensorFlow stack (``policy.keras`` or ``policy.weights.h5``)."""
    import torch

    from factory6g.torch_io import dense_from_keras, keras_layer_weights

    net = build_actor_critic(num_ut, state_dim, hidden_dim)
    layers = dict(keras_layer_weights(path))
    norm = layers.pop("layer_normalization")
    net.input_norm.weight.data.copy_(torch.as_tensor(norm[0]))
    net.input_norm.bias.data.copy_(torch.as_tensor(norm[1]))
    for conv, name in ((net.conv_1, "conv1d"), (net.conv_2, "conv1d_1")):
        kernel, bias = layers.pop(name)
        conv.weight.data.copy_(torch.as_tensor(np.transpose(kernel, (2, 1, 0)).copy()))
        conv.bias.data.copy_(torch.as_tensor(bias))
    # Dense layers in creation order: user, context, schedule, power, value.
    dense_names = sorted((n for n in layers if n.startswith("dense")), key=lambda n: int(n.split("_")[1]) if "_" in n else 0)
    targets = (net.user_dense, net.context_dense, net.schedule_td, net.power_td, net.value_head)
    if len(dense_names) != len(targets):
        raise ValueError(f"Expected 5 dense layers in {path}, found {dense_names}.")
    for lin, name in zip(targets, dense_names):
        w, b = dense_from_keras(*layers[name])
        if tuple(w.shape) != tuple(lin.weight.shape):
            raise ValueError(f"{name}: kernel {tuple(w.shape)} does not fit {tuple(lin.weight.shape)}.")
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
    return net.eval()


def load_actor_critic(model_dir: str | Path):
    """A trained policy: ``policy.pt`` (PyTorch stack) or a converted ``policy.keras`` (TensorFlow stack)."""
    import torch

    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    dims = meta.get("network") or {}
    num_ut = int(dims.get("num_ut", 4))
    state_dim = int(dims.get("state_dim", 0)) or None
    hidden = int(dims.get("hidden_dim", meta.get("ppo", {}).get("hidden_dim", 128)))
    if (model_dir / "policy.pt").exists():
        sd = torch.load(model_dir / "policy.pt", map_location="cpu", weights_only=True)
        state_dim = int(sd["input_norm.weight"].shape[0])
        num_ut = int(dims.get("num_ut", num_ut))
        net = build_actor_critic(num_ut, state_dim, hidden)
        net.load_state_dict(sd)
        return net.eval()
    src = model_dir / "policy.keras"
    if not src.exists():
        src = model_dir / "policy.weights.h5"
    from factory6g.torch_io import keras_layer_weights

    state_dim = int(dict(keras_layer_weights(src))["layer_normalization"][0].shape[0])
    return load_actor_critic_from_keras(src, num_ut, state_dim, hidden)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    return x - np.log(np.sum(np.exp(x)))


def sample_action(sched_logits: np.ndarray, power_logits: np.ndarray, num_active: int, rng: np.random.Generator):
    """Draw an ordered selection and power choices; return them with their log-probability."""
    remaining = list(range(len(sched_logits)))
    selected: list[int] = []
    logp = 0.0
    for _ in range(num_active):
        lp = _log_softmax(sched_logits[remaining])
        k = int(rng.choice(len(remaining), p=np.exp(lp)))
        logp += float(lp[k])
        selected.append(remaining.pop(k))
    power_high: list[int] = []
    for u in selected:
        p = 1.0 / (1.0 + np.exp(-power_logits[u]))
        hi = int(rng.random() < p)
        power_high.append(hi)
        logp += float(np.log(p if hi else 1.0 - p))
    return selected, power_high, logp


def greedy_action(sched_logits: np.ndarray, power_logits: np.ndarray, num_active: int):
    selected = [int(i) for i in np.argsort(-sched_logits)[:num_active]]
    power_high = [int(power_logits[u] >= 0.0) for u in selected]
    return selected, power_high


def _batch_logp_entropy(torch, sched, power, selected, power_high, num_active):
    """Log-probability of stored ordered selections under new logits (vectorised)."""
    F = torch.nn.functional
    num_ut = sched.shape[-1]
    selected = torch.as_tensor(selected, dtype=torch.long)
    power_high = torch.as_tensor(power_high, dtype=torch.float32)
    logp = torch.zeros(sched.shape[0])
    available = torch.ones_like(sched)
    for k in range(num_active):
        masked = torch.where(available > 0, sched, torch.full_like(sched, -1e9))
        logsm = F.log_softmax(masked, dim=-1)
        pick = selected[:, k]
        logp = logp + logsm.gather(1, pick[:, None])[:, 0]
        available = available * (1.0 - F.one_hot(pick, num_ut).to(sched.dtype))
    for k in range(num_active):
        pick = selected[:, k]
        m = power.gather(1, pick[:, None])[:, 0]
        hi = power_high[:, k]
        logp = logp + hi * F.logsigmoid(m) + (1.0 - hi) * F.logsigmoid(-m)
    first = torch.softmax(sched, dim=-1)
    entropy = -torch.sum(first * torch.log(first + 1e-12), dim=-1)
    pw = torch.sigmoid(power)
    entropy = entropy + torch.mean(-(pw * torch.log(pw + 1e-12) + (1 - pw) * torch.log(1 - pw + 1e-12)), dim=-1)
    return logp, entropy


def completed_iterations(output_dir: str | Path) -> int:
    """Iterations already completed by a (possibly partial) run in ``output_dir``."""
    state = Path(output_dir) / "train_state.json"
    if not state.exists():
        return 0
    return int(json.loads(state.read_text(encoding="utf-8"))["iteration"])


def train_ppo(
    env,
    cfg: PPOConfig,
    output_dir: str | Path,
    *,
    max_new_iterations: int | None = None,
    log_every: int = 1,
) -> dict[str, Any]:
    """Train an actor-critic with PPO on ``env``, resuming from ``output_dir`` if a
    partial run is there.

    A run may be split into segments (``max_new_iterations``) executed in fresh
    processes. Each segment restores the network weights, the Adam state, the
    action sampler's and the episode sampler's generator states, and the
    training history, and saves them again at its end, so memory retained by the
    simulator does not accumulate across the whole run. The caller reseeds the
    simulator's generators per segment (see ``train_rl_scheduler.py``).
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = completed_iterations(output_dir)
    rng = np.random.default_rng(cfg.seed)
    history: list[dict[str, float]] = []
    ep_returns: list[float] = []
    elapsed_before = 0.0
    if start > 0:
        state = json.loads((output_dir / "train_state.json").read_text(encoding="utf-8"))
        rng.bit_generator.state = state["action_rng"]
        env.rng.bit_generator.state = state["episode_rng"]
        history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
        ep_returns = state.get("recent_episode_returns", [])
        elapsed_before = float(state.get("elapsed_s", 0.0))
        if "lagrange" in state and hasattr(env, "lagrange"):
            env.lagrange = {float(k): float(v) for k, v in state["lagrange"].items()}
    obs = env.reset()
    num_ut, state_dim = obs.shape
    torch.manual_seed(cfg.seed)  # network initialisation
    net = build_actor_critic(num_ut, state_dim, cfg.hidden_dim)
    # Keras Adam's epsilon (1e-7), so the optimiser matches the TensorFlow stack.
    opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, eps=1e-7)
    if start > 0:
        net.load_state_dict(torch.load(output_dir / "policy.pt", map_location="cpu", weights_only=True))
        opt.load_state_dict(torch.load(output_dir / "optimizer.pt", map_location="cpu", weights_only=True))
    t0 = time.perf_counter() - elapsed_before
    ep_return = 0.0
    end = cfg.iterations if max_new_iterations is None else min(cfg.iterations, start + max_new_iterations)

    for it in range(start, end):
        buf = {k: [] for k in ("obs", "sel", "pw", "logp", "rew", "val", "done")}
        errors = bits = 0
        for _ in range(cfg.steps_per_iteration):
            with torch.no_grad():
                sched, power, value = net(obs[None])
            sel, pwh, logp = sample_action(sched.numpy()[0], power.numpy()[0], env.num_active, rng)
            nxt, rew, done, out = env.step(sel, pwh)
            errors += out.bit_errors
            bits += out.total_bits
            for k, v in zip(("obs", "sel", "pw", "logp", "rew", "val", "done"),
                            (obs, sel, pwh, logp, rew, float(value.numpy()[0]), done)):
                buf[k].append(v)
            ep_return += rew
            if done:
                ep_returns.append(ep_return)
                ep_return = 0.0
                obs = env.reset()
            else:
                obs = nxt
        with torch.no_grad():
            _, _, last_val = net(obs[None])
        rew = np.asarray(buf["rew"], np.float32)
        val = np.asarray(buf["val"] + [float(last_val.numpy()[0])], np.float32)
        dones = np.asarray(buf["done"], np.float32)
        adv = np.zeros_like(rew)
        gae = 0.0
        for t in reversed(range(len(rew))):
            nonterminal = 1.0 - dones[t]
            delta = rew[t] + cfg.gamma * val[t + 1] * nonterminal - val[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            adv[t] = gae
        ret = adv + val[:-1]
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)

        o = np.asarray(buf["obs"], np.float32)
        s = np.asarray(buf["sel"], np.int32)
        p = np.asarray(buf["pw"], np.int32)
        lp_old = np.asarray(buf["logp"], np.float32)
        idx = np.arange(len(rew))
        for _ in range(cfg.update_epochs):
            rng.shuffle(idx)
            for start in range(0, len(idx), cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                with torch.enable_grad():
                    sched, power, value = net(o[mb])
                    logp, ent = _batch_logp_entropy(torch, sched, power, s[mb], p[mb], env.num_active)
                    ratio = torch.exp(logp - torch.as_tensor(lp_old[mb]))
                    a = torch.as_tensor(adv_n[mb])
                    pg = -torch.mean(torch.minimum(ratio * a, torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * a))
                    vl = torch.mean((torch.as_tensor(ret[mb]) - value) ** 2)
                    loss = pg + cfg.value_coef * vl - cfg.entropy_coef * torch.mean(ent)
                    opt.zero_grad()
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        constraint = {}
        if getattr(env, "objective", None) == "constrained":
            observed = env.dual_update()
            constraint = {"rollout_bler_by_ebno": observed, "lagrange": {str(k): v for k, v in env.lagrange.items()}}
        rec = {
            "iteration": it + 1,
            "mean_step_reward": float(rew.mean()),
            "mean_episode_return": float(np.mean(ep_returns[-20:])) if ep_returns else float("nan"),
            "rollout_ber": float(errors / max(bits, 1)),
            "elapsed_s": time.perf_counter() - t0,
            **constraint,
        }
        history.append(rec)
        if (it + 1) % log_every == 0:
            print(json.dumps(rec), flush=True)
    save_rl_checkpoint(output_dir, net, cfg, env, history)
    torch.save(opt.state_dict(), output_dir / "optimizer.pt")
    (output_dir / "train_state.json").write_text(json.dumps({
        "iteration": end,
        "action_rng": rng.bit_generator.state,
        "episode_rng": env.rng.bit_generator.state,
        "recent_episode_returns": ep_returns[-20:],
        "elapsed_s": time.perf_counter() - t0,
        **({"lagrange": {str(k): v for k, v in env.lagrange.items()}} if hasattr(env, "lagrange") else {}),
    }), encoding="utf-8")
    return {"history": history, "completed": end}


def save_rl_checkpoint(output_dir: Path, net, cfg: PPOConfig, env, history: list[dict[str, float]]) -> None:
    from datetime import datetime, timezone

    import torch

    torch.save(net.state_dict(), output_dir / "policy.pt")
    meta = {
        "format": RL_CHECKPOINT_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "PPO (clipped surrogate, GAE)",
        "framework": "pytorch",
        "network": {"num_ut": net.num_ut, "state_dim": net.state_dim, "hidden_dim": net.hidden_dim},
        "objective": env.objective,
        "ppo": asdict(cfg),
        "environment": {
            "channel_models": list(env.channel_models),
            "ebno_grid_db": list(env.ebno_grid),
            "batch_size": env.batch_size,
            "episode_length": env.episode_length,
            "num_active": env.num_active,
            "power_levels": [0.5, 1.0],
            "estimator": "adaptive",
        },
        **({"constraint": {"bler_target": env.bler_target, "dual_step": env.dual_step, "lagrange_cap": env.lagrange_cap,
                           "lagrange": {str(k): v for k, v in env.lagrange.items()}}}
           if getattr(env, "objective", None) == "constrained" else {}),
        "environment_steps": len(history) * cfg.steps_per_iteration,
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
