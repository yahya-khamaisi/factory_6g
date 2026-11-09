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
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = layers.Input(shape=(num_ut, state_dim), name="policy_state")
    x = layers.LayerNormalization(name="input_norm")(inputs)
    x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv_1")(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv_2")(x)
    user = layers.Dense(hidden_dim, activation="relu", name="user_dense")(x)
    pooled = layers.GlobalAveragePooling1D(name="global_pool")(user)
    context = layers.Dense(hidden_dim, activation="relu", name="context_dense")(pooled)
    joint = layers.Concatenate(axis=-1, name="joint_features")(
        [user, layers.RepeatVector(num_ut, name="context_repeat")(context)]
    )
    sched = layers.Reshape((num_ut,), name="schedule_logits")(layers.Dense(1, name="schedule_td")(joint))
    power = layers.Reshape((num_ut,), name="power_logits")(layers.Dense(1, name="power_td")(joint))
    value = layers.Reshape((), name="value")(layers.Dense(1, name="value_head")(context))
    return tf.keras.Model(inputs=inputs, outputs=[sched, power, value], name="factory6g_rl_scheduler")


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


def _batch_logp_entropy(tf, sched, power, selected, power_high, num_active):
    """Log-probability of stored ordered selections under new logits (vectorised)."""
    num_ut = sched.shape[-1]
    logp = tf.zeros(tf.shape(sched)[0])
    available = tf.ones_like(sched)
    for k in range(num_active):
        masked = tf.where(available > 0, sched, tf.fill(tf.shape(sched), -1e9))
        logsm = tf.nn.log_softmax(masked, axis=-1)
        pick = selected[:, k]
        logp += tf.gather(logsm, pick, batch_dims=1)
        available = available * (1.0 - tf.one_hot(pick, num_ut))
    for k in range(num_active):
        pick = selected[:, k]
        m = tf.gather(power, pick, batch_dims=1)
        hi = tf.cast(power_high[:, k], tf.float32)
        logp += hi * tf.math.log_sigmoid(m) + (1.0 - hi) * tf.math.log_sigmoid(-m)
    first = tf.nn.softmax(sched, axis=-1)
    entropy = -tf.reduce_sum(first * tf.math.log(first + 1e-12), axis=-1)
    pw = tf.sigmoid(power)
    entropy += tf.reduce_mean(-(pw * tf.math.log(pw + 1e-12) + (1 - pw) * tf.math.log(1 - pw + 1e-12)), axis=-1)
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
    import tensorflow as tf

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
    net = build_actor_critic(num_ut, state_dim, cfg.hidden_dim)
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    opt.build(net.trainable_variables)
    if start > 0:
        net.load_weights(output_dir / "policy.weights.h5")
        opt_state = np.load(output_dir / "optimizer.npz")
        for var, key in zip(opt.variables, sorted(opt_state.files, key=lambda k: int(k.split("_")[1]))):
            var.assign(opt_state[key])
    t0 = time.perf_counter() - elapsed_before
    ep_return = 0.0
    end = cfg.iterations if max_new_iterations is None else min(cfg.iterations, start + max_new_iterations)

    for it in range(start, end):
        buf = {k: [] for k in ("obs", "sel", "pw", "logp", "rew", "val", "done")}
        errors = bits = 0
        for _ in range(cfg.steps_per_iteration):
            sched, power, value = net(obs[None], training=False)
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
        _, _, last_val = net(obs[None], training=False)
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
                with tf.GradientTape() as tape:
                    sched, power, value = net(o[mb], training=True)
                    logp, ent = _batch_logp_entropy(tf, sched, power, s[mb], p[mb], env.num_active)
                    ratio = tf.exp(logp - lp_old[mb])
                    a = adv_n[mb]
                    pg = -tf.reduce_mean(tf.minimum(ratio * a, tf.clip_by_value(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * a))
                    vl = tf.reduce_mean((ret[mb] - value) ** 2)
                    loss = pg + cfg.value_coef * vl - cfg.entropy_coef * tf.reduce_mean(ent)
                grads = tape.gradient(loss, net.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 0.5)
                opt.apply_gradients(zip(grads, net.trainable_variables))

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
    np.savez(output_dir / "optimizer.npz", **{f"var_{i}": v.numpy() for i, v in enumerate(opt.variables)})
    net.save_weights(output_dir / "policy.weights.h5")
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

    net.save(output_dir / "policy.keras")
    meta = {
        "format": RL_CHECKPOINT_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "PPO (clipped surrogate, GAE)",
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
