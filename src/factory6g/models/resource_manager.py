from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from factory6g.sim.types import ResourceManagerFeedback


_EPS = 1e-9


@dataclass(frozen=True)
class ResourceDirectives:
    active_ut_mask: list[int] | None = None
    per_ut_power: list[float] | None = None
    pilot_reuse_factor: int | None = None


class ResourceManager:
    """Base scheduling/power-control interface used by the benchmark loop."""

    needs_channel_feedback: bool = False

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        return ResourceDirectives()


class StaticResourceManager(ResourceManager):
    def __init__(
        self,
        active_ut_mask: list[int] | None = None,
        per_ut_power: list[float] | None = None,
        pilot_reuse_factor: int | None = None,
    ) -> None:
        self._active_ut_mask = active_ut_mask
        self._per_ut_power = per_ut_power
        self._pilot_reuse_factor = pilot_reuse_factor

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        return ResourceDirectives(
            active_ut_mask=self._active_ut_mask,
            per_ut_power=self._per_ut_power,
            pilot_reuse_factor=self._pilot_reuse_factor,
        )


class RoundRobinResourceManager(ResourceManager):
    """Cyclic activation of ``num_active`` terminals at unit power.

    The pointer is kept per operating point (keyed by Eb/N0 like the PF and
    queue-aware state), so the pair sequence one Eb/N0 point sees does not depend
    on how many other points are still open in the benchmark loop. A point seen
    for the first time starts from a rotating offset, so a sweep of distinct
    Eb/N0 values (dataset generation samples one per record) still rotates.
    """

    def __init__(self, num_active: int = 1) -> None:
        self.num_active = max(1, int(num_active))
        self._index_by_state: dict[float, int] = {}
        self._next_start_index = 0

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = _num_ut(config)
        state_key = _state_key(ebno_db)
        current_index = self._index_by_state.get(state_key)
        if current_index is None:
            current_index = self._next_start_index
            self._next_start_index = (current_index + self.num_active) % num_ut
        mask = [0] * num_ut
        for offset in range(min(self.num_active, num_ut)):
            mask[(current_index + offset) % num_ut] = 1
        self._index_by_state[state_key] = (current_index + self.num_active) % num_ut
        return ResourceDirectives(active_ut_mask=mask)


def _feedback_h_hat_numpy(feedback: ResourceManagerFeedback) -> np.ndarray:
    h_hat = feedback.h_hat
    if hasattr(h_hat, "numpy"):
        h_hat = h_hat.numpy()
    return np.asarray(h_hat, dtype=np.complex64)


def _channel_power_per_user(feedback: ResourceManagerFeedback) -> np.ndarray:
    h_hat = _feedback_h_hat_numpy(feedback)
    power = np.abs(h_hat) ** 2
    reduced = power.sum(axis=(1, 2, 4, 5, 6))
    return reduced.mean(axis=0).astype(np.float64)


def _instantaneous_rate_per_user(feedback: ResourceManagerFeedback, ebno_db: float) -> np.ndarray:
    avg_power = _channel_power_per_user(feedback)
    snr_linear = max(10.0 ** (float(ebno_db) / 10.0), _EPS)
    return np.log2(1.0 + avg_power * snr_linear)


def _normalized_rate_metric(feedback: ResourceManagerFeedback, ebno_db: float) -> np.ndarray:
    inst_rates = _instantaneous_rate_per_user(feedback, ebno_db)
    scale = max(float(np.max(inst_rates, initial=0.0)), _EPS)
    return inst_rates / scale


def _effective_gain_matrix(feedback: ResourceManagerFeedback) -> np.ndarray:
    h_hat = _feedback_h_hat_numpy(feedback)
    if h_hat.ndim != 7:
        raise ValueError(f"Expected h_hat with 7 dimensions, got shape {h_hat.shape}.")

    mean_h = np.mean(h_hat, axis=0)
    user_first = np.moveaxis(mean_h, 2, 0)
    vectors = user_first.reshape(user_first.shape[0], -1)
    gram = vectors @ vectors.conj().T

    num_ut = gram.shape[0]
    direct = np.maximum(np.real(np.diag(gram)), _EPS)
    gains = np.zeros((num_ut, num_ut), dtype=np.float64)
    diag_indices = np.diag_indices(num_ut)
    gains[diag_indices] = direct
    for rx_idx in range(num_ut):
        denom = max(direct[rx_idx], _EPS)
        for tx_idx in range(num_ut):
            if rx_idx == tx_idx:
                continue
            gains[rx_idx, tx_idx] = float((abs(gram[rx_idx, tx_idx]) ** 2) / denom)

    return gains / max(float(np.mean(direct)), _EPS)


def _num_ut(config: dict[str, Any]) -> int:
    return max(1, int(config.get("num_ut", 8)))


def _mask_from_indices(indices: Sequence[int], num_ut: int) -> list[int]:
    mask = [0] * num_ut
    for idx in indices:
        if 0 <= int(idx) < num_ut:
            mask[int(idx)] = 1
    return mask


def _top_indices(metric: np.ndarray, count: int) -> np.ndarray:
    if metric.size == 0:
        return np.array([], dtype=np.int64)
    active_count = max(1, min(int(count), metric.size))
    return np.argsort(metric)[::-1][:active_count]


def _normalize_power(
    power: np.ndarray,
    mask: Sequence[int],
    *,
    max_power: float = 1.0,
    min_active_power: float = 0.15,
) -> list[float]:
    power_arr = np.clip(np.asarray(power, dtype=np.float64), 0.0, None)
    mask_arr = np.asarray(mask, dtype=bool)
    out = np.zeros_like(power_arr, dtype=np.float64)
    if not np.any(mask_arr):
        return out.tolist()

    active_power = power_arr[mask_arr]
    peak = max(float(np.max(active_power, initial=0.0)), _EPS)
    normalized = (active_power / peak) * max_power
    normalized = np.clip(normalized, min_active_power, max_power)
    out[mask_arr] = normalized
    return out.tolist()


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-3)
    shifted = np.asarray(values, dtype=np.float64) / temp
    shifted = shifted - float(np.max(shifted, initial=0.0))
    exp_values = np.exp(shifted)
    return exp_values / max(float(np.sum(exp_values)), _EPS)


def _state_key(ebno_db: float) -> float:
    return round(float(ebno_db), 6)


class MaxThroughputResourceManager(ResourceManager):
    def __init__(self, num_active: int = 1) -> None:
        self.needs_channel_feedback = True
        self.num_active = max(1, int(num_active))

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = _num_ut(config)
        if feedback is None:
            return ResourceDirectives(active_ut_mask=[1] * num_ut)
        avg_power = _channel_power_per_user(feedback)
        top_indices = _top_indices(avg_power, self.num_active)
        mask = _mask_from_indices(top_indices.tolist(), num_ut)
        return ResourceDirectives(active_ut_mask=mask)


class ProportionalFairResourceManager(ResourceManager):
    def __init__(self, num_active: int = 1, alpha: float = 0.9) -> None:
        self.needs_channel_feedback = True
        self.num_active = max(1, int(num_active))
        self.alpha = float(alpha)
        # Rate history is kept per Eb/No point, as the queue-aware and learned
        # managers keep theirs. A single shared history would make the decision
        # at one operating point depend on the order the sweep visited the
        # others, which is not a property of proportional fair scheduling.
        self._avg_rates_by_state: dict[float, np.ndarray] = {}

    def avg_rates_for(self, ebno_db: float) -> np.ndarray | None:
        """Return the rate history held for one operating point, if any."""
        return self._avg_rates_by_state.get(_state_key(ebno_db))

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = _num_ut(config)
        state_key = _state_key(ebno_db)
        avg_rates = self._avg_rates_by_state.get(state_key)
        if avg_rates is None or avg_rates.shape[0] != num_ut:
            avg_rates = np.full(num_ut, 1e-3, dtype=np.float64)
        if feedback is None:
            self._avg_rates_by_state[state_key] = avg_rates
            return ResourceDirectives(active_ut_mask=[1] * num_ut)

        inst_rates = _instantaneous_rate_per_user(feedback, ebno_db)
        pf_metric = inst_rates / np.maximum(avg_rates, _EPS)
        top_indices = _top_indices(pf_metric, self.num_active)
        mask = _mask_from_indices(top_indices.tolist(), num_ut)

        for idx in range(num_ut):
            if mask[idx]:
                avg_rates[idx] = (1.0 - self.alpha) * avg_rates[idx] + self.alpha * inst_rates[idx]
            else:
                avg_rates[idx] = (1.0 - self.alpha) * avg_rates[idx]
        self._avg_rates_by_state[state_key] = avg_rates
        return ResourceDirectives(active_ut_mask=mask)


class WMMSEResourceManager(ResourceManager):
    """Scalar effective-channel WMMSE scheduler/power allocator."""

    def __init__(
        self,
        num_active: int = 1,
        iterations: int = 12,
        damping: float = 0.5,
        max_power: float = 1.0,
        min_active_power: float = 0.15,
        user_weights: Sequence[float] | None = None,
    ) -> None:
        self.needs_channel_feedback = True
        self.num_active = max(1, int(num_active))
        self.iterations = max(1, int(iterations))
        self.damping = float(np.clip(damping, 0.0, 0.99))
        self.max_power = float(max_power)
        self.min_active_power = float(min_active_power)
        self.user_weights = None if user_weights is None else np.asarray(user_weights, dtype=np.float64)

    def _weights(self, num_ut: int) -> np.ndarray:
        if self.user_weights is None:
            return np.ones(num_ut, dtype=np.float64)
        if self.user_weights.shape[0] != num_ut:
            raise ValueError(
                f"WMMSE user_weights length {self.user_weights.shape[0]} does not match num_ut={num_ut}."
            )
        return np.maximum(self.user_weights, _EPS)

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = _num_ut(config)
        if feedback is None:
            mask = _mask_from_indices(range(min(self.num_active, num_ut)), num_ut)
            power = _normalize_power(np.ones(num_ut, dtype=np.float64), mask, max_power=self.max_power)
            return ResourceDirectives(active_ut_mask=mask, per_ut_power=power, pilot_reuse_factor=1)

        gains = _effective_gain_matrix(feedback)
        weights = self._weights(num_ut)
        diag_gain = np.maximum(np.diag(gains), _EPS)
        noise = 1.0 / max(10.0 ** (float(ebno_db) / 10.0), _EPS)

        v = np.full(num_ut, np.sqrt(max(self.max_power, _EPS)) * 0.5, dtype=np.float64)
        for _ in range(self.iterations):
            total_rx = gains @ (v**2) + noise
            signal = np.sqrt(diag_gain) * v
            u = signal / np.maximum(total_rx, _EPS)
            mse = 1.0 - 2.0 * u * signal + (u**2) * total_rx
            w = weights / np.maximum(mse, _EPS)
            denom = gains.T @ (w * (u**2))
            updated_v = (w * u * np.sqrt(diag_gain)) / np.maximum(denom, _EPS)
            updated_v = np.clip(updated_v, 0.0, np.sqrt(max(self.max_power, _EPS)))
            v = self.damping * v + (1.0 - self.damping) * updated_v

        power = v**2
        interference = gains @ power - diag_gain * power
        weighted_rate = weights * np.log2(1.0 + (diag_gain * power) / np.maximum(noise + interference, _EPS))
        selected = _top_indices(weighted_rate, self.num_active)
        mask = _mask_from_indices(selected.tolist(), num_ut)
        power_out = _normalize_power(
            power,
            mask,
            max_power=self.max_power,
            min_active_power=self.min_active_power,
        )
        return ResourceDirectives(active_ut_mask=mask, per_ut_power=power_out, pilot_reuse_factor=1)


class QueueAwareLyapunovResourceManager(ResourceManager):
    """
    Virtual-queue drift-plus-penalty scheduler.

    The current runtime does not expose real packet queues, so this manager
    maintains synthetic backlog state per Eb/No point to approximate
    queue-aware scheduling pressure.
    """

    def __init__(
        self,
        num_active: int = 1,
        arrival_rate: float | Sequence[float] = 0.35,
        utility_weight: float = 0.2,
        queue_cap: float = 10.0,
        initial_queue: float = 0.0,
        max_power: float = 1.0,
        min_active_power: float = 0.2,
    ) -> None:
        self.needs_channel_feedback = True
        self.num_active = max(1, int(num_active))
        self.arrival_rate = arrival_rate
        self.utility_weight = float(utility_weight)
        self.queue_cap = float(queue_cap)
        self.initial_queue = float(initial_queue)
        self.max_power = float(max_power)
        self.min_active_power = float(min_active_power)
        self._queues_by_state: dict[float, np.ndarray] = {}

    def _arrivals(self, num_ut: int) -> np.ndarray:
        if np.isscalar(self.arrival_rate):
            return np.full(num_ut, float(self.arrival_rate), dtype=np.float64)
        arrivals = np.asarray(list(self.arrival_rate), dtype=np.float64)
        if arrivals.shape[0] != num_ut:
            raise ValueError(
                f"Queue-aware arrival_rate length {arrivals.shape[0]} does not match num_ut={num_ut}."
            )
        return arrivals

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = _num_ut(config)
        state_key = _state_key(ebno_db)
        queues = self._queues_by_state.get(state_key)
        if queues is None or queues.shape[0] != num_ut:
            queues = np.full(num_ut, self.initial_queue, dtype=np.float64)

        queues = np.minimum(queues + self._arrivals(num_ut), self.queue_cap)
        if feedback is None:
            score = queues + self.utility_weight
            rate_metric = np.ones(num_ut, dtype=np.float64)
        else:
            rate_metric = _normalized_rate_metric(feedback, ebno_db)
            score = (queues + self.utility_weight) * rate_metric

        selected = _top_indices(score, self.num_active)
        mask = _mask_from_indices(selected.tolist(), num_ut)
        mask_arr = np.asarray(mask, dtype=np.float64)
        served = mask_arr * rate_metric
        queues = np.maximum(queues - served, 0.0)
        self._queues_by_state[state_key] = queues

        power_basis = score + 0.25 * queues
        power_out = _normalize_power(
            power_basis,
            mask,
            max_power=self.max_power,
            min_active_power=self.min_active_power,
        )
        return ResourceDirectives(active_ut_mask=mask, per_ut_power=power_out, pilot_reuse_factor=1)


class DRLResourceManager(ResourceManager):
    """
    Actor-style RM interface with optional Keras policy loading.

    If no trained policy artifact is supplied, the manager falls back to a
    lightweight heuristic actor so it remains runnable inside the simulator.
    """

    def __init__(
        self,
        num_active: int = 1,
        model_path: str | None = None,
        temperature: float = 0.75,
        fairness_weight: float = 0.35,
        history_alpha: float = 0.15,
        max_power: float = 1.0,
        min_active_power: float = 0.2,
        policy_score_weight: float = 1.0,
        channel_gain_weight: float = 0.0,
    ) -> None:
        self.needs_channel_feedback = True
        self.num_active = max(1, int(num_active))
        self.temperature = float(temperature)
        self.fairness_weight = float(fairness_weight)
        self.history_alpha = float(np.clip(history_alpha, 0.01, 1.0))
        self.max_power = float(max_power)
        self.min_active_power = float(min_active_power)
        self.policy_score_weight = float(policy_score_weight)
        self.channel_gain_weight = float(channel_gain_weight)
        self.model = None
        self.model_path = model_path
        self.policy_checkpoint = None
        self._avg_rate_by_state: dict[float, np.ndarray] = {}

        if model_path:
            try:
                from factory6g.models.drl_policy import load_policy_checkpoint

                self.policy_checkpoint = load_policy_checkpoint(model_path)
                self.model = self.policy_checkpoint.model
                print(f"Loaded DRL Resource Manager checkpoint from {model_path}")
            except Exception as exc:
                print(f"Failed to load DRL resource manager model from {model_path}: {exc}")
                print("DRLResourceManager will use heuristic actor fallback.")

    def _predict_policy(
        self,
        channel_energy: np.ndarray,
        fairness_debt: np.ndarray,
        ebno_db: float,
        num_ut: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.policy_checkpoint is None:
            raise RuntimeError("No DRL policy model loaded.")

        from factory6g.models.drl_policy import build_policy_state, predict_policy_outputs

        state = build_policy_state(channel_energy, ebno_db, fairness_debt=fairness_debt)
        outputs = predict_policy_outputs(self.policy_checkpoint, state)
        sched_np = np.asarray(outputs["schedule_output"], dtype=np.float64).reshape(-1)
        power_np = np.asarray(outputs["power_output"], dtype=np.float64).reshape(-1)
        if sched_np.shape[0] < num_ut:
            sched_np = np.pad(sched_np, (0, num_ut - sched_np.shape[0]), constant_values=float(np.min(sched_np)))
        if power_np.shape[0] < num_ut:
            power_np = np.pad(power_np, (0, num_ut - power_np.shape[0]), constant_values=0.0)
        return sched_np[:num_ut], power_np[:num_ut]

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = _num_ut(config)
        if feedback is None:
            mask = _mask_from_indices(range(min(self.num_active, num_ut)), num_ut)
            power = _normalize_power(np.ones(num_ut, dtype=np.float64), mask, max_power=self.max_power)
            return ResourceDirectives(active_ut_mask=mask, per_ut_power=power, pilot_reuse_factor=1)

        state_key = _state_key(ebno_db)
        avg_rates = self._avg_rate_by_state.get(state_key)
        if avg_rates is None or avg_rates.shape[0] != num_ut:
            avg_rates = np.full(num_ut, 1e-3, dtype=np.float64)

        inst_rates = _instantaneous_rate_per_user(feedback, ebno_db)
        norm_rates = inst_rates / max(float(np.max(inst_rates, initial=0.0)), _EPS)
        fairness_debt = 1.0 / np.maximum(avg_rates, 1e-3)
        fairness_debt = fairness_debt / max(float(np.max(fairness_debt, initial=0.0)), _EPS)

        if self.policy_checkpoint is not None:
            from factory6g.models.drl_policy import channel_energy_from_h_hat, project_policy_to_directives

            channel_energy = channel_energy_from_h_hat(feedback.h_hat)[0]
            sched_scores, power_scores = self._predict_policy(channel_energy, fairness_debt, ebno_db, num_ut)
            sched_scores = (
                self.policy_score_weight * sched_scores
                + self.channel_gain_weight * norm_rates
                + self.fairness_weight * fairness_debt
            )
            power_scores = self.policy_score_weight * power_scores + self.channel_gain_weight * norm_rates
            mask, power_out = project_policy_to_directives(
                sched_scores,
                power_scores,
                num_active=self.num_active,
                max_power=self.max_power,
                min_active_power=self.min_active_power,
            )
        else:
            logits = norm_rates + self.fairness_weight * fairness_debt
            policy_probs = _softmax(logits, self.temperature)
            sched_scores = policy_probs + 0.5 * norm_rates
            power_scores = 0.5 * policy_probs + 0.5 * norm_rates + 0.25 * self.fairness_weight * fairness_debt
            selected = _top_indices(sched_scores, self.num_active)
            mask = _mask_from_indices(selected.tolist(), num_ut)
            power_out = _normalize_power(
                power_scores,
                mask,
                max_power=self.max_power,
                min_active_power=self.min_active_power,
            )

        served = np.asarray(mask, dtype=np.float64) * inst_rates
        avg_rates = (1.0 - self.history_alpha) * avg_rates + self.history_alpha * np.maximum(served, 1e-3)
        self._avg_rate_by_state[state_key] = avg_rates
        return ResourceDirectives(active_ut_mask=mask, per_ut_power=power_out, pilot_reuse_factor=1)


def _learned_manager(num_active: int, kwargs: dict[str, Any]) -> ResourceManager:
    """PPO checkpoints load as RLResourceManager; legacy actor checkpoints as DRLResourceManager."""
    from factory6g.models.rl_resource_manager import RLResourceManager, is_rl_checkpoint

    if is_rl_checkpoint(kwargs.get("model_path")):
        return RLResourceManager(num_active=num_active, **kwargs)
    return DRLResourceManager(num_active=num_active, **kwargs)


def create_resource_manager(
    name: str,
    *,
    num_ut: int,
    num_active: int,
    cnn_model_path: str | None,
    drl_model_path: str | None,
    manager_kwargs: dict[str, Any] | None = None,
) -> ResourceManager:
    name_lower = name.lower()
    kwargs = dict(manager_kwargs or {})
    if (
        "reliability_drl" in name_lower
        or "reliability-drl" in name_lower
        or "reliabilitydrl" in name_lower
    ):
        kwargs.setdefault("model_path", "models/reliability_drl_resource_manager_policy")
        return _learned_manager(num_active, kwargs)
    if "constrained_drl" in name_lower or "constrained-drl" in name_lower:
        kwargs.setdefault("model_path", "models/constrained_drl_ppo_policy")
        return _learned_manager(num_active, kwargs)
    if "static" in name_lower:
        return StaticResourceManager(active_ut_mask=[1] * num_ut, per_ut_power=[1.0] * num_ut)
    if "round" in name_lower:
        return RoundRobinResourceManager(num_active=num_active, **kwargs)
    if "wmmse" in name_lower:
        return WMMSEResourceManager(num_active=num_active, **kwargs)
    if "queue" in name_lower or "lyapunov" in name_lower or "backpressure" in name_lower:
        return QueueAwareLyapunovResourceManager(num_active=num_active, **kwargs)
    if "max" in name_lower:
        return MaxThroughputResourceManager(num_active=num_active, **kwargs)
    if "prop" in name_lower or "pf" in name_lower:
        return ProportionalFairResourceManager(num_active=num_active, **kwargs)
    if "drl" in name_lower or "ppo" in name_lower or "actor" in name_lower:
        kwargs.setdefault("model_path", drl_model_path)
        return _learned_manager(num_active, kwargs)
    if "cnn" in name_lower:
        from factory6g.models.cnn_resource_manager import CNNResourceManager

        kwargs.setdefault("model_path", cnn_model_path)
        return CNNResourceManager(**kwargs)
    raise ValueError(f"Unknown resource manager: {name}")
