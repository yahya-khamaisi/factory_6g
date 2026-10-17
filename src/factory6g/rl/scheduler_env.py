"""Uplink scheduling environment for reinforcement learning.

Each step is one Monte Carlo batch of the coded OFDM uplink used by the
scheduler stage: the terminals send a neutral probe, the base station estimates
the channel with the adaptive estimator, the agent chooses which ``num_active``
terminals transmit and at which power level, and the data transmission is
decoded. The reward is computed from the decoded bits, so the agent is trained
on the same PHY chain it is evaluated on.

An episode is a sequence of batches at one operating point and one channel
model. A served-rate history carried across the steps of an episode drives the
fairness debt in the state, so the decision problem is sequential.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from factory6g.models.model import Model
from factory6g.models.resource_manager import ResourceDirectives
from factory6g.sim.stages.common import extract_error_stats, per_ut_delivered_bits, transmitted_ut_mask

# Power levels available to the agent for an active terminal (fraction of the
# maximum output power).
POWER_LEVELS = (0.5, 1.0)
HISTORY_ALPHA = 0.15
RATE_FLOOR = 1e-3


def fairness_debt(avg_rates: np.ndarray) -> np.ndarray:
    """Normalised inverse served-rate history; 1 marks the most starved terminal."""
    inv = 1.0 / np.maximum(avg_rates, RATE_FLOOR)
    return inv / max(float(np.max(inv)), 1e-12)


def policy_state(h_hat: Any, ebno_db: float, debt: np.ndarray) -> np.ndarray:
    """State matrix [num_ut, fft_size + 2].

    Columns: log10 of the channel energy per subcarrier, averaged over receive
    antennas, OFDM symbols and all slots of the batch; the fairness debt; and
    Eb/No in dB divided by 20.
    """
    h = np.asarray(h_hat.numpy() if hasattr(h_hat, "numpy") else h_hat)
    # h_hat: [batch, rx, rx_ant, tx, stream, ofdm_sym, fft]
    energy = np.mean(np.abs(h) ** 2, axis=(0, 1, 2, 4, 5))  # -> [tx, fft]
    log_energy = np.log10(np.maximum(energy, 1e-9)).astype(np.float32)
    num_ut = log_energy.shape[0]
    extra = np.stack([debt.astype(np.float32), np.full(num_ut, ebno_db / 20.0, np.float32)], axis=1)
    return np.concatenate([log_energy, extra], axis=1)


@dataclass
class StepOutcome:
    reward: float
    bit_errors: int
    total_bits: int
    delivered: np.ndarray
    tx_power_fraction: float


class SchedulerEnv:
    """Sequential scheduling environment over the simulator.

    Args:
        system_config: flat runtime config (``Factory6GConfig.system_runtime_config``).
        estimator_kwargs: kwargs for the adaptive estimator.
        objective: ``"throughput"`` or ``"reliability"`` (see ``reward``).
        channel_models: channel models sampled per episode.
        ebno_grid: operating points sampled per episode.
        batch_size: slots per step (matches the evaluation batch).
        episode_length: steps per episode.
        num_active: terminals the agent must activate per step.
        rng: numpy Generator for episode sampling.
    """

    def __init__(
        self,
        system_config: dict[str, Any],
        estimator_kwargs: dict[str, Any],
        *,
        objective: str,
        channel_models: tuple[str, ...] = ("rayleigh", "rician", "tr38901"),
        ebno_grid: tuple[float, ...] = tuple(float(v) for v in range(0, 21, 2)),
        batch_size: int = 20,
        episode_length: int = 16,
        num_active: int = 2,
        rng: np.random.Generator | None = None,
    ) -> None:
        if objective not in {"throughput", "reliability"}:
            raise ValueError(f"Unknown objective '{objective}'.")
        self.objective = objective
        self.channel_models = channel_models
        self.ebno_grid = ebno_grid
        self.batch_size = int(batch_size)
        self.episode_length = int(episode_length)
        self.num_active = int(num_active)
        self.num_ut = int(system_config.get("num_ut", 4))
        self.rng = rng or np.random.default_rng()
        self._models: dict[str, Model] = {}
        for ch in channel_models:
            cfg = copy.deepcopy(system_config)
            cfg["channel_model_type"] = ch
            # Training never sees evaluation positions of the ray-traced bank.
            cfg["rt_bank_split"] = "train"
            self._models[ch] = Model(cfg, estimator_type="adaptive", estimator_kwargs=estimator_kwargs)
        self._info_bits = int(self._models[channel_models[0]]._transmitter.num_info_bits)
        self._step_index = 0
        self._channel = channel_models[0]
        self._ebno_db = ebno_grid[0]
        self._avg_rates = np.full(self.num_ut, RATE_FLOOR)
        self._context = None

    # ------------------------------------------------------------------ API
    def reset(self) -> np.ndarray:
        self._channel = self.channel_models[int(self.rng.integers(len(self.channel_models)))]
        self._ebno_db = float(self.ebno_grid[int(self.rng.integers(len(self.ebno_grid)))])
        self._avg_rates = np.full(self.num_ut, RATE_FLOOR)
        self._step_index = 0
        return self._observe()

    def step(self, selected: list[int], power_high: list[int]) -> tuple[np.ndarray, float, bool, StepOutcome]:
        """Apply one decision.

        Args:
            selected: indices of the ``num_active`` terminals to activate.
            power_high: per selected terminal, 1 for full power, 0 for the low level.
        """
        mask = [0] * self.num_ut
        power = [0.0] * self.num_ut
        for u, hi in zip(selected, power_high):
            mask[u] = 1
            power[u] = POWER_LEVELS[1] if hi else POWER_LEVELS[0]
        directives = ResourceDirectives(active_ut_mask=mask, per_ut_power=power, pilot_reuse_factor=1)
        model = self._models[self._channel]
        result = model.run_batch(self._context, directives=directives, include_details=True)
        active = transmitted_ut_mask(directives, num_ut=self.num_ut)
        stats = extract_error_stats(result["bits"], result["bits_hat"], ut_mask=active)
        delivered = np.asarray(per_ut_delivered_bits(result["bits"], result["bits_hat"], ut_mask=active), dtype=float)
        tx_fraction = float(sum(power)) / float(self.num_active)
        outcome = StepOutcome(
            reward=0.0,
            bit_errors=int(stats["bit_errors"]),
            total_bits=int(stats["total_bits"]),
            delivered=delivered,
            tx_power_fraction=tx_fraction,
        )
        outcome.reward = self.reward(outcome, fairness_debt(self._avg_rates))
        # Served-rate history from the activation mask alone (1 if served), so
        # the scheduler can maintain the identical history at evaluation, where
        # it never sees decoded bits.
        served_rate = np.asarray(mask, dtype=float)
        self._avg_rates = (1.0 - HISTORY_ALPHA) * self._avg_rates + HISTORY_ALPHA * np.maximum(served_rate, RATE_FLOOR)
        self._step_index += 1
        done = self._step_index >= self.episode_length
        obs = self._observe() if not done else np.zeros_like(self._last_obs)
        return obs, outcome.reward, done, outcome

    def reward(self, outcome: StepOutcome, debt_before: np.ndarray) -> float:
        """Per-step reward.

        throughput: delivered-bit fraction of the served terminals, each weighted
            by its fairness debt before the decision (a proportional-fair style
            objective: serving a starved terminal pays more).
        reliability: -log10 of the batch BER over the served terminals, mapped
            to [0, 1] between BER 1e-1 and 1e-6 (zero errors counts as 1e-6),
            minus 0.1 times the fraction of maximum transmit power used.
        """
        per_ut_capacity = float(self.batch_size * self._info_bits)
        if self.objective == "throughput":
            served = outcome.delivered > 0
            weights = np.where(served, debt_before, 0.0)
            return float(np.sum(weights * outcome.delivered) / (self.num_active * per_ut_capacity))
        ber = outcome.bit_errors / max(outcome.total_bits, 1)
        ber = min(max(ber, 1e-6), 1e-1)
        reliability = (-np.log10(ber) - 1.0) / 5.0
        return float(reliability - 0.1 * outcome.tx_power_fraction)

    # ------------------------------------------------------------ internals
    def _observe(self) -> np.ndarray:
        model = self._models[self._channel]
        self._context = model.prepare_batch_context(self.batch_size, self._ebno_db, include_feedback=True)
        self._last_obs = policy_state(self._context.feedback.h_hat, self._ebno_db, fairness_debt(self._avg_rates))
        return self._last_obs
