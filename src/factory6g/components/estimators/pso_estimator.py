"""PSO-tuned structured channel estimator.

This estimator keeps the public ``pso`` estimator name, but replaces the
previous global polynomial fit with a constrained search over physically
meaningful denoisers:

- DFT delay-domain truncation
- Frequency-domain LMMSE smoothing
- A blend of the two

Particle Swarm Optimization (PSO) searches for three global hyperparameters
per call:

- ``tap_ratio``: delay support for DFT truncation
- ``r_freq``: frequency correlation parameter for LMMSE smoothing
- ``blend``: interpolation between DFT and LMMSE candidates

The objective is an unsupervised LS-consistency loss with channel-structure
regularization:

- pilot-symbol consistency with the LS estimate
- delay-domain tail energy beyond the chosen support
- frequency roughness penalty
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def _delay_truncate(h_ls: np.ndarray, tap_count: int) -> np.ndarray:
    h_delay = np.fft.ifft(h_ls, axis=-1)
    h_delay[..., tap_count:] = 0.0
    return np.fft.fft(h_delay, axis=-1).astype(np.complex64)


def _lmmse_smooth(
    h_ls: np.ndarray,
    *,
    noise_linear: float,
    r_freq: float,
    fft_size: int,
    eig_cache: dict[float, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, float]:
    key = round(float(r_freq), 3)
    cached = eig_cache.get(key)
    if cached is None:
        k_indices = np.arange(fft_size)
        delta_k = np.abs(k_indices[:, None] - k_indices[None, :])
        correlation = (key ** delta_k).astype(np.float32)
        eigvals, eigvecs = np.linalg.eigh(correlation.astype(np.float64))
        eigvals = np.maximum(eigvals.astype(np.float32), 1e-9)
        eigvecs = eigvecs.astype(np.complex64)
        eig_cache[key] = (eigvals, eigvecs)
    eigvals, eigvecs = eig_cache[key]

    shrinkage = eigvals / (eigvals + max(float(noise_linear), 1e-9))
    flat = h_ls.reshape(-1, fft_size)
    projected = flat @ eigvecs
    smoothed = (projected * shrinkage) @ np.conjugate(eigvecs.T)
    return smoothed.reshape(h_ls.shape).astype(np.complex64), float(np.mean(shrinkage))


def _pilot_consistency_loss(
    candidate: np.ndarray,
    reference: np.ndarray,
    pilot_symbol_indices: tuple[int, ...],
) -> float:
    if pilot_symbol_indices:
        candidate = candidate[..., list(pilot_symbol_indices), :]
        reference = reference[..., list(pilot_symbol_indices), :]
    signal_power = max(float(np.mean(np.abs(reference) ** 2)), 1e-9)
    return float(np.mean(np.abs(candidate - reference) ** 2) / signal_power)


def _tail_energy_ratio(candidate: np.ndarray, tap_count: int) -> float:
    h_delay = np.fft.ifft(candidate, axis=-1)
    total = max(float(np.mean(np.abs(h_delay) ** 2)), 1e-9)
    if tap_count >= h_delay.shape[-1]:
        return 0.0
    tail = float(np.mean(np.abs(h_delay[..., tap_count:]) ** 2))
    return tail / total


def _roughness_penalty(candidate: np.ndarray) -> float:
    if candidate.shape[-1] < 3:
        return 0.0
    signal_power = max(float(np.mean(np.abs(candidate) ** 2)), 1e-9)
    second_diff = np.diff(candidate, n=2, axis=-1)
    return float(np.mean(np.abs(second_diff) ** 2) / signal_power)


class PSOChannelEstimator(Block):
    """Search a structured DFT/LMMSE blend instead of fitting a polynomial."""

    def __init__(
        self,
        config: dict,
        resource_grid: ResourceGrid,
        degree: int = 3,
        swarm_size: int = 8,
        iters: int = 12,
        inertia_start: float = 0.7,
        inertia_end: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        early_stop_patience: int = 3,
        min_rel_improvement: float = 1e-3,
        seed: int = 42,
        tap_ratio_min: float = 0.35,
        tap_ratio_max: float = 1.0,
        r_freq_min: float = 0.92,
        r_freq_max: float = 0.995,
        tail_weight: float = 2.0,
        roughness_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self._base = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self._rg = resource_grid
        self.swarm_size = int(max(2, swarm_size))
        self.iters = int(max(1, iters))
        self.inertia_start = float(inertia_start)
        self.inertia_end = float(inertia_end)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.early_stop_patience = int(max(1, early_stop_patience))
        self.min_rel_improvement = float(max(min_rel_improvement, 0.0))
        self._rng = np.random.default_rng(seed)
        self.fft_size = int(resource_grid.fft_size)
        self.cp_length = int(resource_grid.cyclic_prefix_length)
        self.pilot_symbol_indices = tuple(int(v) for v in config.get("pilot_ofdm_symbol_indices", []))
        self.tap_ratio_min = float(tap_ratio_min)
        self.tap_ratio_max = float(max(tap_ratio_min, tap_ratio_max))
        self.r_freq_min = float(r_freq_min)
        self.r_freq_max = float(max(r_freq_min, r_freq_max))
        self.tail_weight = float(tail_weight)
        self.roughness_weight = float(roughness_weight)
        self._eig_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self.last_params = {
            "tap_ratio": 1.0,
            "r_freq": 0.98,
            "blend": 0.5,
        }

    def _tap_count(self, tap_ratio: float) -> int:
        ratio = _clip(tap_ratio, self.tap_ratio_min, self.tap_ratio_max)
        raw = int(round(self.cp_length * ratio))
        return int(min(max(raw, 1), self.fft_size))

    def _candidate_from_params(
        self,
        h_ls: np.ndarray,
        *,
        noise_linear: float,
        tap_ratio: float,
        r_freq: float,
        blend: float,
        dft_cache: dict[int, np.ndarray],
        lmmse_cache: dict[float, tuple[np.ndarray, float]],
    ) -> tuple[np.ndarray, float, int]:
        tap_count = self._tap_count(tap_ratio)
        if tap_count not in dft_cache:
            dft_cache[tap_count] = _delay_truncate(h_ls, tap_count=tap_count)
        r_key = round(_clip(r_freq, self.r_freq_min, self.r_freq_max), 3)
        if r_key not in lmmse_cache:
            lmmse_cache[r_key] = _lmmse_smooth(
                h_ls,
                noise_linear=noise_linear,
                r_freq=r_key,
                fft_size=self.fft_size,
                eig_cache=self._eig_cache,
            )
        h_dft = dft_cache[tap_count]
        h_lmmse, shrinkage = lmmse_cache[r_key]
        blend_weight = _clip(blend, 0.0, 1.0)
        candidate = (
            blend_weight * h_lmmse
            + (1.0 - blend_weight) * h_dft
        ).astype(np.complex64)
        err_scale = (
            blend_weight * shrinkage
            + (1.0 - blend_weight) * (float(tap_count) / float(self.fft_size))
        )
        return candidate, float(err_scale), tap_count

    def _objective(
        self,
        candidate: np.ndarray,
        reference: np.ndarray,
        *,
        tap_count: int,
    ) -> float:
        consistency = _pilot_consistency_loss(candidate, reference, self.pilot_symbol_indices)
        tail = _tail_energy_ratio(candidate, tap_count=tap_count)
        roughness = _roughness_penalty(candidate)
        return consistency + self.tail_weight * tail + self.roughness_weight * roughness

    def call(self, y: torch.Tensor, noise_variance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_ls, err_var = self._base(y, noise_variance)

        h_ls_np = h_ls.detach().cpu().numpy().astype(np.complex64)
        avg_no = float(torch.mean(torch.as_tensor(noise_variance, dtype=torch.float32)))
        dft_cache: dict[int, np.ndarray] = {}
        lmmse_cache: dict[float, tuple[np.ndarray, float]] = {}

        lower = np.array([self.tap_ratio_min, self.r_freq_min, 0.0], dtype=np.float32)
        upper = np.array([self.tap_ratio_max, self.r_freq_max, 1.0], dtype=np.float32)
        span = upper - lower

        particles = lower + self._rng.random((self.swarm_size, 3), dtype=np.float32) * span
        velocities = np.zeros_like(particles, dtype=np.float32)
        personal_best = particles.copy()
        personal_values = np.full(self.swarm_size, np.inf, dtype=np.float32)
        global_best = particles[0].copy()
        global_value = float("inf")
        stagnant_steps = 0

        def evaluate(position: np.ndarray) -> tuple[float, float]:
            candidate, err_scale, tap_count = self._candidate_from_params(
                h_ls_np,
                noise_linear=avg_no,
                tap_ratio=float(position[0]),
                r_freq=float(position[1]),
                blend=float(position[2]),
                dft_cache=dft_cache,
                lmmse_cache=lmmse_cache,
            )
            return self._objective(candidate, h_ls_np, tap_count=tap_count), err_scale

        for step in range(self.iters):
            for idx in range(self.swarm_size):
                value, _ = evaluate(particles[idx])
                if value < personal_values[idx]:
                    personal_values[idx] = value
                    personal_best[idx] = particles[idx].copy()
                if value < global_value:
                    prev = global_value
                    global_value = value
                    global_best = particles[idx].copy()
                    if np.isfinite(prev):
                        rel_gain = (prev - value) / max(abs(prev), 1e-12)
                        stagnant_steps = stagnant_steps + 1 if rel_gain < self.min_rel_improvement else 0
                    else:
                        stagnant_steps = 0

            if stagnant_steps >= self.early_stop_patience:
                break

            inertia = self.inertia_start + (self.inertia_end - self.inertia_start) * (
                step / max(1, self.iters - 1)
            )
            r1 = self._rng.random((self.swarm_size, 3), dtype=np.float32)
            r2 = self._rng.random((self.swarm_size, 3), dtype=np.float32)
            velocities = (
                inertia * velocities
                + self.c1 * r1 * (personal_best - particles)
                + self.c2 * r2 * (global_best - particles)
            )
            particles = np.clip(particles + velocities, lower, upper)

        best_candidate, best_err_scale, _ = self._candidate_from_params(
            h_ls_np,
            noise_linear=avg_no,
            tap_ratio=float(global_best[0]),
            r_freq=float(global_best[1]),
            blend=float(global_best[2]),
            dft_cache=dft_cache,
            lmmse_cache=lmmse_cache,
        )
        self.last_params = {
            "tap_ratio": float(global_best[0]),
            "r_freq": float(global_best[1]),
            "blend": float(global_best[2]),
        }

        h_hat = torch.as_tensor(best_candidate, device=h_ls.device).to(h_ls.dtype)
        err_var_out = err_var * best_err_scale
        return h_hat, err_var_out
