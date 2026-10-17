from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid

from .dft_estimator import DFTChannelEstimator
from .lmmse_estimator import LMMSEChannelEstimator


def select_quality_branch(quality: float, quality_low: float, quality_high: float) -> str:
    if quality < quality_low:
        return "low"
    if quality < quality_high:
        return "mid"
    return "high"


_BRANCH_LMMSE_WEIGHT = {"low": 1.0, "high": 0.0}
# Terminal axis of an LS estimate [batch, rx, rx_ant, tx, stream, symbols, subcarriers].
_TERMINAL_AXIS = 3


class AdaptiveHybridChannelEstimator(Block):
    """
    Adaptive LS/DFT/LMMSE hybrid:
    - low quality  -> full LMMSE smoothing
    - mid quality  -> blended DFT + LMMSE
    - high quality -> DFT-only denoising

    With ``per_terminal_branch`` (default) each terminal gets its own quality
    proxy and branch, pooled over the batch, its antenna ports and the grid, so
    a weak terminal is smoothed while a strong one is only truncated. With it
    off, one branch is chosen for the whole batch (the v1 rule).
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        config: dict | None = None,
        quality_low: float = 3.0,
        quality_high: float = 12.0,
        blend_mid_weight: float = 0.5,
        dft_tap_ratio: float = 1.0,
        lmmse_r_freq: float = 0.98,
        noise_bin_db: float = 0.5,
        per_terminal_branch: bool = True,
    ) -> None:
        super().__init__()
        self._ls_estimator = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self._dft = DFTChannelEstimator(resource_grid, dft_tap_ratio=dft_tap_ratio)
        self._lmmse = LMMSEChannelEstimator(
            resource_grid,
            r_freq=lmmse_r_freq,
            noise_bin_db=noise_bin_db,
        )
        self.quality_low = float(quality_low)
        self.quality_high = float(quality_high)
        self.blend_mid_weight = float(min(max(blend_mid_weight, 0.0), 1.0))
        self.per_terminal_branch = bool(per_terminal_branch)
        # Branch of the most recent call: one string, or one per terminal.
        self.last_branch: str | list[str] = "high"

    @staticmethod
    def quality_proxy(h_ls: tf.Tensor, no: tf.Tensor) -> float:
        signal_power = tf.reduce_mean(tf.abs(h_ls) ** 2)
        noise_power = tf.reduce_mean(tf.cast(no, tf.float32))
        ratio = signal_power / tf.maximum(noise_power, tf.constant(1e-12, tf.float32))
        return float(ratio.numpy())

    @staticmethod
    def quality_proxy_per_terminal(h_ls: tf.Tensor, no: tf.Tensor) -> np.ndarray:
        """Eq. quality-proxy evaluated separately for each terminal."""
        axes = [a for a in range(len(h_ls.shape)) if a != _TERMINAL_AXIS]
        signal_power = tf.reduce_mean(tf.abs(h_ls) ** 2, axis=axes)
        noise_power = tf.reduce_mean(tf.cast(no, tf.float32))
        ratio = signal_power / tf.maximum(noise_power, tf.constant(1e-12, tf.float32))
        return ratio.numpy()

    def _blend_weight(self, branch: str) -> float:
        return _BRANCH_LMMSE_WEIGHT.get(branch, self.blend_mid_weight)

    def estimate_from_ls(
        self,
        h_ls: tf.Tensor,
        err_var_ls: tf.Tensor,
        no: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, str | list[str]]:
        if self.per_terminal_branch:
            return self._estimate_per_terminal(h_ls, err_var_ls, no)
        quality = self.quality_proxy(h_ls, no)
        branch = select_quality_branch(quality, self.quality_low, self.quality_high)
        self.last_branch = branch

        if branch == "low":
            h_hat, err_var = self._lmmse.estimate_from_ls(h_ls, err_var_ls, no)
            return h_hat, err_var, branch

        h_dft, err_dft = self._dft.estimate_from_ls(h_ls, err_var_ls)
        if branch == "high":
            return h_dft, err_dft, branch

        h_lmmse, err_lmmse = self._lmmse.estimate_from_ls(h_ls, err_var_ls, no)
        w = tf.cast(self.blend_mid_weight, h_dft.dtype)
        h_mid = w * h_lmmse + (1.0 - w) * h_dft
        err_mid = self.blend_mid_weight * err_lmmse + (1.0 - self.blend_mid_weight) * err_dft
        return h_mid, err_mid, branch

    def _estimate_per_terminal(
        self,
        h_ls: tf.Tensor,
        err_var_ls: tf.Tensor,
        no: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, list[str]]:
        quality = self.quality_proxy_per_terminal(h_ls, no)
        branches = [select_quality_branch(float(q), self.quality_low, self.quality_high) for q in quality]
        self.last_branch = branches
        weights = np.array([self._blend_weight(b) for b in branches], dtype=np.float32)

        # A refinement is formed only if some terminal uses it, so the cost of a
        # batch in which every terminal takes one pure branch is unchanged.
        if np.all(weights == 1.0):
            h_hat, err_var = self._lmmse.estimate_from_ls(h_ls, err_var_ls, no)
            return h_hat, err_var, branches
        h_dft, err_dft = self._dft.estimate_from_ls(h_ls, err_var_ls)
        if np.all(weights == 0.0):
            return h_dft, err_dft, branches
        # Otherwise both are formed for the whole batch and mixed per terminal;
        # a weight of 1 or 0 reproduces that terminal's pure branch exactly.
        h_lmmse, err_lmmse = self._lmmse.estimate_from_ls(h_ls, err_var_ls, no)
        shape = [1] * len(h_ls.shape)
        shape[_TERMINAL_AXIS] = len(branches)
        w = tf.reshape(tf.constant(weights), shape)
        h_hat = tf.cast(w, h_dft.dtype) * h_lmmse + tf.cast(1.0 - w, h_dft.dtype) * h_dft
        err_var = (
            tf.cast(w, tf.float32) * tf.cast(err_lmmse, tf.float32)
            + tf.cast(1.0 - w, tf.float32) * tf.cast(err_dft, tf.float32)
        )
        return h_hat, err_var, branches

    def call(self, y: tf.Tensor, no: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        h_ls, err_var_ls = self._ls_estimator(y, no)
        h_hat, err_var, _ = self.estimate_from_ls(h_ls, err_var_ls, no)
        return h_hat, err_var
