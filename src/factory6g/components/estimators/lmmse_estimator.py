from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid


class LMMSEChannelEstimator(Block):
    """
    Frequency-domain LMMSE smoother with spectral shrinkage acceleration.

    It computes LS once and applies:
      H_hat = H_ls * U * diag(lambda/(lambda+sigma_n)) * U^H
    where U/lambda come from one-time eigendecomposition of the Hermitian
    correlation matrix R_freq.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        config: dict | None = None,
        r_freq: float = 0.98,
        noise_bin_db: float = 0.5,
    ) -> None:
        super().__init__()
        self._rg = resource_grid
        self._ls_estimator = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self.fft_size = int(resource_grid.fft_size)
        self.r_freq = float(r_freq)
        self.noise_bin_db = float(noise_bin_db)

        k_indices = np.arange(self.fft_size)
        delta_k = np.abs(k_indices[:, None] - k_indices[None, :])
        correlation = (self.r_freq ** delta_k).astype(np.float32)
        self.R_freq = torch.as_tensor(correlation).to(torch.complex64)
        eigvals, eigvecs = np.linalg.eigh(correlation.astype(np.float64))
        eigvals = np.maximum(eigvals.astype(np.float32), 1e-9)
        self._eigvals = torch.as_tensor(eigvals, dtype=torch.float32)
        self._eigvecs = torch.as_tensor(eigvecs.astype(np.complex64))
        self._eigvecs_h = self._eigvecs.conj().T.contiguous()
        self._shrinkage_cache: dict[float, torch.Tensor] = {}

    @staticmethod
    def _avg_noise(no) -> float:
        return max(float(torch.mean(torch.as_tensor(no, dtype=torch.float32))), 1e-12)

    def _noise_bin_key(self, noise_linear: float) -> float | None:
        if self.noise_bin_db <= 0.0:
            return None
        noise_db = 10.0 * np.log10(max(noise_linear, 1e-12))
        return round(noise_db / self.noise_bin_db) * self.noise_bin_db

    def _shrinkage_for_noise(self, noise_linear: float) -> torch.Tensor:
        key = self._noise_bin_key(noise_linear)
        if key is None:
            noise_eff = noise_linear
        else:
            if key in self._shrinkage_cache:
                return self._shrinkage_cache[key]
            noise_eff = 10.0 ** (key / 10.0)
        shrinkage = (self._eigvals / (self._eigvals + noise_eff)).to(torch.complex64)
        if key is not None:
            self._shrinkage_cache[key] = shrinkage
        return shrinkage

    def estimate_from_ls(self, h_ls: torch.Tensor, err_var_ls: torch.Tensor, no) -> Tuple[torch.Tensor, torch.Tensor]:
        shrinkage = self._shrinkage_for_noise(self._avg_noise(no))
        flat = h_ls.to(torch.complex64).reshape(-1, self.fft_size)
        projected = (flat @ self._eigvecs) * shrinkage
        h_lmmse = (projected @ self._eigvecs_h).reshape(h_ls.shape)
        smoothing_factor = torch.mean(shrinkage).real
        return h_lmmse, err_var_ls * smoothing_factor.to(err_var_ls.dtype)

    def call(self, y: torch.Tensor, no: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_ls, err_var_ls = self._ls_estimator(y, no)
        return self.estimate_from_ls(h_ls, err_var_ls, no)
