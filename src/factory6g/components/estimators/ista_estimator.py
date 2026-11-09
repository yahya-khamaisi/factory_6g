from __future__ import annotations

from typing import Tuple

import torch
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid


class ISTAChannelEstimator(Block):
    """Nonlinear sparse channel estimator using ISTA (Iterative Shrinkage-Thresholding).

    Instead of hard tap truncation (DFT estimator), solves a sparse recovery
    problem in the delay domain via iterative soft-thresholding:

        min_h  ||h||_1   s.t.   IFFT(h_ls) ≈ h

    The soft-threshold operator  S_λ(x) = sign(x) * max(|x| - λ, 0)  is
    genuinely nonlinear, giving better denoising for sparse multipath channels
    compared to the linear DFT truncation approach.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        config: dict | None = None,
        num_iterations: int = 10,
        lambda_scale: float = 1.0,
        step_size: float = 0.5,
    ) -> None:
        super().__init__()
        self._rg = resource_grid
        self._ls_estimator = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self.fft_size = int(resource_grid.fft_size)
        self.num_iterations = int(max(num_iterations, 1))
        self.lambda_scale = float(lambda_scale)
        self.step_size = float(step_size)

    @staticmethod
    def _soft_threshold(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """Complex soft-thresholding: shrink magnitude, preserve phase."""
        magnitude = torch.abs(x)
        shrunk = torch.clamp(magnitude - threshold, min=0.0)
        phase = x / torch.clamp(magnitude, min=1e-10).to(x.dtype)
        return phase * shrunk.to(x.dtype)

    def _ista_refine(self, h_ls: torch.Tensor, no: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply ISTA in the delay domain; return the estimate and the active-tap fraction."""
        no_scalar = torch.mean(torch.as_tensor(no, dtype=torch.float32))
        threshold = self.lambda_scale * torch.sqrt(no_scalar)
        step = self.step_size
        h = torch.fft.ifft(h_ls, dim=-1)
        for _ in range(self.num_iterations):
            residual = torch.fft.ifft(h_ls - torch.fft.fft(h, dim=-1), dim=-1)
            h = self._soft_threshold(h + step * residual, threshold * step)
        active_fraction = torch.mean((torch.abs(h) > threshold * step).to(torch.float32))
        active_fraction = torch.clamp(active_fraction, 1.0 / self.fft_size, 1.0)
        return torch.fft.fft(h, dim=-1), active_fraction

    def call(self, y: torch.Tensor, no: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_ls, err_var_ls = self._ls_estimator(y, no)
        h_hat, active_fraction = self._ista_refine(h_ls, no)
        return h_hat, err_var_ls * active_fraction.to(err_var_ls.dtype)
