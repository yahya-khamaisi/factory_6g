from __future__ import annotations

from typing import Tuple

import torch
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid


class DFTChannelEstimator(Block):
    """DFT-denoising estimator using delay-domain tap truncation."""

    def __init__(
        self,
        resource_grid: ResourceGrid,
        config: dict | None = None,
        dft_tap_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self._rg = resource_grid
        self._ls_estimator = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self.fft_size = int(resource_grid.fft_size)
        self.cp_length = int(resource_grid.cyclic_prefix_length)
        self.dft_tap_ratio = float(dft_tap_ratio)
        target_taps = int(round(self.cp_length * max(self.dft_tap_ratio, 0.0)))
        self.tap_count = int(min(max(target_taps, 1), self.fft_size))

    def estimate_from_ls(self, h_ls: torch.Tensor, err_var_ls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_delay = torch.fft.ifft(h_ls, dim=-1)
        mask = (torch.arange(self.fft_size, device=h_delay.device) < self.tap_count).to(h_delay.dtype)
        h_dft = torch.fft.fft(h_delay * mask, dim=-1)
        err_var_dft = err_var_ls * (float(self.tap_count) / float(self.fft_size))
        return h_dft, err_var_dft

    def call(self, y: torch.Tensor, no: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_ls, err_var_ls = self._ls_estimator(y, no)
        return self.estimate_from_ls(h_ls, err_var_ls)
