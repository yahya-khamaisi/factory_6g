from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResourceManagerFeedback:
    h_hat: torch.Tensor
    err_var: torch.Tensor


@dataclass(frozen=True)
class BatchContext:
    batch_size: int
    ebno_db: float
    noise_variance: torch.Tensor
    h_freq: torch.Tensor
    probe_noise: torch.Tensor
    data_noise: torch.Tensor
    source_bits: torch.Tensor
    feedback: ResourceManagerFeedback | None
