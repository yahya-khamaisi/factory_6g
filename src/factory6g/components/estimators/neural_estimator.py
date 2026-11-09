from __future__ import annotations

import json
import logging
import os
from typing import Tuple

import numpy as np
import torch
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid
from sionna.phy.utils import ebnodb2no

logger = logging.getLogger(__name__)

# Conditioning input of the network. Legacy weights were trained on the batch
# Eb/No normalised over 0-20 dB; weights trained with per-terminal large-scale
# gains use the effective Eb/No of each terminal, estimated from the LS grid
# energy, over a wider range. The choice is read from the model's metadata file
# so that training and inference always use the same rule.
COND_BATCH_EBNO = "batch_ebno"
COND_LS_ENERGY = "ls_energy_effective_ebno"
LEGACY_CONDITIONING = {"input": COND_BATCH_EBNO, "range_db": [0.0, 20.0]}
# Covers the v2 effective-SNR spread: hall gains of about -23 to +11 dB re the
# hall average, plus per-slot fading, over a 0-30 dB Eb/No sweep (smoke gate,
# results/smoke_v2/smoke_estimators.json: p1 -28 dB at 0 dB, p99 40 dB at 30 dB).
LS_ENERGY_CONDITIONING = {"input": COND_LS_ENERGY, "range_db": [-30.0, 45.0]}
# Floor on the estimated link power (linear), i.e. -30 dB, so log10 stays finite
# when noise dominates the LS grid.
_MIN_LINK_POWER = 1e-3
# Axes of an LS estimate [batch, rx, rx_ant, tx, stream, symbols, subcarriers]
# pooled to get one power estimate per (slot, terminal): the slot's own link
# power, including its small-scale fade, over the antenna ports and the grid.
_POOL_AXES = (1, 2, 4, 5, 6)


def effective_ebno_db_np(h_ls: np.ndarray, err_var_mean: float, ebno_db: float) -> np.ndarray:
    """Per-slot, per-terminal effective Eb/No (dB) from an LS estimate, broadcast to every grid.

    The LS grid energy is the link power plus the LS error variance, so
    subtracting the error variance estimates the link power P_u. A unit-power
    link is the reference of ``ebno_db``, so the terminal sees
    ``ebno_db + 10 log10(P_u)``. Returns an array of shape ``h_ls.shape[:-2]``
    (one value per antenna-terminal grid). Must match the TensorFlow version.
    """
    energy = np.mean(np.abs(h_ls) ** 2, axis=_POOL_AXES, keepdims=True)
    power = np.maximum(energy - float(err_var_mean), _MIN_LINK_POWER)
    cond = float(ebno_db) + 10.0 * np.log10(power)
    return np.broadcast_to(cond[..., 0, 0], h_ls.shape[:-2]).astype(np.float32)


def load_conditioning(model_path: str) -> dict:
    """Conditioning rule recorded beside the weights; legacy rule if absent."""
    meta_path = model_path + ".metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            cond = json.load(fh).get("conditioning")
        if cond:
            return {"input": str(cond["input"]), "range_db": [float(v) for v in cond["range_db"]]}
    return dict(LEGACY_CONDITIONING)


class NeuralEstimatorNet(torch.nn.Module):
    """SNR-conditioned residual Conv2D network used for channel estimation.

    Input shape:  [N, num_ofdm_symbols, fft_size, 3]
                  channels: re(h_ls), im(h_ls), snr_map (normalised Eb/No)
    Output shape: [N, num_ofdm_symbols, fft_size, 2]
                  channels: re(delta), im(delta)  — the correction to add to h_ls

    Channels-last at the interface, as in the TensorFlow version, so datasets
    and weights carry over; the convolutions run channels-first inside.
    """

    def __init__(self, num_ofdm_symbols: int, fft_size: int) -> None:
        super().__init__()
        self.num_ofdm_symbols = int(num_ofdm_symbols)
        self.fft_size = int(fft_size)
        self.conv1 = torch.nn.Conv2d(3, 64, 9, padding=4)
        self.conv2 = torch.nn.Conv2d(64, 32, 5, padding=2)
        self.conv3 = torch.nn.Conv2d(32, 2, 5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.conv3(x).permute(0, 2, 3, 1)


def build_neural_estimator_model(num_ofdm_symbols: int, fft_size: int) -> NeuralEstimatorNet:
    return NeuralEstimatorNet(num_ofdm_symbols, fft_size)


def torch_weights_path(model_path: str) -> str:
    """``x.keras`` -> ``x.pt``; any other path is used as given."""
    return model_path[: -len(".keras")] + ".pt" if model_path.endswith(".keras") else model_path


def load_neural_estimator(model_path: str, num_ofdm_symbols: int, fft_size: int) -> NeuralEstimatorNet:
    """Load PyTorch weights (``.pt``), or convert the Keras archive of the TensorFlow stack."""
    net = NeuralEstimatorNet(num_ofdm_symbols, fft_size)
    pt = torch_weights_path(model_path)
    if os.path.exists(pt):
        net.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    elif model_path.endswith(".keras") and os.path.exists(model_path):
        from factory6g.torch_io import conv2d_from_keras, keras_layer_weights

        layers = keras_layer_weights(model_path)
        if len(layers) != 3:
            raise ValueError(f"Expected 3 conv layers in {model_path}, found {[n for n, _ in layers]}.")
        for conv, (_, (kernel, bias)) in zip((net.conv1, net.conv2, net.conv3), layers):
            w, b = conv2d_from_keras(kernel, bias)
            if tuple(w.shape) != tuple(conv.weight.shape):
                raise ValueError(f"Kernel shape {tuple(w.shape)} does not fit {tuple(conv.weight.shape)}.")
            conv.weight.data.copy_(w)
            conv.bias.data.copy_(b)
    else:
        raise OSError(f"No weights at '{pt}' or '{model_path}'.")
    return net.eval()


class NeuralChannelEstimator(Block):
    """Nonlinear channel estimator using a pre-trained SNR-conditioned Conv2D network.

    The network takes the LS channel estimate together with a normalised SNR
    map and predicts the *residual correction* (h_true - h_ls). The final
    estimate is h_hat = h_ls + delta.

    Weights are read from ``<model_path minus .keras>.pt`` when present, else
    converted from the Keras archive at ``model_path`` (TensorFlow stack). If
    neither exists, falls back to LS estimation and emits a warning.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        config: dict | None = None,
        model_path: str = "models/neural_estimator.keras",
        nn_err_var_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self._rg = resource_grid
        self._ls_estimator = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self.num_ofdm_symbols = int(resource_grid.num_ofdm_symbols)
        self.fft_size = int(resource_grid.fft_size)
        self.nn_err_var_scale = float(nn_err_var_scale)
        # Offset between -10*log10(no) and Eb/No in dB (ebnodb2no divides by bits
        # per symbol, code rate and the data share of the grid energy).
        cfg = config or {}
        self._no_to_ebno_offset_db = float(
            -10.0 * np.log10(float(ebnodb2no(0.0, int(cfg.get("num_bits_per_symbol", 2)), float(cfg.get("coderate", 0.5)), resource_grid)))
        )
        self._nn_available = False
        self._nn_model: NeuralEstimatorNet | None = None
        self.conditioning = load_conditioning(model_path)
        self._cond_min_db, self._cond_max_db = self.conditioning["range_db"]
        try:
            self._nn_model = load_neural_estimator(model_path, self.num_ofdm_symbols, self.fft_size)
            self._nn_available = True
            logger.info("NeuralChannelEstimator: loaded weights for '%s'.", model_path)
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("NeuralChannelEstimator: could not load model from '%s' (%s). Falling back to plain LS.", model_path, exc)

    @staticmethod
    def effective_ebno_db(h_ls: torch.Tensor, err_var_ls: torch.Tensor, ebno_db: torch.Tensor) -> torch.Tensor:
        """PyTorch twin of ``effective_ebno_db_np``: one value per antenna-terminal grid."""
        energy = torch.mean(torch.abs(h_ls) ** 2, dim=_POOL_AXES, keepdim=True)
        err = torch.mean(torch.as_tensor(err_var_ls).to(energy.dtype))
        power = torch.clamp(energy - err, min=_MIN_LINK_POWER)
        cond = torch.as_tensor(ebno_db, dtype=energy.dtype) + 10.0 * torch.log10(power)
        return cond[..., 0, 0].expand(h_ls.shape[:-2])

    def _apply_network(self, h_ls: torch.Tensor, cond_db: torch.Tensor) -> torch.Tensor:
        original_shape = h_ls.shape
        flat = h_ls.reshape(-1, self.num_ofdm_symbols, self.fft_size)
        ri = torch.stack([flat.real, flat.imag], dim=-1).to(torch.float32)
        n_flat = flat.shape[0]
        cond = torch.as_tensor(cond_db, dtype=torch.float32)
        cond = cond.reshape(-1).expand(n_flat) if cond.dim() == 0 else cond.reshape(-1)
        cond_norm = torch.clamp((cond - self._cond_min_db) / (self._cond_max_db - self._cond_min_db), 0.0, 1.0)
        snr_map = cond_norm[:, None, None, None].expand(n_flat, self.num_ofdm_symbols, self.fft_size, 1)
        with torch.no_grad():
            delta_ri = self._nn_model(torch.cat([ri, snr_map], dim=-1))
        delta = torch.complex(delta_ri[..., 0], delta_ri[..., 1]).to(h_ls.dtype)
        return (flat + delta).reshape(original_shape)

    def ebno_db_from_noise(self, no) -> torch.Tensor:
        """Invert ebnodb2no: recover the Eb/No (dB) that produced noise variance ``no``."""
        no_safe = torch.clamp(torch.mean(torch.as_tensor(no, dtype=torch.float32)), min=1e-10)
        return -10.0 * torch.log10(no_safe) - self._no_to_ebno_offset_db

    def call(self, y: torch.Tensor, no: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_ls, err_var_ls = self._ls_estimator(y, no)
        if not self._nn_available:
            return h_ls, err_var_ls
        ebno_db = self.ebno_db_from_noise(no)
        if self.conditioning["input"] == COND_LS_ENERGY:
            cond_db = self.effective_ebno_db(h_ls, err_var_ls, ebno_db)
        else:
            cond_db = ebno_db
        h_hat = self._apply_network(h_ls, cond_db)
        return h_hat, err_var_ls * self.nn_err_var_scale
