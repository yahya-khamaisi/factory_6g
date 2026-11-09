from __future__ import annotations

import torch
from sionna.phy.channel import (
    ApplyOFDMChannel,
    GenerateOFDMChannel,
    RayleighBlockFading,
    gen_single_sector_topology as gen_topology,
)
from sionna.phy.channel.tr38901 import RMa, UMa, UMi
from sionna.phy.ofdm import ResourceGrid

from .antenna import AntennaConfig
from .large_scale import HallLargeScale
from .raytracing import load_bank_split
from factory6g.sim.rng import rng

# Mean small-scale link energy of each model without normalisation, used to put
# every model on unit mean energy with one fixed constant (system model v2).
# Rayleigh and Rician are unit-mean analytically; the TR 38.901 UMi value was
# measured over 2 x 1920 slots x 4 links at two seeds (1.828 and 1.837) with
# path loss and shadowing off and the study's antenna arrays.
SMALL_SCALE_MEAN_ENERGY = {"rayleigh": 1.0, "rician": 1.0, "tr38901": 1.83}

# Channel models that need no TR 38.901 topology drop.
_NO_TOPOLOGY = ("rayleigh", "rician", "raytraced", "awgn")


class ChannelModel:
    """Channel wrapper that separates topology, channel sampling, and channel application."""

    def __init__(self, config: dict, antenna_config: AntennaConfig, resource_grid: ResourceGrid):
        self.config = config
        self.antenna_config = antenna_config
        self.resource_grid = resource_grid

        channel_model_type = self.config.get("channel_model_type", "tr38901")
        # System model v2 ("hall"): each batch every terminal keeps one hall
        # position, whose large-scale gain persists over the batch, and links are
        # not power-normalised per slot (thesis/notes/SYSTEM_MODEL_V2.md).
        # "none" keeps the v1 per-link, per-slot normalisation.
        self.large_scale_model = str(self.config.get("large_scale_model", "hall")).lower()
        if self.large_scale_model not in ("hall", "none"):
            raise ValueError(f"Unknown large_scale_model '{self.large_scale_model}'.")
        per_slot_norm = self.large_scale_model == "none"
        self._large_scale = None
        if self.large_scale_model == "hall" and channel_model_type != "awgn":
            self._large_scale = HallLargeScale(
                self.config.get("rt_channel_bank", "data/rt_channel_bank.npz"),
                self.config.get("rt_bank_split", "eval"),
                num_ut=int(self.config.get("num_ut", 4)),
                carrier_frequency_hz=float(self.config.get("carrier_frequency", 3.5e9)),
            )
        if channel_model_type == "rayleigh":
            self._channel_model = self._create_rayleigh_channel()
            self._generator = GenerateOFDMChannel(self._channel_model, resource_grid, normalize_channel=per_slot_norm)
        elif channel_model_type == "rician":
            # Rician = deterministic LoS + Rayleigh NLoS, blended by K-factor
            self._channel_model = self._create_rayleigh_channel()
            self._generator = GenerateOFDMChannel(self._channel_model, resource_grid, normalize_channel=per_slot_norm)
            self._rician_k_factor = float(self.config.get("rician_k_factor", 1.0))
        elif channel_model_type == "raytraced":
            # Site-specific: uplink CFRs traced offline in the factory hall
            # (factory6g.components.raytracing); each slot draws one bank
            # position per terminal.
            self._channel_model = None
            self._generator = None
            self._rt_bank = self._load_rt_bank()
        elif channel_model_type == "awgn":
            self._channel_model = None
            self._generator = None
        else:  # tr38901
            self._channel_model = self._create_tr38901_channel()
            self._generator = GenerateOFDMChannel(self._channel_model, resource_grid, normalize_channel=per_slot_norm)
        self._applier = ApplyOFDMChannel()
        # The TensorFlow stack traced topology drop and channel generation in one
        # graph to stop a native memory leak; PyTorch runs both eagerly.
        self._tr38901_graph = None
        if channel_model_type not in _NO_TOPOLOGY:
            self._tr38901_graph = self._drop_and_generate

    def _load_rt_bank(self) -> torch.Tensor:
        path = self.config.get("rt_channel_bank", "data/rt_channel_bank.npz")
        split = self.config.get("rt_bank_split", "eval")
        h = load_bank_split(path, split)  # [N, num_bs_ant, F]
        num_bs_ant = int(self.config.get("num_bs_ant", 8))
        if h.shape[1] != num_bs_ant or h.shape[2] != self.resource_grid.fft_size:
            raise ValueError(
                f"Channel bank {path} has shape {h.shape[1:]}, expected "
                f"({num_bs_ant}, {self.resource_grid.fft_size})."
            )
        if self.config.get("direction", "uplink") != "uplink" or int(self.config.get("num_ut_ant", 1)) != 1:
            raise ValueError("The ray-traced channel bank supports uplink with single-antenna terminals only.")
        return torch.as_tensor(h).to(torch.complex64)

    @property
    def last_large_scale(self) -> dict | None:
        """Positions, LOS flags and gains (dB, re hall average) of the latest batch."""
        return None if self._large_scale is None else self._large_scale.last

    def _sample_raytraced_hall(self, batch_size: int) -> torch.Tensor:
        # One traced position per terminal for the whole batch; static terminals,
        # so the response is constant over every slot and symbol of the batch.
        idx = self._large_scale.draw_indices()
        scale = self._large_scale.raytraced_scale(idx)
        h = self._rt_bank[idx] * scale                                       # [U, M, F]
        h = h.permute(1, 0, 2)                                               # [M, U, F]
        h = h[None, None, :, :, None, None, :]                               # [1, 1, M, U, 1, 1, F]
        return h.repeat(batch_size, 1, 1, 1, 1, self.resource_grid.num_ofdm_symbols, 1)

    def _apply_hall_gain(self, h: torch.Tensor) -> torch.Tensor:
        # Put the small-scale model on unit mean energy with a fixed constant
        # (per-slot and per-terminal fluctuations are kept), then apply each
        # terminal's persistent large-scale gain.
        cmt = self.config.get("channel_model_type", "tr38901")
        h = h / SMALL_SCALE_MEAN_ENERGY[cmt] ** 0.5
        gain = self._large_scale.statistical_gain(self._large_scale.draw_indices())   # [U]
        amp = torch.sqrt(gain).to(h.device).to(torch.complex64)
        return h * amp.reshape(1, 1, 1, -1, 1, 1, 1)

    def _sample_raytraced(self, batch_size: int) -> torch.Tensor:
        num_ut = int(self.config.get("num_ut", 4))
        idx = torch.randint(0, self._rt_bank.shape[0], (batch_size, num_ut), generator=rng("cpu"))
        h = self._rt_bank[idx]                                  # [B, U, M, F]
        # Same per-link normalisation as cir_to_ofdm_channel(normalize=True):
        # unit mean energy over receive antennas and subcarriers.
        energy = torch.mean(torch.abs(h) ** 2, dim=(2, 3), keepdim=True)
        h = h / torch.sqrt(energy).to(torch.complex64)
        h = h.permute(0, 2, 1, 3)                               # [B, M, U, F]
        h = h[:, None, :, :, None, None, :]                     # [B, 1, M, U, 1, 1, F]
        # Static terminals: the response is constant over the slot.
        return h.repeat(1, 1, 1, 1, 1, self.resource_grid.num_ofdm_symbols, 1)

    def _create_tr38901_channel(self):
        channel_params = {
            "carrier_frequency": self.config.get("carrier_frequency", 3.5e9),
            "o2i_model": self.config.get("o2i_model", "low"),
            "ut_array": self.antenna_config.get_ut_array(),
            "bs_array": self.antenna_config.get_bs_array(),
            "direction": self.config.get("direction", "uplink"),
            "enable_pathloss": self.config.get("enable_pathloss", False),
            "enable_shadow_fading": self.config.get("enable_shadow_fading", False),
        }
        scenario_lower = self.config.get("scenario", "umi").lower()
        if scenario_lower == "umi":
            # Sionna 2.x defaults UMi to TR 38.901 v19.2; pin v16.1, the edition
            # Sionna 1.2.1 implemented, so results stay comparable.
            return UMi(**channel_params, spec_version=str(self.config.get("tr38901_spec_version", "16.1")))
        if scenario_lower == "uma":
            return UMa(**channel_params)
        if scenario_lower == "rma":
            return RMa(**channel_params)
        raise ValueError(f"Unknown scenario: {scenario_lower}. Supported: umi, uma, rma.")

    def _create_rayleigh_channel(self):
        if self.config.get("direction", "uplink") == "uplink":
            num_rx = 1
            num_rx_ant = int(self.config.get("num_bs_ant", 32))
            num_tx = int(self.config.get("num_ut", 8))
            num_tx_ant = int(self.config.get("num_ut_ant", 1))
        else:
            num_rx = int(self.config.get("num_ut", 8))
            num_rx_ant = int(self.config.get("num_ut_ant", 1))
            num_tx = 1
            num_tx_ant = int(self.config.get("num_bs_ant", 32))
        return RayleighBlockFading(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
        )

    def _drop_and_generate(self, batch_size: int) -> torch.Tensor:
        self._set_topology_eager(batch_size)
        return self._generator(batch_size)

    def set_topology(self, batch_size: int) -> None:
        # For TR 38.901 the topology is dropped inside the graph of
        # sample_frequency_response, together with the channel draw.
        return

    def _set_topology_eager(self, batch_size: int) -> None:
        if self.config.get("channel_model_type", "tr38901") in _NO_TOPOLOGY:
            return
        topology = gen_topology(
            batch_size,
            self.config.get("num_ut", 8),
            self.config.get("scenario", "umi"),
            min_ut_velocity=self.config.get("min_ut_velocity", 0.0),
            max_ut_velocity=self.config.get("max_ut_velocity", 0.0),
        )
        if hasattr(self._channel_model, "set_topology"):
            self._channel_model.set_topology(*topology)

    def sample_frequency_response(self, batch_size: int) -> torch.Tensor:
        cmt = self.config.get("channel_model_type", "tr38901")
        if cmt == "awgn":
            direction = self.config.get("direction", "uplink")
            if direction == "uplink":
                num_rx, num_rx_ant = 1, int(self.config.get("num_bs_ant", 8))
                num_tx, num_tx_ant = int(self.config.get("num_ut", 4)), int(self.config.get("num_ut_ant", 1))
            else:
                num_rx, num_rx_ant = int(self.config.get("num_ut", 4)), int(self.config.get("num_ut_ant", 1))
                num_tx, num_tx_ant = 1, int(self.config.get("num_bs_ant", 8))
            shape = [
                batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant,
                self.resource_grid.num_ofdm_symbols,
                self.resource_grid.fft_size,
            ]
            return torch.ones(shape, dtype=torch.complex64)
        if cmt == "raytraced":
            if self._large_scale is not None:
                return self._sample_raytraced_hall(batch_size)
            return self._sample_raytraced(batch_size)
        if self._tr38901_graph is not None:
            h = self._tr38901_graph(batch_size)
        else:
            h = self._generator(batch_size)
        if cmt == "rician":
            k = float(self._rician_k_factor)
            h = (k / (k + 1.0)) ** 0.5 + (1.0 / (k + 1.0)) ** 0.5 * h
        if self._large_scale is not None:
            h = self._apply_hall_gain(h)
        return h

    def apply_frequency_response(self, x_rg: torch.Tensor, h_freq: torch.Tensor) -> torch.Tensor:
        return self._applier(x_rg, h_freq, no=None)

    def sample_noise(self, y_shape, noise_var) -> torch.Tensor:
        shape = [int(v) for v in y_shape]
        noise_std = torch.sqrt(torch.as_tensor(noise_var, dtype=torch.float32) / 2.0)
        g = rng("cpu")
        noise_re = torch.randn(shape, generator=g) * noise_std
        noise_im = torch.randn(shape, generator=g) * noise_std
        return torch.complex(noise_re, noise_im)

    def received_shape_from_response(self, h_freq: torch.Tensor) -> list[int]:
        shape = h_freq.shape
        return [shape[0], shape[1], shape[2], shape[5], shape[6]]

    def get_channel_model(self):
        return self._channel_model
