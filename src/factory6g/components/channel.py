from __future__ import annotations

import tensorflow as tf
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
        # TR 38.901: dropping the topology and generating the channel eagerly
        # retains ~0.3 MB of native memory per call, which exhausts memory in
        # long campaigns. Both steps run together in one traced graph instead.
        self._tr38901_graph = None
        if channel_model_type not in _NO_TOPOLOGY:
            self._tr38901_graph = tf.function(self._drop_and_generate, reduce_retracing=True)

    def _load_rt_bank(self) -> tf.Tensor:
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
        return tf.constant(h, dtype=tf.complex64)

    @property
    def last_large_scale(self) -> dict | None:
        """Positions, LOS flags and gains (dB, re hall average) of the latest batch."""
        return None if self._large_scale is None else self._large_scale.last

    def _sample_raytraced_hall(self, batch_size: int) -> tf.Tensor:
        # One traced position per terminal for the whole batch; static terminals,
        # so the response is constant over every slot and symbol of the batch.
        idx = self._large_scale.draw_indices()
        scale = self._large_scale.raytraced_scale(idx)
        h = tf.gather(self._rt_bank, idx) * tf.cast(scale, tf.complex64)   # [U, M, F]
        h = tf.transpose(h, [1, 0, 2])                                       # [M, U, F]
        h = h[tf.newaxis, tf.newaxis, :, :, tf.newaxis, tf.newaxis, :]       # [1, 1, M, U, 1, 1, F]
        return tf.tile(h, [batch_size, 1, 1, 1, 1, self.resource_grid.num_ofdm_symbols, 1])

    def _apply_hall_gain(self, h: tf.Tensor) -> tf.Tensor:
        # Put the small-scale model on unit mean energy with a fixed constant
        # (per-slot and per-terminal fluctuations are kept), then apply each
        # terminal's persistent large-scale gain.
        cmt = self.config.get("channel_model_type", "tr38901")
        h = h / tf.cast(SMALL_SCALE_MEAN_ENERGY[cmt] ** 0.5, tf.complex64)
        gain = self._large_scale.statistical_gain(self._large_scale.draw_indices())   # [U]
        amp = tf.cast(tf.sqrt(gain), tf.complex64)
        return h * tf.reshape(amp, [1, 1, 1, -1, 1, 1, 1])

    def _sample_raytraced(self, batch_size: int) -> tf.Tensor:
        num_ut = int(self.config.get("num_ut", 4))
        idx = tf.random.uniform([batch_size, num_ut], maxval=tf.shape(self._rt_bank)[0], dtype=tf.int32)
        h = tf.gather(self._rt_bank, idx)                      # [B, U, M, F]
        # Same per-link normalisation as cir_to_ofdm_channel(normalize=True):
        # unit mean energy over receive antennas and subcarriers.
        energy = tf.reduce_mean(tf.square(tf.abs(h)), axis=(2, 3), keepdims=True)
        h = h / tf.cast(tf.sqrt(energy), tf.complex64)
        h = tf.transpose(h, [0, 2, 1, 3])                       # [B, M, U, F]
        h = h[:, tf.newaxis, :, :, tf.newaxis, tf.newaxis, :]   # [B, 1, M, U, 1, 1, F]
        # Static terminals: the response is constant over the slot.
        return tf.tile(h, [1, 1, 1, 1, 1, self.resource_grid.num_ofdm_symbols, 1])

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
            return UMi(**channel_params)
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

    def _drop_and_generate(self, batch_size: int) -> tf.Tensor:
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

    def sample_frequency_response(self, batch_size: int) -> tf.Tensor:
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
            return tf.ones(shape, dtype=tf.complex64)
        if cmt == "raytraced":
            with tf.device("/CPU:0"):
                if self._large_scale is not None:
                    return self._sample_raytraced_hall(batch_size)
                return self._sample_raytraced(batch_size)
        with tf.device("/CPU:0"):
            if self._tr38901_graph is not None:
                h = self._tr38901_graph(batch_size)
            else:
                h = self._generator(batch_size)
        if cmt == "rician":
            k = tf.cast(self._rician_k_factor, tf.float32)
            los = tf.cast(tf.sqrt(k / (k + 1.0)), tf.complex64)
            nlos_scale = tf.cast(tf.sqrt(1.0 / (k + 1.0)), tf.complex64)
            h = los + nlos_scale * h
        if self._large_scale is not None:
            h = self._apply_hall_gain(h)
        return h

    def apply_frequency_response(self, x_rg: tf.Tensor, h_freq: tf.Tensor) -> tf.Tensor:
        with tf.device("/CPU:0"):
            return self._applier(x_rg, h_freq, no=None)

    def sample_noise(self, y_shape: tf.TensorShape | tf.Tensor, noise_var: tf.Tensor) -> tf.Tensor:
        if isinstance(y_shape, tf.TensorShape):
            if None in y_shape:
                raise ValueError("Cannot sample noise from a partially defined TensorShape.")
            shape = y_shape.as_list()
        else:
            shape = y_shape

        with tf.device("/CPU:0"):
            noise_std = tf.sqrt(tf.cast(noise_var, tf.float32) / 2.0)
            noise_re = tf.random.normal(shape, stddev=noise_std, dtype=tf.float32)
            noise_im = tf.random.normal(shape, stddev=noise_std, dtype=tf.float32)
        return tf.complex(noise_re, noise_im)

    def received_shape_from_response(self, h_freq: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(h_freq)
        return tf.stack([shape[0], shape[1], shape[2], shape[5], shape[6]])

    def get_channel_model(self):
        return self._channel_model
