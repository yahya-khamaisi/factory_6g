from __future__ import annotations

import time

import numpy as np
import tensorflow as tf
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid
from sionna.phy.utils import ebnodb2no

from factory6g.sim.types import BatchContext, ResourceManagerFeedback

from ..components.antenna import AntennaConfig
from ..components.channel import ChannelModel
from ..components.receiver import Receiver
from ..components.transmitter import Transmitter
from ..components.estimators import PSOChannelEstimator
from .resource_manager import ResourceDirectives


class Model:
    """End-to-end PHY model with explicit batch contexts for fair Monte Carlo reuse."""

    def __init__(
        self,
        config: dict[str, object],
        perfect_csi: bool = False,
        estimator_type: str = "ls",
        estimator_kwargs: dict | None = None,
    ) -> None:
        self.config = config.copy()
        self.perfect_csi = perfect_csi
        self.estimator_type = estimator_type
        self.estimator_kwargs = estimator_kwargs or {}

        num_ofdm_symbols = int(self.config.get("num_ofdm_symbols", 14))
        fft_size = int(self.config.get("fft_size", 512))
        subcarrier_spacing = float(self.config.get("subcarrier_spacing", 30e3))
        num_tx = int(self.config.get("num_ut", 8))
        num_streams_per_tx = int(self.config.get("num_ut_ant", 1))
        cyclic_prefix_length = int(self.config.get("cyclic_prefix_length", 20))
        pilot_ofdm_symbol_indices = self.config.get("pilot_ofdm_symbol_indices", [2, 11])

        rx_tx_association = np.zeros([1, num_tx], dtype=np.int32)
        rx_tx_association[0, :] = 1

        self._rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=cyclic_prefix_length,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
        )

        self._sm = StreamManagement(rx_tx_association, num_streams_per_tx)

        self._antenna_config = AntennaConfig(self.config)
        self._transmitter = Transmitter(self.config, self._rg)
        self._channel = ChannelModel(self.config, self._antenna_config, self._rg)
        self._receiver = Receiver(
            self.config,
            self._rg,
            self._sm,
            self._transmitter._encoder,
            perfect_csi=perfect_csi,
            channel_estimator=self._build_channel_estimator(),
        )

    channel_hook = None  # optional callable applied to each prepared channel response

    @property
    def last_large_scale(self) -> dict | None:
        """Terminal positions, LOS flags and gains of the latest prepared batch."""
        return self._channel.last_large_scale

    def _build_channel_estimator(self):
        if self.perfect_csi:
            return None
        estimator_type = self.estimator_type.lower()
        if estimator_type in {"ls", "ls_nn", "ls-nn"}:
            return LSChannelEstimator(self._rg, interpolation_type="nn")
        if estimator_type in {"ls_lin", "ls-lin", "ls_linear"}:
            return LSChannelEstimator(self._rg, interpolation_type="lin")
        if estimator_type in {"pso", "dso"}:
            return PSOChannelEstimator(self.config, self._rg, **self.estimator_kwargs)
        if estimator_type in {"dft", "dft-based"}:
            from ..components.estimators import DFTChannelEstimator

            return DFTChannelEstimator(self._rg, config=self.config, **self.estimator_kwargs)
        if estimator_type in {"lmmse", "approx_lmmse"}:
            from ..components.estimators import LMMSEChannelEstimator

            return LMMSEChannelEstimator(self._rg, config=self.config, **self.estimator_kwargs)
        if estimator_type in {"adaptive", "adaptive_hybrid"}:
            from ..components.estimators import AdaptiveHybridChannelEstimator

            return AdaptiveHybridChannelEstimator(self._rg, config=self.config, **self.estimator_kwargs)
        if estimator_type in {"ista", "ista_sparse"}:
            from ..components.estimators import ISTAChannelEstimator

            return ISTAChannelEstimator(self._rg, config=self.config, **self.estimator_kwargs)
        if estimator_type in {"neural", "neural_net", "nn"}:
            from ..components.estimators import NeuralChannelEstimator

            return NeuralChannelEstimator(self._rg, config=self.config, **self.estimator_kwargs)
        if estimator_type == "perfect":
            return None
        raise ValueError(f"Unsupported estimator_type '{self.estimator_type}'.")

    def default_directives(self) -> ResourceDirectives:
        num_ut = int(self.config.get("num_ut", 8))
        return ResourceDirectives(
            active_ut_mask=[1] * num_ut,
            per_ut_power=[1.0] * num_ut,
            pilot_reuse_factor=1,
        )

    def prepare_batch_context(
        self,
        batch_size: int,
        ebno_db: float,
        include_feedback: bool,
    ) -> BatchContext:
        self._channel.set_topology(batch_size)
        h_freq = self._channel.sample_frequency_response(batch_size)
        if self.channel_hook is not None:
            # Scenario layer (factory6g.sim.industry): blockage, port faults, interference.
            h_freq = self.channel_hook(h_freq)

        noise_variance = tf.cast(
            ebnodb2no(
                ebno_db,
                self.config.get("num_bits_per_symbol", 2),
                self.config.get("coderate", 0.5),
                self._rg,
            ),
            tf.float32,
        )
        y_shape = self._channel.received_shape_from_response(h_freq)
        probe_noise = self._channel.sample_noise(y_shape, noise_variance)
        data_noise = self._channel.sample_noise(y_shape, noise_variance)
        source_bits = self._transmitter.sample_information_bits(batch_size)

        feedback = None
        if include_feedback:
            probe_directives = self.default_directives()
            x_probe, _, _ = self._transmitter.call(batch_size, directives=probe_directives)
            y_probe = self._channel.apply_frequency_response(x_probe, h_freq) + probe_noise
            if self.perfect_csi:
                feedback = ResourceManagerFeedback(
                    h_hat=h_freq,
                    err_var=tf.zeros(tf.shape(h_freq), dtype=noise_variance.dtype),
                )
            else:
                h_hat, err_var = self._receiver.estimate_channel(y_probe, noise_variance)
                feedback = ResourceManagerFeedback(h_hat=h_hat, err_var=err_var)

        return BatchContext(
            batch_size=batch_size,
            ebno_db=float(ebno_db),
            noise_variance=noise_variance,
            h_freq=h_freq,
            probe_noise=probe_noise,
            data_noise=data_noise,
            source_bits=source_bits,
            feedback=feedback,
        )

    def estimate_feedback(self, batch_context: BatchContext) -> ResourceManagerFeedback:
        """Scheduler feedback from this model's own estimator on the context's probe.

        The probe carries the context's source bits and probe noise, so every
        estimator asked for feedback on one context observes the same probe
        (P1 holds across estimators, not only across schedulers).
        """
        probe_directives = self.default_directives()
        x_probe, _, _ = self._transmitter.call(
            batch_context.batch_size,
            directives=probe_directives,
            bits=batch_context.source_bits,
        )
        y_probe = (
            self._channel.apply_frequency_response(x_probe, batch_context.h_freq)
            + batch_context.probe_noise
        )
        if self.perfect_csi:
            return ResourceManagerFeedback(
                h_hat=batch_context.h_freq,
                err_var=tf.zeros(tf.shape(batch_context.h_freq), dtype=batch_context.noise_variance.dtype),
            )
        h_hat, err_var = self._receiver.estimate_channel(y_probe, batch_context.noise_variance)
        return ResourceManagerFeedback(h_hat=h_hat, err_var=err_var)

    def run_batch(
        self,
        batch_context: BatchContext,
        directives: ResourceDirectives | None = None,
        include_details: bool = True,
    ) -> dict:
        active_directives = directives or self.default_directives()

        x_rg, bits, qam_symbols = self._transmitter.call(
            batch_context.batch_size,
            directives=active_directives,
            bits=batch_context.source_bits,
        )
        y = self._channel.apply_frequency_response(x_rg, batch_context.h_freq) + batch_context.data_noise

        if self.perfect_csi:
            h_hat = batch_context.h_freq
            err_var = 0.0
        else:
            h_hat, err_var = self._receiver.estimate_channel(y, batch_context.noise_variance)

        if include_details:
            start = time.perf_counter()
            x_hat, no_eff = self._receiver.equalize(y, h_hat, err_var, batch_context.noise_variance)
            llr = self._receiver.demap(x_hat, no_eff)
            bits_hat, decoder_iter = self._receiver.decode(llr)
            processing_latency_sec = time.perf_counter() - start
            air_interface_latency_sec = self._estimate_air_interface_latency()
            runtime_latency_sec = processing_latency_sec + air_interface_latency_sec
            total_energy_joules = self._estimate_energy(runtime_latency_sec, air_interface_latency_sec, decoder_iter)
            noise_power = self._noise_power_value(batch_context.noise_variance)
            tx_power_w = self._allocated_tx_power_w(active_directives)
            latency_model_sec = self._modelled_latency_sec(air_interface_latency_sec)
            return {
                "bits": bits.numpy(),
                "bits_hat": bits_hat.numpy(),
                "decoder_iterations": decoder_iter.numpy(),
                "channel": batch_context.h_freq.numpy(),
                "channel_hat": h_hat.numpy() if hasattr(h_hat, "numpy") else h_hat,
                # Mean error variance the estimator reports to the equaliser,
                # logged so it can be checked against the measured MSE.
                "err_var_mean": float(np.mean(err_var.numpy() if hasattr(err_var, "numpy") else err_var)),
                "qam": qam_symbols.numpy(),
                "qam_hat": x_hat.numpy(),
                "no_eff": no_eff.numpy(),
                "noise_power": noise_power,
                "latency_sec": air_interface_latency_sec,
                "processing_latency_sec": processing_latency_sec,
                "runtime_latency_sec": runtime_latency_sec,
                "energy_joules": total_energy_joules,
                "latency_model_sec": latency_model_sec,
                "tx_power_w": tx_power_w,
                "tx_energy_joules": tx_power_w * air_interface_latency_sec * batch_context.batch_size,
            }

        if self.perfect_csi:
            bits_hat = self._receiver.process_with_perfect_csi(y, batch_context.h_freq, batch_context.noise_variance)
        else:
            bits_hat = self._receiver(y, h_hat, err_var, batch_context.noise_variance)
        return {
            "bits": bits.numpy(),
            "bits_hat": bits_hat.numpy(),
        }

    def _estimate_air_interface_latency(self) -> float:
        subcarrier_spacing = float(self.config.get("subcarrier_spacing", 30e3))
        fft_size = float(self.config.get("fft_size", 512))
        cyclic_prefix_length = float(self.config.get("cyclic_prefix_length", 20))
        num_ofdm_symbols = float(self.config.get("num_ofdm_symbols", 14))
        symbol_duration = 1.0 / subcarrier_spacing
        cyclic_prefix_ratio = cyclic_prefix_length / max(fft_size, 1.0)
        return symbol_duration * (1.0 + cyclic_prefix_ratio) * num_ofdm_symbols

    def _modelled_latency_sec(self, air_interface_latency_sec: float) -> float:
        """One slot on the air plus a fixed receiver processing budget, in slots."""
        budget_slots = float(self.config.get("processing_budget_slots", 1.0))
        return air_interface_latency_sec * (1.0 + budget_slots)

    def _allocated_tx_power_w(self, directives: ResourceDirectives) -> float:
        """Total uplink transmit power implied by the resource directives.

        Each terminal transmits at most the configured maximum output power; the
        directives scale it by the power weight and switch it off when masked.
        """
        num_ut = int(self.config.get("num_ut", 8))
        p_max_w = 10.0 ** ((float(self.config.get("ut_max_tx_power_dbm", 23.0)) - 30.0) / 10.0)
        mask = np.ones(num_ut) if directives.active_ut_mask is None else np.asarray(directives.active_ut_mask, dtype=float)[:num_ut]
        power = np.ones(num_ut) if directives.per_ut_power is None else np.asarray(directives.per_ut_power, dtype=float)[:num_ut]
        return float(np.sum((mask > 0) * np.clip(power, 0.0, None)) * p_max_w)

    def _estimate_energy(
        self,
        runtime_latency_sec: float,
        air_interface_latency_sec: float,
        decoder_iter: tf.Tensor,
    ) -> float:
        num_info_bits = self._transmitter.num_info_bits
        safe_latency = max(runtime_latency_sec, 1e-12)
        encoding_power_watts = 10e-3 * (num_info_bits / safe_latency) / 1e6
        encoding_energy = encoding_power_watts * safe_latency * 0.1
        tx_energy = 0.2 * air_interface_latency_sec
        rx_energy = 0.1 * air_interface_latency_sec
        avg_iterations = float(tf.reduce_mean(decoder_iter))
        decoding_power_watts = 50e-3 * (num_info_bits / safe_latency) / 1e6 * (1.0 + avg_iterations / 10.0)
        decoding_energy = decoding_power_watts * safe_latency * 0.3
        return encoding_energy + tx_energy + rx_energy + decoding_energy

    @staticmethod
    def _noise_power_value(noise_variance: tf.Tensor) -> float:
        if hasattr(noise_variance, "numpy"):
            value = noise_variance.numpy()
        else:
            value = noise_variance
        if np.isscalar(value):
            return float(value)
        arr = np.array(value)
        return float(arr.item()) if arr.size == 1 else float(np.mean(arr))

    def get_config(self) -> dict:
        return self.config.copy()

    def get_transmitter(self) -> Transmitter:
        return self._transmitter

    def get_channel(self) -> ChannelModel:
        return self._channel

    def get_receiver(self) -> Receiver:
        return self._receiver
