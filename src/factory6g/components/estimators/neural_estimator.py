from __future__ import annotations

import json
import logging
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
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


def build_neural_estimator_model(num_ofdm_symbols: int, fft_size: int) -> tf.keras.Model:
    """Build the SNR-conditioned residual Conv2D network used for channel estimation.

    Input shape:  [N, num_ofdm_symbols, fft_size, 3]
                  channels: re(h_ls), im(h_ls), snr_map (normalised Eb/No)
    Output shape: [N, num_ofdm_symbols, fft_size, 2]
                  channels: re(delta), im(delta)  — the correction to add to h_ls

    At inference: h_hat = h_ls + complex(re(delta), im(delta))
    """
    inputs = tf.keras.layers.Input(
        shape=(num_ofdm_symbols, fft_size, 3), name="h_ls_snr"
    )
    x = tf.keras.layers.Conv2D(64, (9, 9), padding="same", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu", name="conv2")(x)
    outputs = tf.keras.layers.Conv2D(2, (5, 5), padding="same", activation="linear", name="conv3")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="neural_channel_estimator")


class NeuralChannelEstimator(Block):
    """Nonlinear channel estimator using a pre-trained SNR-conditioned Conv2D network.

    The network takes the LS channel estimate together with a normalised SNR
    map and predicts the *residual correction* (h_true - h_ls).  The final
    estimate is h_hat = h_ls + delta, which is easier to learn than direct
    channel regression because the corrections are small at high SNR.

    Training:
        Use ``docker compose run train-estimator`` (or ``python train.py``) to
        generate training data and train the network.  Weights are saved to the
        path specified by ``model_path`` (default: ``models/neural_estimator.keras``).

    If no weights file is found at ``model_path``, falls back to LS estimation
    and emits a warning.
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
        # Offset between -10*log10(no) and Eb/No in dB. The noise variance comes
        # from ebnodb2no, which divides by bits per symbol, code rate and the
        # data share of the grid energy, so the offset must be removed to feed
        # the network the same Eb/No it was trained on.
        cfg = config or {}
        self._no_to_ebno_offset_db = float(
            -10.0
            * np.log10(
                float(
                    ebnodb2no(
                        0.0,
                        int(cfg.get("num_bits_per_symbol", 2)),
                        float(cfg.get("coderate", 0.5)),
                        resource_grid,
                    )
                )
            )
        )
        self._nn_available = False
        self._nn_model: tf.keras.Model | None = None
        self.conditioning = load_conditioning(model_path)
        self._cond_min_db, self._cond_max_db = self.conditioning["range_db"]

        try:
            self._nn_model = tf.keras.models.load_model(model_path, compile=False)
            self._nn_available = True
            logger.info("NeuralChannelEstimator: loaded weights from '%s'.", model_path)
        except (OSError, ValueError) as exc:
            logger.warning(
                "NeuralChannelEstimator: could not load model from '%s' (%s). "
                "Falling back to plain LS. Run 'python train.py' inside Docker to train.",
                model_path,
                exc,
            )

    def effective_ebno_db(self, h_ls: tf.Tensor, err_var_ls: tf.Tensor, ebno_db: tf.Tensor) -> tf.Tensor:
        """TensorFlow twin of ``effective_ebno_db_np``: one value per antenna-terminal grid."""
        energy = tf.reduce_mean(tf.abs(h_ls) ** 2, axis=_POOL_AXES, keepdims=True)
        err = tf.reduce_mean(tf.cast(err_var_ls, energy.dtype))
        power = tf.maximum(energy - err, _MIN_LINK_POWER)
        cond = tf.cast(ebno_db, energy.dtype) + 10.0 * tf.math.log(power) / tf.math.log(tf.constant(10.0, energy.dtype))
        return tf.broadcast_to(cond[..., 0, 0], tf.shape(h_ls)[:-2])

    def _apply_network(self, h_ls: tf.Tensor, cond_db: tf.Tensor) -> tf.Tensor:
        """Reshape h_ls + conditioning map, run the network, add residual back to h_ls.

        ``cond_db`` is a scalar (legacy batch Eb/No) or one value per grid,
        shaped like ``h_ls`` without its last two axes.
        """
        original_shape = tf.shape(h_ls)
        # Flatten all leading dims into a batch dimension
        flat = tf.reshape(h_ls, [-1, self.num_ofdm_symbols, self.fft_size])
        # Split complex → real/imag: [N, ofdm_syms, fft_size, 2]
        ri = tf.stack([tf.math.real(flat), tf.math.imag(flat)], axis=-1)
        ri = tf.cast(ri, tf.float32)

        # Normalised conditioning map: per grid → [N, ofdm_syms, fft_size, 1]
        n_flat = tf.shape(flat)[0]
        cond = tf.cast(cond_db, tf.float32)
        cond = tf.broadcast_to(tf.reshape(cond, [-1]), [n_flat]) if cond.shape.rank == 0 else tf.reshape(cond, [-1])
        cond_norm = tf.clip_by_value(
            (cond - self._cond_min_db) / (self._cond_max_db - self._cond_min_db), 0.0, 1.0
        )
        snr_map = tf.ones([n_flat, self.num_ofdm_symbols, self.fft_size, 1]) * cond_norm[:, None, None, None]

        ri3 = tf.concat([ri, snr_map], axis=-1)  # [N, ofdm_syms, fft_size, 3]

        # Network predicts the correction (residual)
        delta_ri = self._nn_model(ri3, training=False)  # [N, ofdm_syms, fft_size, 2]

        delta = tf.complex(delta_ri[..., 0], delta_ri[..., 1])
        delta = tf.cast(tf.reshape(delta, tf.shape(flat)), h_ls.dtype)

        h_refined_flat = flat + delta
        return tf.reshape(h_refined_flat, original_shape)

    def ebno_db_from_noise(self, no: tf.Tensor) -> tf.Tensor:
        """Invert ebnodb2no: recover the Eb/No (dB) that produced noise variance ``no``."""
        no_mean = tf.reduce_mean(tf.cast(no, tf.float32))
        # Guard against log(0): clamp noise to a small positive value
        no_safe = tf.maximum(no_mean, 1e-10)
        no_db = -10.0 * tf.math.log(no_safe) / tf.math.log(10.0)
        return no_db - self._no_to_ebno_offset_db

    def call(self, y: tf.Tensor, no: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        h_ls, err_var_ls = self._ls_estimator(y, no)

        if not self._nn_available:
            return h_ls, err_var_ls

        ebno_db = self.ebno_db_from_noise(no)
        if self.conditioning["input"] == COND_LS_ENERGY:
            cond_db = self.effective_ebno_db(h_ls, err_var_ls, ebno_db)
        else:
            cond_db = ebno_db

        h_hat = self._apply_network(h_ls, cond_db)
        err_var = err_var_ls * tf.cast(self.nn_err_var_scale, err_var_ls.dtype)
        return h_hat, err_var
