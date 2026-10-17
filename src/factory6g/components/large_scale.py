"""Persistent large-scale terminal state in the factory hall (system model v2).

Each batch, every terminal draws one position in the hall from the ray-traced
channel bank and keeps it for every slot of the batch. The position fixes the
terminal's large-scale gain:

* for the small-scale models (Rayleigh, Rician, TR 38.901 UMi), 3GPP TR 38.901
  Table 7.4.1-1 InF-SL path loss with log-normal shadowing, LOS or NLOS according
  to the ray trace's direct-path flag at that position;
* for the ray-traced model, the traced link energy itself.

Gains are reported relative to a fixed hall-average reference, computed once from
all bank positions (both splits), so Eb/N0 is the operating point of a terminal
with the average gain. See thesis/notes/SYSTEM_MODEL_V2.md.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf

SHADOWING_SIGMA_DB = {True: 4.3, False: 5.7}  # TR 38.901 InF-SL, LOS / NLOS


def inf_sl_path_loss_db(d3d_m: np.ndarray, los: np.ndarray, carrier_frequency_hz: float) -> np.ndarray:
    """TR 38.901 Table 7.4.1-1 InF-SL path loss in dB."""
    fc_ghz = carrier_frequency_hz / 1e9
    d = np.maximum(np.asarray(d3d_m, dtype=np.float64), 1.0)
    pl_los = 31.84 + 21.50 * np.log10(d) + 19.00 * np.log10(fc_ghz)
    pl_nlos = np.maximum(pl_los, 33.00 + 25.50 * np.log10(d) + 20.00 * np.log10(fc_ghz))
    return np.where(np.asarray(los, dtype=bool), pl_los, pl_nlos)


def _mean_shadowing_factor(sigma_db: np.ndarray) -> np.ndarray:
    """E[10^(S/10)] for S ~ N(0, sigma^2) in dB."""
    s = np.asarray(sigma_db, dtype=np.float64) * math.log(10.0) / 10.0
    return np.exp(0.5 * s * s)


class HallLargeScale:
    """Draws per-batch terminal positions and their large-scale gains."""

    def __init__(self, bank_path: str | Path, split: str, *, num_ut: int, carrier_frequency_hz: float) -> None:
        bank_path = Path(bank_path)
        manifest = json.loads(Path(str(bank_path) + ".manifest.json").read_text(encoding="utf-8"))
        self.bs_position = np.asarray(manifest["hall"]["bs_position"], dtype=np.float64)
        self.num_ut = int(num_ut)
        self.carrier_frequency_hz = float(carrier_frequency_hz)
        with np.load(bank_path) as data:
            splits = {s: (data[f"{s}_positions"].astype(np.float64), data[f"{s}_los"].astype(bool),
                          np.mean(np.abs(data[f"{s}_h"]) ** 2, axis=(1, 2)))
                      for s in ("train", "eval") if f"{s}_positions" in data}
        if split not in splits:
            raise KeyError(f"Channel bank {bank_path} has no split '{split}'.")
        if splits[split][0].shape[0] < self.num_ut:
            raise ValueError(f"Split '{split}' has fewer positions than terminals.")
        self.positions, self.los, rt_energy = splits[split]
        self.d3d = np.linalg.norm(self.positions - self.bs_position, axis=1)
        self.path_loss_db = inf_sl_path_loss_db(self.d3d, self.los, self.carrier_frequency_hz)
        self.sigma_db = np.where(self.los, SHADOWING_SIGMA_DB[True], SHADOWING_SIGMA_DB[False])
        self.rt_energy = rt_energy

        # Fixed references over every bank position, so both splits share one scale.
        all_pos = np.concatenate([v[0] for v in splits.values()])
        all_los = np.concatenate([v[1] for v in splits.values()])
        all_rt = np.concatenate([v[2] for v in splits.values()])
        all_pl = inf_sl_path_loss_db(np.linalg.norm(all_pos - self.bs_position, axis=1), all_los,
                                     self.carrier_frequency_hz)
        all_sigma = np.where(all_los, SHADOWING_SIGMA_DB[True], SHADOWING_SIGMA_DB[False])
        self.g_ref_statistical = float(np.mean(10.0 ** (-all_pl / 10.0) * _mean_shadowing_factor(all_sigma)))
        self.g_ref_raytraced = float(np.mean(all_rt))

        self._pl_tf = tf.constant(self.path_loss_db, tf.float32)
        self._sigma_tf = tf.constant(self.sigma_db, tf.float32)
        self.last: dict[str, np.ndarray] | None = None

    def draw_indices(self) -> tf.Tensor:
        """U distinct bank positions for one batch (uses the global TF generator)."""
        n = self.positions.shape[0]
        return tf.argsort(tf.random.uniform([n]))[: self.num_ut]

    def statistical_gain(self, idx: tf.Tensor) -> tf.Tensor:
        """Linear gain relative to the hall-average reference, [U] float32."""
        pl = tf.gather(self._pl_tf, idx)
        shadowing = tf.random.normal([self.num_ut]) * tf.gather(self._sigma_tf, idx)
        gain_db = -pl + shadowing - 10.0 * math.log10(self.g_ref_statistical)
        self._record(idx, gain_db)
        return tf.pow(10.0, gain_db / 10.0)

    def raytraced_scale(self, idx: tf.Tensor) -> float:
        """Amplitude scale that references traced responses to the hall average."""
        gain_db = 10.0 * np.log10(self.rt_energy[idx.numpy()] / self.g_ref_raytraced)
        self._record(idx, tf.constant(gain_db, tf.float32))
        return 1.0 / math.sqrt(self.g_ref_raytraced)

    def _record(self, idx: tf.Tensor, gain_db: tf.Tensor) -> None:
        i = idx.numpy()
        self.last = {
            "positions": self.positions[i].copy(),
            "los": self.los[i].copy(),
            "gain_db": np.asarray(gain_db.numpy() if hasattr(gain_db, "numpy") else gain_db, dtype=np.float64),
        }
