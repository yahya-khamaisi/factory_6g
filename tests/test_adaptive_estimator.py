from __future__ import annotations

import time

import numpy as np
import pytest
import tensorflow as tf
from sionna.phy.ofdm import ResourceGrid

from factory6g.components.estimators import (
    AdaptiveHybridChannelEstimator,
    LMMSEChannelEstimator,
    select_quality_branch,
)
from factory6g.models.model import Model

from .conftest import make_tiny_config, set_all_seeds


def _tiny_resource_grid() -> ResourceGrid:
    return ResourceGrid(
        num_ofdm_symbols=4,
        fft_size=16,
        subcarrier_spacing=30000.0,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=4,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[1, 3],
    )


def test_select_quality_branch_covers_all_ranges():
    assert select_quality_branch(0.5, 3.0, 12.0) == "low"
    assert select_quality_branch(6.0, 3.0, 12.0) == "mid"
    assert select_quality_branch(30.0, 3.0, 12.0) == "high"


def test_adaptive_estimator_shape_dtype_and_branch_selection():
    # Single-branch (v1) rule; the per-terminal rule is tested in test_estimator_v2.py.
    estimator = AdaptiveHybridChannelEstimator(
        _tiny_resource_grid(), quality_low=3.0, quality_high=12.0, per_terminal_branch=False
    )
    h_ls = tf.complex(tf.ones([1, 1, 1, 1, 1, 4, 16], dtype=tf.float32), tf.zeros([1, 1, 1, 1, 1, 4, 16]))
    err_var_ls = tf.ones([1, 1, 1, 1, 1, 4, 16], dtype=tf.float32)

    h_low, e_low, branch_low = estimator.estimate_from_ls(h_ls, err_var_ls, tf.constant(100.0, tf.float32))
    h_mid, e_mid, branch_mid = estimator.estimate_from_ls(h_ls, err_var_ls, tf.constant(0.2, tf.float32))
    h_high, e_high, branch_high = estimator.estimate_from_ls(h_ls, err_var_ls, tf.constant(0.01, tf.float32))

    assert branch_low == "low"
    assert branch_mid == "mid"
    assert branch_high == "high"
    assert h_low.dtype == tf.complex64
    assert h_mid.dtype == tf.complex64
    assert h_high.dtype == tf.complex64
    assert e_low.dtype == tf.float32
    assert e_mid.dtype == tf.float32
    assert e_high.dtype == tf.float32
    assert tuple(h_low.shape) == (1, 1, 1, 1, 1, 4, 16)
    assert tuple(e_low.shape) == (1, 1, 1, 1, 1, 4, 16)


def test_lmmse_spectral_shrinkage_matches_direct_inverse():
    rg = _tiny_resource_grid()
    estimator = LMMSEChannelEstimator(rg, r_freq=0.97, noise_bin_db=0.0)

    tf.random.set_seed(2026)
    real = tf.random.normal([2, 1, 1, 1, 1, 4, 16], dtype=tf.float32)
    imag = tf.random.normal([2, 1, 1, 1, 1, 4, 16], dtype=tf.float32)
    h_ls = tf.complex(real, imag)
    err_var_ls = tf.ones([2, 1, 1, 1, 1, 4, 16], dtype=tf.float32)
    no = tf.constant(0.3, dtype=tf.float32)

    h_fast, err_fast = estimator.estimate_from_ls(h_ls, err_var_ls, no)

    r = estimator.R_freq
    eye = tf.eye(estimator.fft_size, dtype=tf.complex64)
    w_ref = tf.matmul(r, tf.linalg.inv(r + tf.cast(no, tf.complex64) * eye))
    h_ref = tf.reshape(
        tf.matmul(tf.reshape(h_ls, [-1, estimator.fft_size]), tf.transpose(w_ref)),
        tf.shape(h_ls),
    )
    smooth = tf.math.real(tf.linalg.trace(w_ref)) / float(estimator.fft_size)
    err_ref = err_var_ls * smooth

    assert tf.reduce_max(tf.abs(h_fast - h_ref)).numpy() < 1e-4
    assert tf.reduce_max(tf.abs(err_fast - err_ref)).numpy() < 1e-5


def _benchmark_model(
    *,
    estimator_type: str,
    estimator_kwargs: dict[str, float] | None,
    runtime_config: dict[str, object],
    ebno_values: list[float],
    reps: int,
) -> tuple[float, float]:
    model = Model(
        config=runtime_config,
        estimator_type=estimator_type,
        estimator_kwargs=estimator_kwargs or {},
        perfect_csi=False,
    )
    runtime_total = 0.0
    ber_samples: list[float] = []
    for ebno in ebno_values:
        for rep in range(reps):
            set_all_seeds(4000 + int(ebno * 10) + rep)
            context = model.prepare_batch_context(batch_size=1, ebno_db=ebno, include_feedback=False)
            start = time.perf_counter()
            result = model.run_batch(context, include_details=False)
            runtime_total += time.perf_counter() - start
            ber_samples.append(float(np.mean(result["bits"] != result["bits_hat"])))
    return runtime_total, float(np.mean(ber_samples))


@pytest.mark.slow
def test_adaptive_vs_lmmse_runtime_accuracy_tradeoff_gate():
    config_data = make_tiny_config("results")
    config_data["system"]["fft_size"] = 64
    runtime_config = config_data["system"] | config_data["transceiver"]
    ebno_values = [18.0, 22.0]

    lmmse_runtime, lmmse_ber = _benchmark_model(
        estimator_type="lmmse",
        estimator_kwargs={"r_freq": 0.98, "noise_bin_db": 0.5},
        runtime_config=runtime_config,
        ebno_values=ebno_values,
        reps=2,
    )
    adaptive_runtime, adaptive_ber = _benchmark_model(
        estimator_type="adaptive",
        estimator_kwargs={
            "quality_low": 0.1,
            "quality_high": 0.2,
            "blend_mid_weight": 0.5,
            "dft_tap_ratio": 1.0,
            "lmmse_r_freq": 0.98,
        },
        runtime_config=runtime_config,
        ebno_values=ebno_values,
        reps=2,
    )

    assert adaptive_runtime <= 0.8 * lmmse_runtime
    assert adaptive_ber - lmmse_ber <= 0.005
