"""Coupled estimator x scheduler stage: shared probe (P1) and method keys."""

import dataclasses

import numpy as np

from factory6g.models.model import Model
from factory6g.sim.config import load_config
from factory6g.sim.stages.coupled import (
    coupled_method_key,
    run_coupled_stage,
    split_coupled_method_key,
)


def _runtime_config(channel: str = "rayleigh") -> dict:
    cfg = dict(load_config("config/config.json").system_runtime_config)
    cfg["channel_model_type"] = channel
    return cfg


def test_method_key_round_trip():
    key = coupled_method_key("neural", "reliability_drl")
    assert key == "neural+reliability_drl"
    assert split_coupled_method_key(key) == ("neural", "reliability_drl")


def test_feedback_uses_the_context_probe():
    cfg = _runtime_config()
    ctx = Model(cfg, estimator_type="ls").prepare_batch_context(batch_size=2, ebno_db=4.0, include_feedback=False)

    perfect = Model(cfg, estimator_type="perfect", perfect_csi=True).estimate_feedback(ctx)
    np.testing.assert_array_equal(perfect.h_hat.numpy(), ctx.h_freq.numpy())

    # Two LS models built separately see the same probe, so their feedback is identical.
    fb_a = Model(cfg, estimator_type="ls").estimate_feedback(ctx)
    fb_b = Model(cfg, estimator_type="ls").estimate_feedback(ctx)
    np.testing.assert_allclose(fb_a.h_hat.numpy(), fb_b.h_hat.numpy())


def test_coupled_stage_crosses_rosters():
    config = load_config("config/config.json")
    config = dataclasses.replace(
        config,
        estimators=dataclasses.replace(config.estimators, enabled=["ls", "perfect"]),
        resource_managers=dataclasses.replace(config.resource_managers, enabled=["static", "max_throughput"]),
        monte_carlo=dataclasses.replace(
            config.monte_carlo, batch_size=2, min_batches=1, max_batches=1, min_total_bits=1,
            ebno_min=4.0, ebno_max=4.0,
        ),
    )
    result = run_coupled_stage(config)
    assert result["stage"] == "coupled"
    assert sorted(result["methods"]) == [
        "ls+max_throughput", "ls+static", "perfect+max_throughput", "perfect+static",
    ]
    for metrics in result["methods"].values():
        assert len(metrics["ber"]) == len(result["ebno_db_range"])
