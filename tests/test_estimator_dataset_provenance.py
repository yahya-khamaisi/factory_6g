"""The estimator training dataset must record where it came from.

Without these fields a set of trained weights cannot be reported against its
training distribution, which is what made the retained Chapter 3 manifest
unusable as provenance.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from factory6g.training.train_estimator import generate_dataset

from .conftest import make_tiny_config


def _flat_config(tmp_path) -> dict:
    flat: dict = {}
    for section in make_tiny_config(str(tmp_path)).values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


def test_manifest_records_generation_provenance(tmp_path):
    output_path = str(tmp_path / "train.npz")

    generate_dataset(
        config=_flat_config(tmp_path),
        output_path=output_path,
        num_batches=1,
        ebno_db_range=(3.0, 7.0),
        channel_types=("rayleigh",),
        seed=99,
    )

    manifest = json.loads((tmp_path / "train.npz.manifest.json").read_text())

    assert manifest["channel_types"] == ["rayleigh"]
    assert manifest["seed"] == 99
    assert manifest["ebno_db_range"] == [3.0, 7.0]
    assert manifest["num_generation_batches"] == 1
    assert manifest["total_samples"] > 0
    assert manifest["chunks"]
    assert manifest["created_at_utc"]


def test_same_seed_reproduces_the_same_samples(tmp_path):
    def run(tag: str) -> np.ndarray:
        from .conftest import set_all_seeds

        set_all_seeds(7)
        output_path = str(tmp_path / f"{tag}.npz")
        generate_dataset(
            config=_flat_config(tmp_path),
            output_path=output_path,
            num_batches=1,
            ebno_db_range=(5.0, 5.0),
            channel_types=("rayleigh",),
            seed=7,
        )
        with np.load(f"{output_path}.chunk0.npz") as data:
            return data["h_ls"].copy()

    np.testing.assert_array_equal(run("a"), run("b"))


def test_dataset_stores_conditioning_and_training_records_it(tmp_path):
    from factory6g.components.estimators.neural_estimator import LS_ENERGY_CONDITIONING, load_conditioning
    from factory6g.training.train_estimator import train

    output_path = str(tmp_path / "train.npz")
    generate_dataset(
        config=_flat_config(tmp_path),
        output_path=output_path,
        num_batches=2,
        ebno_db_range=(0.0, 30.0),
        channel_types=("rayleigh",),
        seed=5,
    )
    manifest = json.loads((tmp_path / "train.npz.manifest.json").read_text())
    assert manifest["conditioning"] == LS_ENERGY_CONDITIONING
    with np.load(manifest["chunks"][0]) as chunk:
        assert chunk["cond_db"].shape == chunk["ebno_db"].shape

    model_path = str(tmp_path / "net.keras")
    meta = train(output_path, model_path, epochs=1, batch_size=8, seed=5, loss_weighting="noise")
    assert meta["training"]["loss_weighting"] == "noise"
    assert load_conditioning(model_path) == LS_ENERGY_CONDITIONING
    assert meta["test_nmse"] and meta["test_nmse_fine"] and meta["test_nmse_by_cond"]

    from factory6g.training.train_estimator import evaluate

    refreshed = evaluate(output_path, model_path, batch_size=8)
    assert refreshed["training"]["loss_weighting"] == "noise"  # training record kept
    assert [r["nmse_neural_db"] for r in refreshed["test_nmse"]] == pytest.approx(
        [r["nmse_neural_db"] for r in meta["test_nmse"]], abs=1e-4
    )
