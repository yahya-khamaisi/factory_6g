from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

# Legacy supervised DRL scheduler (not used by any campaign); TensorFlow stack only.
tf = pytest.importorskip("tensorflow")

from factory6g.models.drl_policy import (
    POLICY_HISTORY_FILENAME,
    POLICY_METADATA_FILENAME,
    POLICY_MODEL_FILENAME,
    POLICY_NORMALIZATION_FILENAME,
    build_policy_state,
    build_policy_training_inputs,
    compile_policy_model,
    create_policy_model,
    load_policy_checkpoint,
    predict_policy_outputs,
    project_policy_to_directives,
    save_policy_checkpoint,
)
from scripts.tools.train_drl_resource_manager import _ber_log_reliability_target
from factory6g.models.resource_manager import create_resource_manager
from factory6g.sim.types import ResourceManagerFeedback


def _synthetic_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channel_energy = np.array(
        [
            [[1.2, 1.0, 0.8, 0.7], [0.9, 0.8, 0.6, 0.5], [0.4, 0.3, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]],
            [[1.1, 0.9, 0.8, 0.6], [0.8, 0.8, 0.7, 0.6], [0.5, 0.4, 0.4, 0.3], [0.3, 0.2, 0.2, 0.2]],
            [[0.3, 0.3, 0.2, 0.2], [1.2, 1.0, 0.9, 0.9], [1.1, 1.0, 0.8, 0.7], [0.2, 0.2, 0.2, 0.1]],
            [[0.4, 0.4, 0.3, 0.2], [1.0, 1.0, 0.9, 0.8], [0.9, 0.8, 0.8, 0.7], [0.2, 0.2, 0.1, 0.1]],
            [[0.2, 0.2, 0.1, 0.1], [0.3, 0.2, 0.2, 0.2], [1.3, 1.1, 1.0, 0.9], [1.1, 1.0, 0.8, 0.7]],
            [[0.2, 0.2, 0.1, 0.1], [0.4, 0.3, 0.2, 0.2], [1.1, 1.1, 1.0, 0.8], [1.0, 0.9, 0.8, 0.8]],
        ],
        dtype=np.float32,
    )
    ebno_db = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float32)
    active_mask = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    per_ut_power = np.array(
        [
            [1.0, 0.8, 0.0, 0.0],
            [1.0, 0.85, 0.0, 0.0],
            [0.0, 1.0, 0.9, 0.0],
            [0.0, 1.0, 0.85, 0.0],
            [0.0, 0.0, 1.0, 0.9],
            [0.0, 0.0, 1.0, 0.95],
        ],
        dtype=np.float32,
    )
    utility = np.array([0.75, 0.78, 0.82, 0.85, 0.88, 0.9], dtype=np.float32).reshape(-1, 1)
    return channel_energy, ebno_db, active_mask, per_ut_power, utility


def _make_feedback() -> ResourceManagerFeedback:
    h_hat = np.zeros((1, 1, 1, 4, 1, 1, 4), dtype=np.complex64)
    user_vectors = np.array(
        [
            [2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.5 + 0.0j, 0.2 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j, 0.1 + 0.0j],
        ],
        dtype=np.complex64,
    )
    for ut_idx in range(4):
        h_hat[0, 0, 0, ut_idx, 0, 0, :] = user_vectors[ut_idx]
    err_var = np.zeros_like(h_hat, dtype=np.float32)
    return ResourceManagerFeedback(h_hat=tf.constant(h_hat), err_var=tf.constant(err_var))


def _train_and_save_checkpoint(tmp_path):
    tf.random.set_seed(7)
    np.random.seed(7)

    channel_energy, ebno_db, active_mask, per_ut_power, utility = _synthetic_training_data()
    x_state, normalization = build_policy_training_inputs(channel_energy, ebno_db)

    model = create_policy_model(tuple(x_state.shape[1:]), output_dim=4, hidden_dim=16, dropout_rate=0.0)
    compile_policy_model(model, learning_rate=0.005, value_loss_weight=0.05)
    history = model.fit(
        x_state,
        {
            "schedule_output": active_mask,
            "power_output": per_ut_power,
            "value_output": utility,
        },
        epochs=1,
        batch_size=2,
        verbose=0,
        validation_split=0.0,
    )

    checkpoint_dir = save_policy_checkpoint(
        tmp_path / "policy_checkpoint",
        model,
        normalization,
        {
            "num_ut": 4,
            "fft_size": 4,
            "state_dim": int(x_state.shape[2]),
            "checkpoint_type": "test_fixture",
        },
        history=history.history,
    )
    return checkpoint_dir, channel_energy, ebno_db


def _patch_root_module_to_keras_src_models(policy_model_path):
    with zipfile.ZipFile(policy_model_path, "r") as src_archive:
        archive_entries = [(info, src_archive.read(info.filename)) for info in src_archive.infolist()]
        config_payload = src_archive.read("config.json")
        config_json = json.loads(config_payload.decode("utf-8"))
        config_json["module"] = "keras.src.models.functional"
        patched_config = json.dumps(config_json).encode("utf-8")

    patched_path = policy_model_path.with_suffix(".patched.keras")
    with zipfile.ZipFile(patched_path, "w") as dst_archive:
        for info, payload in archive_entries:
            if info.filename == "config.json":
                payload = patched_config
            dst_archive.writestr(info, payload)
    patched_path.replace(policy_model_path)


def test_drl_policy_checkpoint_round_trip(tmp_path):
    checkpoint_dir, channel_energy, ebno_db = _train_and_save_checkpoint(tmp_path)

    assert (checkpoint_dir / POLICY_MODEL_FILENAME).exists()
    assert (checkpoint_dir / POLICY_METADATA_FILENAME).exists()
    assert (checkpoint_dir / POLICY_NORMALIZATION_FILENAME).exists()
    assert (checkpoint_dir / POLICY_HISTORY_FILENAME).exists()

    checkpoint = load_policy_checkpoint(checkpoint_dir)
    with (checkpoint_dir / POLICY_METADATA_FILENAME).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    assert checkpoint.metadata["format"] == metadata["format"]
    state = build_policy_state(channel_energy[0], ebno_db[0], fairness_debt=np.ones(4, dtype=np.float32))
    outputs = predict_policy_outputs(checkpoint, state)
    mask, power = project_policy_to_directives(
        np.asarray(outputs["schedule_output"], dtype=np.float32),
        np.asarray(outputs["power_output"], dtype=np.float32),
        num_active=2,
    )

    assert np.asarray(outputs["schedule_output"]).shape == (4,)
    assert np.asarray(outputs["power_output"]).shape == (4,)
    assert sum(mask) == 2
    assert len(power) == 4
    assert all(0.0 <= value <= 1.0 for value in power)


def test_drl_policy_checkpoint_loads_keras_src_models_archive(tmp_path):
    checkpoint_dir, channel_energy, ebno_db = _train_and_save_checkpoint(tmp_path)
    _patch_root_module_to_keras_src_models(checkpoint_dir / POLICY_MODEL_FILENAME)

    checkpoint = load_policy_checkpoint(checkpoint_dir)
    state = build_policy_state(channel_energy[0], ebno_db[0], fairness_debt=np.ones(4, dtype=np.float32))
    outputs = predict_policy_outputs(checkpoint, state)

    assert np.asarray(outputs["schedule_output"]).shape == (4,)
    assert np.asarray(outputs["power_output"]).shape == (4,)


def test_drl_resource_manager_loads_policy_checkpoint(tmp_path):
    checkpoint_dir, _, _ = _train_and_save_checkpoint(tmp_path)
    manager = create_resource_manager(
        "drl",
        num_ut=4,
        num_active=2,
        cnn_model_path=None,
        drl_model_path=str(checkpoint_dir),
        manager_kwargs={},
    )
    directives = manager.get_runtime_directives({"num_ut": 4}, ebno_db=5.0, feedback=_make_feedback())

    assert directives.active_ut_mask is not None
    assert directives.per_ut_power is not None
    assert sum(directives.active_ut_mask) == 2
    assert len(directives.per_ut_power) == 4
    assert all(0.0 <= value <= 1.0 for value in directives.per_ut_power)


def test_reliability_drl_resource_manager_uses_independent_checkpoint_path(tmp_path):
    checkpoint_dir, _, _ = _train_and_save_checkpoint(tmp_path)
    manager = create_resource_manager(
        "reliability_drl",
        num_ut=4,
        num_active=2,
        cnn_model_path=None,
        drl_model_path="unused-drl-path",
        manager_kwargs={"model_path": str(checkpoint_dir)},
    )
    directives = manager.get_runtime_directives({"num_ut": 4}, ebno_db=5.0, feedback=_make_feedback())

    assert getattr(manager, "policy_checkpoint", None) is not None
    assert directives.active_ut_mask is not None
    assert directives.per_ut_power is not None
    assert sum(directives.active_ut_mask) == 2
    assert len(directives.per_ut_power) == 4


def test_drl_policy_channel_gain_guard_biases_projection_to_stronger_users(tmp_path):
    checkpoint_dir, _, _ = _train_and_save_checkpoint(tmp_path)
    manager = create_resource_manager(
        "reliability_drl",
        num_ut=4,
        num_active=2,
        cnn_model_path=None,
        drl_model_path="unused-drl-path",
        manager_kwargs={
            "model_path": str(checkpoint_dir),
            "channel_gain_weight": 100.0,
            "fairness_weight": 0.0,
            "min_active_power": 0.85,
        },
    )
    directives = manager.get_runtime_directives({"num_ut": 4}, ebno_db=5.0, feedback=_make_feedback())

    assert directives.active_ut_mask == [1, 1, 0, 0]
    assert directives.per_ut_power is not None
    assert all(value == 0.0 or value >= 0.85 for value in directives.per_ut_power)


def test_ber_log_reliability_target_separates_low_ber_values():
    target = _ber_log_reliability_target(
        np.array([1e-7, 1e-5, 1e-3, 1e-2], dtype=np.float32),
        ber_clip=1e-2,
        ber_floor=1e-7,
    ).reshape(-1)

    assert np.all(np.diff(target) < 0.0)
    assert target[0] == 1.0
    assert target[-1] == 0.0
