from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from factory6g.sim.rng import seed_all

project_root = Path(__file__).resolve().parents[1]

from factory6g.sim.config import load_config


DEFAULT_PROFILE_CONFIG = Path("config/factory_size_profiles.json")
CSV_TOP_PATHS = 10


@dataclass(frozen=True)
class ProfileDefinition:
    factory_scale: str
    scenario_label: str
    real_world_archetype: str
    narrative: str
    factory_scenario_overrides: dict[str, Any]
    transceiver_overrides: dict[str, Any]
    ray_tracing_overrides: dict[str, Any]
    dataset_defaults: dict[str, Any]


@dataclass(frozen=True)
class GeneratedProfileData:
    profile: ProfileDefinition
    profile_index: int
    seed: int
    frequency_hz: float
    room_dimensions: list[float]
    machine_layout: list[dict[str, float]]
    tx_position: list[float]
    rx_positions: np.ndarray
    paths_a: np.ndarray
    paths_tau: np.ndarray
    csv_rows: list[dict[str, Any]]


@dataclass
class ProfileOutput:
    generated: GeneratedProfileData
    h5_path: Path
    csv_path: Path
    visual_paths: dict[str, str] | None = None


def _count_suffix(count: int) -> str:
    if count % 1000 == 0:
        return f"{count // 1000}k"
    return str(count)


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _ensure_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be an object.")
    return value


def load_base_config(config_path: str) -> dict[str, Any]:
    return load_config(config_path).to_dict()


def _parse_profile_definition(scale_key: str, raw: dict[str, Any]) -> ProfileDefinition:
    required = {
        "factory_scale",
        "scenario_label",
        "real_world_archetype",
        "factory_scenario_overrides",
        "transceiver_overrides",
        "ray_tracing_overrides",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            f"Profile '{scale_key}' is missing required fields: {', '.join(missing)}"
        )
    factory_scale = str(raw["factory_scale"]).upper()
    if factory_scale != scale_key:
        raise ValueError(
            f"Profile '{scale_key}' has factory_scale='{factory_scale}'. Expected '{scale_key}'."
        )
    return ProfileDefinition(
        factory_scale=factory_scale,
        scenario_label=str(raw["scenario_label"]),
        real_world_archetype=str(raw["real_world_archetype"]),
        narrative=str(raw.get("narrative", "")),
        factory_scenario_overrides=_ensure_dict(
            raw["factory_scenario_overrides"],
            f"profiles.{scale_key}.factory_scenario_overrides",
        ),
        transceiver_overrides=_ensure_dict(
            raw["transceiver_overrides"],
            f"profiles.{scale_key}.transceiver_overrides",
        ),
        ray_tracing_overrides=_ensure_dict(
            raw["ray_tracing_overrides"],
            f"profiles.{scale_key}.ray_tracing_overrides",
        ),
        dataset_defaults=_ensure_dict(
            raw.get("dataset_defaults", {}),
            f"profiles.{scale_key}.dataset_defaults",
        ),
    )


def load_profile_defs(profile_config_path: str) -> dict[str, ProfileDefinition]:
    path = Path(profile_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Profile config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    profiles_raw = _ensure_dict(payload.get("profiles"), "profiles")
    required_scales = {"S", "M", "L"}
    missing_scales = sorted(required_scales - set(profiles_raw))
    if missing_scales:
        raise ValueError(
            f"Profile config must contain profiles for S, M, and L. Missing: {', '.join(missing_scales)}"
        )
    profiles: dict[str, ProfileDefinition] = {}
    for scale in sorted(required_scales):
        raw = _ensure_dict(profiles_raw[scale], f"profiles.{scale}")
        profiles[scale] = _parse_profile_definition(scale, raw)
    return profiles


def merge_profile_overrides(base_config: dict[str, Any], profile: ProfileDefinition) -> dict[str, Any]:
    merged = deepcopy(base_config)
    merged["factory_scenario"] = _deep_update(
        _ensure_dict(merged.get("factory_scenario", {}), "factory_scenario"),
        profile.factory_scenario_overrides,
    )
    merged["transceiver"] = _deep_update(
        _ensure_dict(merged.get("transceiver", {}), "transceiver"),
        profile.transceiver_overrides,
    )
    merged["ray_tracing"] = _deep_update(
        _ensure_dict(merged.get("ray_tracing", {}), "ray_tracing"),
        profile.ray_tracing_overrides,
    )
    return merged


def _summary_fieldnames() -> list[str]:
    fields = [
        "sample_index",
        "profile_index",
        "factory_scale",
        "scenario_label",
        "real_world_archetype",
        "pos_x",
        "pos_y",
        "pos_z",
        "bs_rx_distance_m",
        "valid_path_count",
        "dominant_path_mag",
        "dominant_path_delay",
    ]
    for idx in range(CSV_TOP_PATHS):
        fields.append(f"path_{idx}_mag")
        fields.append(f"path_{idx}_delay")
    return fields


def _build_summary_row(
    sample_index: int,
    profile: ProfileDefinition,
    profile_index: int,
    rx_pos: list[float],
    tx_pos: list[float],
    a_val: np.ndarray,
    tau_val: np.ndarray,
) -> dict[str, Any]:
    mag = np.mean(np.abs(a_val), axis=(1, 3)).flatten()
    delays = tau_val.flatten()
    valid_mask = delays >= 0
    valid_count = int(np.sum(valid_mask))
    if valid_count > 0:
        valid_indices = np.where(valid_mask)[0]
        best_local = valid_indices[int(np.argmax(mag[valid_indices]))]
        dominant_mag = float(mag[best_local])
        dominant_delay = float(delays[best_local])
    else:
        dominant_mag = 0.0
        dominant_delay = -1.0
    distance = float(np.linalg.norm(np.asarray(rx_pos, dtype=np.float32) - np.asarray(tx_pos, dtype=np.float32)))

    row: dict[str, Any] = {
        "sample_index": int(sample_index),
        "profile_index": int(profile_index),
        "factory_scale": profile.factory_scale,
        "scenario_label": profile.scenario_label,
        "real_world_archetype": profile.real_world_archetype,
        "pos_x": float(rx_pos[0]),
        "pos_y": float(rx_pos[1]),
        "pos_z": float(rx_pos[2]),
        "bs_rx_distance_m": distance,
        "valid_path_count": valid_count,
        "dominant_path_mag": dominant_mag,
        "dominant_path_delay": dominant_delay,
    }

    sorted_indices = np.argsort(mag)[::-1]
    top_idx = 0
    for idx in sorted_indices:
        if top_idx >= CSV_TOP_PATHS:
            break
        if delays[idx] < 0:
            continue
        row[f"path_{top_idx}_mag"] = float(mag[idx])
        row[f"path_{top_idx}_delay"] = float(delays[idx])
        top_idx += 1
    while top_idx < CSV_TOP_PATHS:
        row[f"path_{top_idx}_mag"] = 0.0
        row[f"path_{top_idx}_delay"] = -1.0
        top_idx += 1
    return row


def _resolve_box_ply_path(box_xml_path: str) -> Path:
    box_dir = Path(box_xml_path).parent
    box_ply_path = box_dir / "meshes" / "box.ply"
    if not box_ply_path.exists():
        raise FileNotFoundError(f"box.ply not found at {box_ply_path}.")
    return box_ply_path


def _build_scene_xml(
    factory_params: dict[str, Any],
    transceiver_params: dict[str, Any],
    *,
    box_ply_path: Path,
    rng: np.random.Generator,
) -> tuple[Path, list[float], list[dict[str, float]]]:
    xml_lines = ['<scene version="2.1.0">']
    xml_lines.append(
        '    <bsdf type="itu-radio-material" id="mat_default"><string name="type" value="concrete"/></bsdf>'
    )

    room = factory_params.get("room_dimensions", [20.0, 20.0, 6.0])
    room_dims = [float(room[0]), float(room[1]), float(room[2])]
    room_len, room_width, room_height = room_dims
    wall_thickness = float(transceiver_params.get("wall_thickness", 0.2))
    room_padding = float(transceiver_params.get("room_padding", 2.0))
    machine_ranges = factory_params.get(
        "machine_size_range",
        [[1.0, 3.0], [1.0, 3.0], [1.0, 2.5]],
    )
    num_machines = int(factory_params.get("num_machines", 5))
    machine_layout: list[dict[str, float]] = []

    def add_shape(shape_id: str, size_xyz: list[float], pos_xyz: list[float]) -> None:
        sx = float(size_xyz[0]) / 10.0
        sy = float(size_xyz[1]) / 10.0
        sz = float(size_xyz[2]) / 5.0
        xml_lines.append(f'    <shape type="ply" id="{shape_id}">')
        xml_lines.append(f'        <string name="filename" value="{box_ply_path}"/>')
        xml_lines.append('        <ref id="mat_default" name="bsdf"/>')
        xml_lines.append('        <transform name="to_world">')
        xml_lines.append(f'            <scale x="{sx}" y="{sy}" z="{sz}"/>')
        xml_lines.append(
            f'            <translate x="{float(pos_xyz[0])}" y="{float(pos_xyz[1])}" z="{float(pos_xyz[2])}"/>'
        )
        xml_lines.append("        </transform>")
        xml_lines.append("    </shape>")

    add_shape("floor", [room_len, room_width, wall_thickness], [0.0, 0.0, -wall_thickness])
    add_shape("ceiling", [room_len, room_width, wall_thickness], [0.0, 0.0, room_height])
    add_shape(
        "wall_left",
        [wall_thickness, room_width, room_height],
        [-room_len / 2 - wall_thickness / 2, 0.0, 0.0],
    )
    add_shape(
        "wall_right",
        [wall_thickness, room_width, room_height],
        [room_len / 2 + wall_thickness / 2, 0.0, 0.0],
    )
    add_shape(
        "wall_front",
        [room_len, wall_thickness, room_height],
        [0.0, -room_width / 2 - wall_thickness / 2, 0.0],
    )
    add_shape(
        "wall_back",
        [room_len, wall_thickness, room_height],
        [0.0, room_width / 2 + wall_thickness / 2, 0.0],
    )

    for machine_idx in range(num_machines):
        sx = float(rng.uniform(machine_ranges[0][0], machine_ranges[0][1]))
        sy = float(rng.uniform(machine_ranges[1][0], machine_ranges[1][1]))
        sz = float(rng.uniform(machine_ranges[2][0], machine_ranges[2][1]))
        px = float(rng.uniform(-room_len / 2 + room_padding, room_len / 2 - room_padding))
        py = float(rng.uniform(-room_width / 2 + room_padding, room_width / 2 - room_padding))
        add_shape(f"machine_{machine_idx}", [sx, sy, sz], [px, py, 0.0])
        machine_layout.append({"x": px, "y": py, "z": 0.0, "sx": sx, "sy": sy, "sz": sz})

    xml_lines.append("</scene>")
    scene_file = Path(f"factory_scene_{os.getpid()}_{int(rng.integers(1_000_000))}.xml")
    scene_file.write_text("\n".join(xml_lines), encoding="utf-8")
    return scene_file, room_dims, machine_layout


def _standardize_paths(
    a_val: np.ndarray,
    tau_val: np.ndarray,
    max_paths: int,
) -> tuple[np.ndarray, np.ndarray]:
    path_axis_a = 4
    path_axis_tau = 2
    current_paths = int(a_val.shape[path_axis_a])
    if current_paths < max_paths:
        a_pad_shape = list(a_val.shape)
        a_pad_shape[path_axis_a] = max_paths - current_paths
        tau_pad_shape = list(tau_val.shape)
        tau_pad_shape[path_axis_tau] = max_paths - current_paths
        a_val = np.concatenate([a_val, np.zeros(a_pad_shape, dtype=a_val.dtype)], axis=path_axis_a)
        tau_val = np.concatenate([tau_val, -np.ones(tau_pad_shape, dtype=tau_val.dtype)], axis=path_axis_tau)
    else:
        a_val = a_val[:, :, :, :, :max_paths, :]
        tau_val = tau_val[:, :, :max_paths]
    return a_val, tau_val


def _generate_profile_samples(
    config: dict[str, Any],
    profile: ProfileDefinition,
    *,
    profile_index: int,
    num_samples: int,
    seed: int,
    preview: bool,
) -> GeneratedProfileData:
    from sionna.rt import (
        PathSolver,
        PlanarArray,
        RadioMaterial,
        Receiver,
        Transmitter,
        load_scene,
    )
    from sionna.rt.scene import box as box_xml_path

    seed_all(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    factory_params = _ensure_dict(config.get("factory_scenario", {}), "factory_scenario")
    system_params = _ensure_dict(config.get("system", {}), "system")
    rt_params = _ensure_dict(config.get("ray_tracing", {}), "ray_tracing")
    transceiver_params = _ensure_dict(config.get("transceiver", {}), "transceiver")

    box_ply_path = _resolve_box_ply_path(box_xml_path)
    scene_xml_path, room_dims, machine_layout = _build_scene_xml(
        factory_params,
        transceiver_params,
        box_ply_path=box_ply_path,
        rng=rng,
    )

    try:
        scene = load_scene(str(scene_xml_path))

        materials_cfg = _ensure_dict(factory_params.get("materials", {}), "factory_scenario.materials")
        metal_cfg = _ensure_dict(materials_cfg.get("metal", {}), "factory_scenario.materials.metal")
        concrete_cfg = _ensure_dict(materials_cfg.get("concrete", {}), "factory_scenario.materials.concrete")
        metal = RadioMaterial(
            str(metal_cfg.get("name", "factory_metal")),
            relative_permittivity=float(metal_cfg.get("relative_permittivity", 1.0)),
            conductivity=float(metal_cfg.get("conductivity", 1e7)),
        )
        concrete = RadioMaterial(
            str(concrete_cfg.get("name", "factory_concrete")),
            relative_permittivity=float(concrete_cfg.get("relative_permittivity", 7.0)),
            conductivity=float(concrete_cfg.get("conductivity", 0.1)),
        )
        if metal.name not in scene.radio_materials:
            scene.add(metal)
        if concrete.name not in scene.radio_materials:
            scene.add(concrete)

        for name, scene_obj in scene.objects.items():
            if name.startswith("machine"):
                scene_obj.radio_material = metal.name
            else:
                scene_obj.radio_material = concrete.name
        if "mat_default" in scene.radio_materials:
            try:
                scene.remove("mat_default")
            except Exception:
                pass

        frequency_hz = float(system_params.get("carrier_frequency", 3.5e9))
        scene.frequency = frequency_hz
        scene.synthetic_array = True

        room_height = float(room_dims[2])
        tx_height_offset = float(transceiver_params.get("tx_height_offset", 1.0))
        tx_position = [0.0, 0.0, room_height - tx_height_offset]
        rx_height = float(transceiver_params.get("rx_height", 1.0))
        antenna_spacing = float(transceiver_params.get("antenna_spacing", 0.5))

        num_bs_ant = max(1, int(system_params.get("num_bs_ant", 8)))
        tx_rows = int(np.sqrt(num_bs_ant))
        while tx_rows > 1 and num_bs_ant % tx_rows != 0:
            tx_rows -= 1
        tx_cols = max(1, num_bs_ant // tx_rows)
        scene.tx_array = PlanarArray(
            num_rows=tx_rows,
            num_cols=tx_cols,
            vertical_spacing=antenna_spacing,
            horizontal_spacing=antenna_spacing,
            pattern=str(transceiver_params.get("tx_pattern", "tr38901")),
            polarization=str(transceiver_params.get("tx_polarization", "cross")),
        )
        scene.add(Transmitter("BS", position=tx_position))

        num_ut_ant = max(1, int(system_params.get("num_ut_ant", 1)))
        rx_rows = int(np.sqrt(num_ut_ant))
        while rx_rows > 1 and num_ut_ant % rx_rows != 0:
            rx_rows -= 1
        rx_cols = max(1, num_ut_ant // rx_rows)
        scene.rx_array = PlanarArray(
            num_rows=rx_rows,
            num_cols=rx_cols,
            vertical_spacing=antenna_spacing,
            horizontal_spacing=antenna_spacing,
            pattern=str(transceiver_params.get("rx_pattern", "iso")),
            polarization=str(transceiver_params.get("rx_polarization", "V")),
        )
        rx = Receiver("RX", position=[0.0, 0.0, rx_height])
        scene.add(rx)

        max_depth = int(rt_params.get("max_depth", 5))
        samples_per_src = int(rt_params.get("samples_per_src", 10_000))
        max_paths = int(rt_params.get("max_paths", 100))
        boundary_padding = float(transceiver_params.get("rx_boundary_padding", 1.0))
        room_len, room_width = float(room_dims[0]), float(room_dims[1])

        solver = PathSolver()
        rx_positions: list[list[float]] = []
        paths_a_list: list[np.ndarray] = []
        paths_tau_list: list[np.ndarray] = []
        csv_rows: list[dict[str, Any]] = []

        print(
            f"[{profile.factory_scale}] Generating {num_samples} samples "
            f"({profile.scenario_label}: {profile.real_world_archetype})"
        )
        for sample_idx in range(num_samples):
            rx_pos = [
                float(rng.uniform(-room_len / 2 + boundary_padding, room_len / 2 - boundary_padding)),
                float(rng.uniform(-room_width / 2 + boundary_padding, room_width / 2 - boundary_padding)),
                rx_height,
            ]
            rx.position = rx_pos
            paths = solver(scene, max_depth=max_depth, samples_per_src=samples_per_src)
            a_val, tau_val = paths.cir(out_type="numpy")
            a_std, tau_std = _standardize_paths(a_val, tau_val, max_paths=max_paths)

            rx_positions.append(rx_pos)
            paths_a_list.append(a_std)
            paths_tau_list.append(tau_std)
            csv_rows.append(
                _build_summary_row(
                    sample_idx,
                    profile=profile,
                    profile_index=profile_index,
                    rx_pos=rx_pos,
                    tx_pos=tx_position,
                    a_val=a_std,
                    tau_val=tau_std,
                )
            )
            if sample_idx % 10 == 0:
                print(f"[{profile.factory_scale}] Sample {sample_idx}/{num_samples}")

        if preview:
            scene.preview()

        return GeneratedProfileData(
            profile=profile,
            profile_index=profile_index,
            seed=seed,
            frequency_hz=frequency_hz,
            room_dimensions=room_dims,
            machine_layout=machine_layout,
            tx_position=tx_position,
            rx_positions=np.asarray(rx_positions, dtype=np.float32),
            paths_a=np.concatenate(paths_a_list, axis=0),
            paths_tau=np.concatenate(paths_tau_list, axis=0),
            csv_rows=csv_rows,
        )
    finally:
        try:
            scene_xml_path.unlink(missing_ok=True)
        except Exception:
            pass


def _write_h5_with_profile_metadata(
    *,
    output_path: Path,
    generated: GeneratedProfileData,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    utf8_dtype = h5py.string_dtype(encoding="utf-8")
    sample_count = int(generated.rx_positions.shape[0])
    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["frequency"] = float(generated.frequency_hz)
        h5f.attrs["factory_scale"] = generated.profile.factory_scale
        h5f.attrs["scenario_label"] = generated.profile.scenario_label
        h5f.attrs["real_world_archetype"] = generated.profile.real_world_archetype
        h5f.attrs["profile_index"] = int(generated.profile_index)
        h5f.attrs["seed"] = int(generated.seed)
        h5f.attrs["room_dimensions"] = json.dumps(generated.room_dimensions)
        h5f.attrs["tx_position"] = json.dumps(generated.tx_position)
        h5f.attrs["machine_layout"] = json.dumps(generated.machine_layout)
        h5f.create_dataset("rx_positions", data=generated.rx_positions)
        h5f.create_dataset("paths_a", data=generated.paths_a)
        h5f.create_dataset("paths_tau", data=generated.paths_tau)
        h5f.create_dataset(
            "profile_index",
            data=np.full(sample_count, generated.profile_index, dtype=np.int32),
        )
        h5f.create_dataset(
            "factory_scale",
            data=np.asarray([generated.profile.factory_scale] * sample_count, dtype=object),
            dtype=utf8_dtype,
        )
        h5f.create_dataset(
            "scenario_label",
            data=np.asarray([generated.profile.scenario_label] * sample_count, dtype=object),
            dtype=utf8_dtype,
        )
        h5f.create_dataset(
            "real_world_archetype",
            data=np.asarray([generated.profile.real_world_archetype] * sample_count, dtype=object),
            dtype=utf8_dtype,
        )


def write_profile_outputs(
    generated: GeneratedProfileData,
    *,
    output_dir: str,
    dataset_prefix: str,
) -> ProfileOutput:
    output_root = Path(output_dir)
    sample_count = int(generated.rx_positions.shape[0])
    suffix = _count_suffix(sample_count)
    base_name = f"{dataset_prefix}_{generated.profile.factory_scale}_{suffix}"
    h5_path = output_root / f"{base_name}.h5"
    csv_path = output_root / f"{base_name}.csv"

    _write_h5_with_profile_metadata(output_path=h5_path, generated=generated)

    output_root.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = _summary_fieldnames()
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in generated.csv_rows:
            writer.writerow(row)
    print(f"[OK] Saved profile dataset: {h5_path}")
    print(f"[OK] Saved profile CSV: {csv_path}")
    return ProfileOutput(generated=generated, h5_path=h5_path, csv_path=csv_path)


def _serialize_combined_csv_rows(profile_outputs: list[ProfileOutput]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_index = 0
    for profile_output in profile_outputs:
        for profile_row in profile_output.generated.csv_rows:
            row_copy = dict(profile_row)
            row_copy["global_sample_index"] = int(global_index)
            row_copy["profile_sample_index"] = int(profile_row["sample_index"])
            rows.append(row_copy)
            global_index += 1
    return rows


def write_combined_outputs(
    profile_outputs: list[ProfileOutput],
    *,
    output_dir: str,
    dataset_prefix: str,
) -> dict[str, Path]:
    if not profile_outputs:
        raise ValueError("No profile outputs were provided for combined export.")

    output_root = Path(output_dir)
    scales_token = "".join([item.generated.profile.factory_scale for item in profile_outputs])
    total_samples = int(sum(item.generated.rx_positions.shape[0] for item in profile_outputs))
    total_suffix = _count_suffix(total_samples)
    base_name = f"{dataset_prefix}_{scales_token}_{total_suffix}"
    h5_path = output_root / f"{base_name}.h5"
    csv_path = output_root / f"{base_name}.csv"
    manifest_path = output_root / f"{base_name}_manifest.json"
    utf8_dtype = h5py.string_dtype(encoding="utf-8")

    max_paths = max(int(item.generated.paths_a.shape[4]) for item in profile_outputs)

    def _pad_paths(
        paths_a: np.ndarray,
        paths_tau: np.ndarray,
        target_paths: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        current_paths = int(paths_a.shape[4])
        if current_paths == target_paths:
            return paths_a, paths_tau
        pad_count = target_paths - current_paths
        if pad_count < 0:
            return paths_a[:, :, :, :, :target_paths, :], paths_tau[:, :, :target_paths]
        a_pad = np.zeros(
            (
                paths_a.shape[0],
                paths_a.shape[1],
                paths_a.shape[2],
                paths_a.shape[3],
                pad_count,
                paths_a.shape[5],
            ),
            dtype=paths_a.dtype,
        )
        tau_pad = -np.ones((paths_tau.shape[0], paths_tau.shape[1], pad_count), dtype=paths_tau.dtype)
        return (
            np.concatenate([paths_a, a_pad], axis=4),
            np.concatenate([paths_tau, tau_pad], axis=2),
        )

    combined_rx = np.concatenate([item.generated.rx_positions for item in profile_outputs], axis=0)
    padded_paths = [_pad_paths(item.generated.paths_a, item.generated.paths_tau, max_paths) for item in profile_outputs]
    combined_paths_a = np.concatenate([item[0] for item in padded_paths], axis=0)
    combined_paths_tau = np.concatenate([item[1] for item in padded_paths], axis=0)
    combined_profile_index = np.concatenate(
        [
            np.full(item.generated.rx_positions.shape[0], item.generated.profile_index, dtype=np.int32)
            for item in profile_outputs
        ],
        axis=0,
    )
    combined_scale = np.concatenate(
        [
            np.asarray([item.generated.profile.factory_scale] * item.generated.rx_positions.shape[0], dtype=object)
            for item in profile_outputs
        ],
        axis=0,
    )
    combined_label = np.concatenate(
        [
            np.asarray([item.generated.profile.scenario_label] * item.generated.rx_positions.shape[0], dtype=object)
            for item in profile_outputs
        ],
        axis=0,
    )
    combined_archetype = np.concatenate(
        [
            np.asarray(
                [item.generated.profile.real_world_archetype] * item.generated.rx_positions.shape[0],
                dtype=object,
            )
            for item in profile_outputs
        ],
        axis=0,
    )
    reference_frequency = float(profile_outputs[0].generated.frequency_hz)

    output_root.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as h5f:
        h5f.attrs["frequency"] = reference_frequency
        h5f.attrs["combined_scales"] = scales_token
        h5f.attrs["total_samples"] = total_samples
        h5f.create_dataset("rx_positions", data=combined_rx)
        h5f.create_dataset("paths_a", data=combined_paths_a)
        h5f.create_dataset("paths_tau", data=combined_paths_tau)
        h5f.create_dataset("profile_index", data=combined_profile_index)
        h5f.create_dataset(
            "factory_scale",
            data=np.asarray(combined_scale, dtype=object),
            dtype=utf8_dtype,
        )
        h5f.create_dataset(
            "scenario_label",
            data=np.asarray(combined_label, dtype=object),
            dtype=utf8_dtype,
        )
        h5f.create_dataset(
            "real_world_archetype",
            data=np.asarray(combined_archetype, dtype=object),
            dtype=utf8_dtype,
        )

    combined_rows = _serialize_combined_csv_rows(profile_outputs)
    combined_fieldnames = ["global_sample_index", "profile_sample_index"] + _summary_fieldnames()
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=combined_fieldnames)
        writer.writeheader()
        for row in combined_rows:
            writer.writerow(row)

    profile_entries: list[dict[str, Any]] = []
    for output in profile_outputs:
        generated = output.generated
        profile_entries.append(
            {
                "factory_scale": generated.profile.factory_scale,
                "profile_index": int(generated.profile_index),
                "scenario_label": generated.profile.scenario_label,
                "real_world_archetype": generated.profile.real_world_archetype,
                "narrative": generated.profile.narrative,
                "sample_count": int(generated.rx_positions.shape[0]),
                "seed": int(generated.seed),
                "room_dimensions": generated.room_dimensions,
                "tx_position": generated.tx_position,
                "machine_layout": generated.machine_layout,
                "h5_path": str(output.h5_path),
                "csv_path": str(output.csv_path),
                "visual_paths": output.visual_paths or {},
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_prefix": dataset_prefix,
        "profiles_token": scales_token,
        "total_samples": total_samples,
        "combined_h5_path": str(h5_path),
        "combined_csv_path": str(csv_path),
        "profile_index_mapping": {
            item.generated.profile.factory_scale: int(item.generated.profile_index)
            for item in profile_outputs
        },
        "profiles": profile_entries,
        "scenario_narratives": {
            item.generated.profile.factory_scale: item.generated.profile.narrative
            for item in profile_outputs
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[OK] Saved combined dataset: {h5_path}")
    print(f"[OK] Saved combined CSV: {csv_path}")
    print(f"[OK] Saved dataset manifest: {manifest_path}")
    return {"h5_path": h5_path, "csv_path": csv_path, "manifest_path": manifest_path}


def generate_single_profile_dataset(
    config: dict[str, Any],
    profile: ProfileDefinition,
    *,
    profile_index: int,
    num_samples: int,
    seed: int,
    preview: bool,
) -> GeneratedProfileData:
    return _generate_profile_samples(
        config,
        profile,
        profile_index=profile_index,
        num_samples=num_samples,
        seed=seed,
        preview=preview,
    )


def _parse_profiles(raw_profiles: str) -> list[str]:
    profiles = [item.strip().upper() for item in raw_profiles.split(",") if item.strip()]
    if not profiles:
        raise ValueError("No profiles selected. Use --profiles with at least one of S, M, L.")
    invalid = [item for item in profiles if item not in {"S", "M", "L"}]
    if invalid:
        raise ValueError(f"Unsupported profile(s): {', '.join(invalid)}")
    return profiles


def run_profile_pipeline(args: argparse.Namespace) -> dict[str, Path]:
    base_config = load_base_config(args.config)
    profile_defs = load_profile_defs(args.profile_config)
    selected_profiles = _parse_profiles(args.profiles)
    if args.preview and len(selected_profiles) != 1:
        raise ValueError("--preview can only be used when a single profile is selected.")

    simulation_cfg = _ensure_dict(base_config.get("simulation", {}), "simulation")
    gpu_id = int(simulation_cfg.get("gpu_id", 0))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))

    profile_outputs: list[ProfileOutput] = []
    for index, scale in enumerate(selected_profiles):
        definition = profile_defs[scale]
        merged_config = merge_profile_overrides(base_config, definition)
        seed_offset = int(definition.dataset_defaults.get("seed_offset", index * 1000))
        profile_seed = int(args.seed) + seed_offset

        generated = generate_single_profile_dataset(
            merged_config,
            definition,
            profile_index=index,
            num_samples=int(args.samples_per_profile),
            seed=profile_seed,
            preview=bool(args.preview),
        )
        profile_output = write_profile_outputs(
            generated,
            output_dir=args.output_dir,
            dataset_prefix=args.dataset_prefix,
        )

        if args.generate_visuals:
            from scripts.tools.visualize_factory_profiles import generate_profile_visuals

            profile_output.visual_paths = generate_profile_visuals(generated, args.visuals_dir)
        profile_outputs.append(profile_output)

    return write_combined_outputs(
        profile_outputs,
        output_dir=args.output_dir,
        dataset_prefix=args.dataset_prefix,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate S/M/L synthetic factory datasets and combined artifacts."
    )
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to the base simulation config.")
    parser.add_argument(
        "--profile-config",
        type=str,
        default=str(DEFAULT_PROFILE_CONFIG),
        help="Path to profile definition JSON.",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="S,M,L",
        help="Comma-separated profile list. Supported values: S,M,L.",
    )
    parser.add_argument(
        "--samples-per-profile",
        type=int,
        default=1000,
        help="Number of channel samples generated for each selected profile.",
    )
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for datasets.")
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default="factory_dataset",
        help="Dataset filename prefix.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--generate-visuals",
        action="store_true",
        help="Generate profile visuals (2D, 3D, KPI panel).",
    )
    parser.add_argument(
        "--visuals-dir",
        type=str,
        default="results/factory_profiles",
        help="Output directory for generated profile visuals.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview scene for a single selected profile only.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_profile_pipeline(args)


if __name__ == "__main__":
    main()
