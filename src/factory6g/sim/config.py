from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


class ConfigError(ValueError):
    """Raised when the user configuration is missing required fields or uses unsupported keys."""


def _strip_json_comments(raw_text: str) -> str:
    """Strip // and /* */ comments while preserving string literals."""
    result: list[str] = []
    i = 0
    in_string = False
    in_line_comment = False
    in_block_comment = False
    escape = False
    length = len(raw_text)

    while i < length:
        ch = raw_text[i]
        nxt = raw_text[i + 1] if i + 1 < length else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                result.append(ch)
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            i += 1
            continue

        if ch == "\"":
            in_string = True
            result.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def _ensure_dict(value: Any, section_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"Section '{section_name}' must be an object.")
    return value


def _validate_keys(section_name: str, raw: dict[str, Any], allowed: set[str]) -> None:
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ConfigError(
            f"Section '{section_name}' contains unsupported keys: {', '.join(unexpected)}"
        )


def _require_keys(section_name: str, raw: dict[str, Any], required: set[str]) -> None:
    missing = sorted(key for key in required if key not in raw)
    if missing:
        raise ConfigError(
            f"Section '{section_name}' is missing required keys: {', '.join(missing)}"
        )


def _ensure_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"'{name}' must be a list.")
    return value


def _ensure_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{name}' must be a number.") from exc


def _ensure_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"'{name}' must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{name}' must be an integer.") from exc


def _ensure_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{name}' must be a boolean.")
    return value


def _ensure_optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _ensure_float(value, name)


def _ensure_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _ensure_int(value, name)


def _ensure_string_list(value: Any, name: str) -> list[str]:
    items = _ensure_list(value, name)
    result: list[str] = []
    for idx, item in enumerate(items):
        if not isinstance(item, str):
            raise ConfigError(f"'{name}[{idx}]' must be a string.")
        result.append(item)
    return result


def _ensure_number_list(value: Any, name: str, expected_length: int | None = None) -> list[float]:
    items = _ensure_list(value, name)
    if expected_length is not None and len(items) != expected_length:
        raise ConfigError(f"'{name}' must contain exactly {expected_length} values.")
    return [_ensure_float(item, f"{name}[{idx}]") for idx, item in enumerate(items)]


def _ensure_matrix(value: Any, name: str, rows: int) -> list[list[float]]:
    items = _ensure_list(value, name)
    if len(items) != rows:
        raise ConfigError(f"'{name}' must contain exactly {rows} rows.")
    matrix: list[list[float]] = []
    for idx, row in enumerate(items):
        matrix.append(_ensure_number_list(row, f"{name}[{idx}]", expected_length=2))
    return matrix


def _ensure_dict_of_dicts(value: Any, name: str) -> dict[str, dict[str, Any]]:
    raw = _ensure_dict(value, name)
    normalized: dict[str, dict[str, Any]] = {}
    for key, item in raw.items():
        if not isinstance(key, str):
            raise ConfigError(f"Keys in '{name}' must be strings.")
        normalized[key] = _ensure_dict(item, f"{name}.{key}")
    return normalized


@dataclass(frozen=True)
class SimulationConfig:
    gpu_id: int
    force_cpu: bool
    log_level: str
    seed: int
    output_dir: str
    plot_results: bool
    # Write a per-batch trace (trace_v1.jsonl.gz) next to each stage's results.
    trace: bool = False
    # Ask TensorFlow for deterministic ops (slower); use for final evidence runs.
    deterministic: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SimulationConfig":
        if "targets" in raw:
            raise ConfigError(
                "'simulation.targets' is not supported. The simulation flow is fixed to "
                "['estimators', 'resource_managers']."
            )
        allowed = {
            "gpu_id",
            "force_cpu",
            "log_level",
            "seed",
            "output_dir",
            "plot_results",
            "trace",
            "deterministic",
        }
        required: set[str] = set()
        _validate_keys("simulation", raw, allowed)
        _require_keys("simulation", raw, required)
        log_level = str(raw.get("log_level", "INFO")).upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError("'simulation.log_level' must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL.")
        return cls(
            gpu_id=_ensure_int(raw.get("gpu_id", 0), "simulation.gpu_id"),
            force_cpu=_ensure_bool(raw.get("force_cpu", False), "simulation.force_cpu"),
            log_level=log_level,
            seed=_ensure_int(raw.get("seed", 42), "simulation.seed"),
            output_dir=str(raw.get("output_dir", "results")),
            plot_results=_ensure_bool(raw.get("plot_results", True), "simulation.plot_results"),
            trace=_ensure_bool(raw.get("trace", False), "simulation.trace"),
            deterministic=_ensure_bool(raw.get("deterministic", False), "simulation.deterministic"),
        )


@dataclass(frozen=True)
class MonteCarloConfig:
    batch_size: int
    min_batches: int
    max_batches: int
    target_block_errors: int | None
    target_ber: float | None
    stop_policy: str
    confidence_level: float
    min_total_bits: int
    ebno_min: float
    ebno_max: float
    ebno_step: float
    # Distinct batches that must contain bit errors before a point may close on
    # its block-error target or count as resolved. With persistent terminal
    # states errors cluster by batch, so slot counts alone overstate the
    # evidence; 0 keeps the v1 rule.
    min_error_batches: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MonteCarloConfig":
        allowed = {
            "batch_size",
            "min_batches",
            "max_batches",
            "target_block_errors",
            "target_ber",
            "stop_policy",
            "confidence_level",
            "min_total_bits",
            "ebno_min",
            "ebno_max",
            "ebno_step",
            "min_error_batches",
        }
        required = {"batch_size", "min_batches", "max_batches", "ebno_min", "ebno_max", "ebno_step"}
        _validate_keys("monte_carlo", raw, allowed)
        _require_keys("monte_carlo", raw, required)
        batch_size = _ensure_int(raw["batch_size"], "monte_carlo.batch_size")
        min_batches = _ensure_int(raw["min_batches"], "monte_carlo.min_batches")
        max_batches = _ensure_int(raw["max_batches"], "monte_carlo.max_batches")
        if batch_size <= 0:
            raise ConfigError("'monte_carlo.batch_size' must be positive.")
        if min_batches <= 0:
            raise ConfigError("'monte_carlo.min_batches' must be positive.")
        if max_batches < min_batches:
            raise ConfigError("'monte_carlo.max_batches' must be >= 'monte_carlo.min_batches'.")
        ebno_min = _ensure_float(raw["ebno_min"], "monte_carlo.ebno_min")
        ebno_max = _ensure_float(raw["ebno_max"], "monte_carlo.ebno_max")
        ebno_step = _ensure_float(raw["ebno_step"], "monte_carlo.ebno_step")
        if ebno_step <= 0:
            raise ConfigError("'monte_carlo.ebno_step' must be positive.")
        if ebno_max < ebno_min:
            raise ConfigError("'monte_carlo.ebno_max' must be >= 'monte_carlo.ebno_min'.")
        confidence_level = _ensure_float(raw.get("confidence_level", 0.95), "monte_carlo.confidence_level")
        if not (0.0 < confidence_level < 1.0):
            raise ConfigError("'monte_carlo.confidence_level' must be between 0 and 1.")
        stop_policy = str(raw.get("stop_policy", "sweep")).lower()
        if stop_policy not in {"sweep", "threshold"}:
            raise ConfigError("'monte_carlo.stop_policy' must be either 'sweep' or 'threshold'.")
        target_ber = _ensure_optional_float(raw.get("target_ber"), "monte_carlo.target_ber")
        if stop_policy == "threshold" and target_ber is None and raw.get("target_block_errors") is None:
            raise ConfigError(
                "'monte_carlo.stop_policy' is 'threshold' but neither 'target_ber' nor 'target_block_errors' is set."
            )
        return cls(
            batch_size=batch_size,
            min_batches=min_batches,
            max_batches=max_batches,
            target_block_errors=_ensure_optional_int(
                raw.get("target_block_errors", 1000),
                "monte_carlo.target_block_errors",
            ),
            target_ber=target_ber,
            stop_policy=stop_policy,
            confidence_level=confidence_level,
            min_total_bits=_ensure_int(raw.get("min_total_bits", 0), "monte_carlo.min_total_bits"),
            ebno_min=ebno_min,
            ebno_max=ebno_max,
            ebno_step=ebno_step,
            min_error_batches=_ensure_int(raw.get("min_error_batches", 0), "monte_carlo.min_error_batches"),
        )

    @property
    def ebno_db_range(self) -> list[float]:
        values = np.arange(self.ebno_min, self.ebno_max + self.ebno_step, self.ebno_step)
        return values.tolist()


@dataclass(frozen=True)
class EstimatorsConfig:
    enabled: list[str]
    kwargs: dict[str, dict[str, Any]]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EstimatorsConfig":
        allowed = {"enabled", "kwargs"}
        required = {"enabled"}
        _validate_keys("estimators", raw, allowed)
        _require_keys("estimators", raw, required)
        enabled = [item.lower() for item in _ensure_string_list(raw["enabled"], "estimators.enabled")]
        return cls(
            enabled=enabled,
            kwargs=_ensure_dict_of_dicts(raw.get("kwargs", {}), "estimators.kwargs"),
        )


@dataclass(frozen=True)
class ResourceManagersConfig:
    enabled: list[str]
    cnn_model_path: str | None
    drl_model_path: str | None
    num_active_users: int
    kwargs: dict[str, dict[str, Any]]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ResourceManagersConfig":
        allowed = {"enabled", "cnn_model_path", "drl_model_path", "num_active_users", "kwargs"}
        required = {"enabled", "num_active_users"}
        _validate_keys("resource_managers", raw, allowed)
        _require_keys("resource_managers", raw, required)
        enabled = [item.lower() for item in _ensure_string_list(raw["enabled"], "resource_managers.enabled")]
        num_active_users = _ensure_int(raw["num_active_users"], "resource_managers.num_active_users")
        if num_active_users <= 0:
            raise ConfigError("'resource_managers.num_active_users' must be positive.")
        model_path = raw.get("cnn_model_path")
        if model_path is not None and not isinstance(model_path, str):
            raise ConfigError("'resource_managers.cnn_model_path' must be a string or null.")
        drl_model_path = raw.get("drl_model_path")
        if drl_model_path is not None and not isinstance(drl_model_path, str):
            raise ConfigError("'resource_managers.drl_model_path' must be a string or null.")
        return cls(
            enabled=enabled,
            cnn_model_path=model_path,
            drl_model_path=drl_model_path,
            num_active_users=num_active_users,
            kwargs=_ensure_dict_of_dicts(raw.get("kwargs", {}), "resource_managers.kwargs"),
        )


@dataclass(frozen=True)
class SystemConfig:
    carrier_frequency: float
    fft_size: int
    subcarrier_spacing: float
    num_ofdm_symbols: int
    cyclic_prefix_length: int
    pilot_ofdm_symbol_indices: list[int]
    num_bs_ant: int
    num_ut: int
    num_ut_ant: int
    num_bits_per_symbol: int
    coderate: float
    num_decoding_iter: int
    channel_model_type: str
    rician_k_factor: float
    scenario: str
    direction: str
    o2i_model: str
    enable_pathloss: bool
    enable_shadow_fading: bool
    min_ut_velocity: float
    max_ut_velocity: float
    ut_max_tx_power_dbm: float = 23.0
    processing_budget_slots: float = 1.0
    # Ray-traced channel bank (channel_model_type "raytraced"); the "eval" split
    # is used for every reported result, "train" only by the training scripts.
    rt_channel_bank: str = "data/rt_channel_bank.npz"
    rt_bank_split: str = "eval"
    # "hall": persistent per-batch terminal positions and large-scale gains, no
    # per-slot link normalisation (system model v2); "none": v1 behaviour.
    large_scale_model: str = "hall"

    _LOCKED_5G_VALUES = {
        "carrier_frequency": 3.5e9,
        "subcarrier_spacing": 30000.0,
        "num_ofdm_symbols": 14,
        "cyclic_prefix_length": 20,
        "coderate": 0.5,
        "scenario": "umi",
        "direction": "uplink",
        "pilot_ofdm_symbol_indices": [2, 11],
    }

    _VALID_MODULATIONS = {1, 2, 4, 6}

    _VALID_CHANNEL_MODELS = {"tr38901", "rayleigh", "rician", "raytraced", "awgn"}
    _VALID_RT_SPLITS = {"train", "eval"}
    _VALID_LARGE_SCALE = {"hall", "none"}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SystemConfig":
        allowed = {
            "carrier_frequency",
            "fft_size",
            "subcarrier_spacing",
            "num_ofdm_symbols",
            "cyclic_prefix_length",
            "pilot_ofdm_symbol_indices",
            "num_bs_ant",
            "num_ut",
            "num_ut_ant",
            "num_bits_per_symbol",
            "coderate",
            "num_decoding_iter",
            "channel_model_type",
            "rician_k_factor",
            "scenario",
            "direction",
            "o2i_model",
            "enable_pathloss",
            "enable_shadow_fading",
            "min_ut_velocity",
            "max_ut_velocity",
            "ut_max_tx_power_dbm",
            "processing_budget_slots",
            "rt_channel_bank",
            "rt_bank_split",
            "large_scale_model",
        }
        required = {
            "carrier_frequency",
            "fft_size",
            "subcarrier_spacing",
            "num_ofdm_symbols",
            "cyclic_prefix_length",
            "pilot_ofdm_symbol_indices",
            "num_bs_ant",
            "num_ut",
            "num_ut_ant",
            "num_bits_per_symbol",
            "coderate",
            "num_decoding_iter",
            "channel_model_type",
            "scenario",
            "direction",
            "o2i_model",
            "enable_pathloss",
            "enable_shadow_fading",
            "min_ut_velocity",
            "max_ut_velocity",
        }
        _validate_keys("system", raw, allowed)
        _require_keys("system", raw, required)
        pilot_indices = _ensure_list(raw["pilot_ofdm_symbol_indices"], "system.pilot_ofdm_symbol_indices")
        normalized_pilot_indices: list[int] = []
        for idx, item in enumerate(pilot_indices):
            normalized_pilot_indices.append(_ensure_int(item, f"system.pilot_ofdm_symbol_indices[{idx}]"))
        parsed = cls(
            carrier_frequency=_ensure_float(raw["carrier_frequency"], "system.carrier_frequency"),
            fft_size=_ensure_int(raw["fft_size"], "system.fft_size"),
            subcarrier_spacing=_ensure_float(raw["subcarrier_spacing"], "system.subcarrier_spacing"),
            num_ofdm_symbols=_ensure_int(raw["num_ofdm_symbols"], "system.num_ofdm_symbols"),
            cyclic_prefix_length=_ensure_int(raw["cyclic_prefix_length"], "system.cyclic_prefix_length"),
            pilot_ofdm_symbol_indices=normalized_pilot_indices,
            num_bs_ant=_ensure_int(raw["num_bs_ant"], "system.num_bs_ant"),
            num_ut=_ensure_int(raw["num_ut"], "system.num_ut"),
            num_ut_ant=_ensure_int(raw["num_ut_ant"], "system.num_ut_ant"),
            num_bits_per_symbol=_ensure_int(raw["num_bits_per_symbol"], "system.num_bits_per_symbol"),
            coderate=_ensure_float(raw["coderate"], "system.coderate"),
            num_decoding_iter=_ensure_int(raw["num_decoding_iter"], "system.num_decoding_iter"),
            channel_model_type=str(raw["channel_model_type"]).lower(),
            rician_k_factor=_ensure_float(raw.get("rician_k_factor", 1.0), "system.rician_k_factor"),
            scenario=str(raw["scenario"]).lower(),
            direction=str(raw["direction"]).lower(),
            o2i_model=str(raw["o2i_model"]).lower(),
            enable_pathloss=_ensure_bool(raw["enable_pathloss"], "system.enable_pathloss"),
            enable_shadow_fading=_ensure_bool(
                raw["enable_shadow_fading"],
                "system.enable_shadow_fading",
            ),
            min_ut_velocity=_ensure_float(raw["min_ut_velocity"], "system.min_ut_velocity"),
            max_ut_velocity=_ensure_float(raw["max_ut_velocity"], "system.max_ut_velocity"),
            ut_max_tx_power_dbm=_ensure_float(
                raw.get("ut_max_tx_power_dbm", 23.0), "system.ut_max_tx_power_dbm"
            ),
            processing_budget_slots=_ensure_float(
                raw.get("processing_budget_slots", 1.0), "system.processing_budget_slots"
            ),
            rt_channel_bank=str(raw.get("rt_channel_bank", "data/rt_channel_bank.npz")),
            rt_bank_split=str(raw.get("rt_bank_split", "eval")).lower(),
            large_scale_model=str(raw.get("large_scale_model", "hall")).lower(),
        )
        if parsed.large_scale_model not in cls._VALID_LARGE_SCALE:
            raise ConfigError(
                f"'system.large_scale_model' must be one of {sorted(cls._VALID_LARGE_SCALE)} "
                f"(got '{parsed.large_scale_model}')."
            )
        if parsed.rt_bank_split not in cls._VALID_RT_SPLITS:
            raise ConfigError(
                f"'system.rt_bank_split' must be one of {sorted(cls._VALID_RT_SPLITS)} "
                f"(got '{parsed.rt_bank_split}')."
            )
        channel_model_type = str(raw["channel_model_type"]).lower()
        if channel_model_type not in cls._VALID_CHANNEL_MODELS:
            raise ConfigError(
                f"'system.channel_model_type' must be one of {sorted(cls._VALID_CHANNEL_MODELS)} "
                f"(got '{channel_model_type}')."
            )
        if parsed.num_bits_per_symbol not in cls._VALID_MODULATIONS:
            raise ConfigError(
                f"'system.num_bits_per_symbol' must be one of {sorted(cls._VALID_MODULATIONS)} "
                f"(got {parsed.num_bits_per_symbol})."
            )
        parsed._validate_locked_5g_profile()
        return parsed

    def _validate_locked_5g_profile(self) -> None:
        self._assert_locked_float("system.carrier_frequency", self.carrier_frequency)
        self._assert_locked_float("system.subcarrier_spacing", self.subcarrier_spacing)
        self._assert_locked_int("system.num_ofdm_symbols", self.num_ofdm_symbols)
        self._assert_locked_int("system.cyclic_prefix_length", self.cyclic_prefix_length)
        self._assert_locked_float("system.coderate", self.coderate)
        self._assert_locked_str("system.scenario", self.scenario)
        self._assert_locked_str("system.direction", self.direction)
        expected_pilots = self._LOCKED_5G_VALUES["pilot_ofdm_symbol_indices"]
        if list(self.pilot_ofdm_symbol_indices) != list(expected_pilots):
            raise ConfigError(
                f"'system.pilot_ofdm_symbol_indices' must be exactly {expected_pilots} "
                f"for locked 5G profile."
            )

    def _assert_locked_float(self, field_name: str, actual: float) -> None:
        expected = float(self._LOCKED_5G_VALUES[field_name.split(".")[-1]])
        if not np.isclose(actual, expected, rtol=1e-9, atol=1e-9):
            raise ConfigError(
                f"'{field_name}' must be {expected} for locked 5G profile (got {actual})."
            )

    def _assert_locked_int(self, field_name: str, actual: int) -> None:
        expected = int(self._LOCKED_5G_VALUES[field_name.split(".")[-1]])
        if int(actual) != expected:
            raise ConfigError(
                f"'{field_name}' must be {expected} for locked 5G profile (got {actual})."
            )

    def _assert_locked_str(self, field_name: str, actual: str) -> None:
        expected = str(self._LOCKED_5G_VALUES[field_name.split(".")[-1]])
        if str(actual).lower() != expected.lower():
            raise ConfigError(
                f"'{field_name}' must be '{expected}' for locked 5G profile (got '{actual}')."
            )


@dataclass(frozen=True)
class TransceiverConfig:
    tx_height_offset: float
    rx_height: float
    antenna_spacing: float
    tx_pattern: str
    tx_polarization: str
    rx_pattern: str
    rx_polarization: str
    wall_thickness: float
    room_padding: float
    rx_boundary_padding: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TransceiverConfig":
        allowed = {
            "tx_height_offset",
            "rx_height",
            "antenna_spacing",
            "tx_pattern",
            "tx_polarization",
            "rx_pattern",
            "rx_polarization",
            "wall_thickness",
            "room_padding",
            "rx_boundary_padding",
        }
        required = allowed
        _validate_keys("transceiver", raw, allowed)
        _require_keys("transceiver", raw, required)
        parsed = cls(
            tx_height_offset=_ensure_float(raw["tx_height_offset"], "transceiver.tx_height_offset"),
            rx_height=_ensure_float(raw["rx_height"], "transceiver.rx_height"),
            antenna_spacing=_ensure_float(raw["antenna_spacing"], "transceiver.antenna_spacing"),
            tx_pattern=str(raw["tx_pattern"]),
            tx_polarization=str(raw["tx_polarization"]),
            rx_pattern=str(raw["rx_pattern"]),
            rx_polarization=str(raw["rx_polarization"]),
            wall_thickness=_ensure_float(raw["wall_thickness"], "transceiver.wall_thickness"),
            room_padding=_ensure_float(raw["room_padding"], "transceiver.room_padding"),
            rx_boundary_padding=_ensure_float(
                raw["rx_boundary_padding"],
                "transceiver.rx_boundary_padding",
            ),
        )
        if parsed.tx_pattern.lower() != "tr38901":
            raise ConfigError(
                "'transceiver.tx_pattern' must be 'tr38901' for locked 5G profile."
            )
        return parsed


@dataclass(frozen=True)
class MaterialConfig:
    name: str
    relative_permittivity: float
    conductivity: float

    @classmethod
    def from_dict(cls, section_name: str, raw: dict[str, Any]) -> "MaterialConfig":
        allowed = {"name", "relative_permittivity", "conductivity"}
        required = allowed
        _validate_keys(section_name, raw, allowed)
        _require_keys(section_name, raw, required)
        return cls(
            name=str(raw["name"]),
            relative_permittivity=_ensure_float(
                raw["relative_permittivity"],
                f"{section_name}.relative_permittivity",
            ),
            conductivity=_ensure_float(raw["conductivity"], f"{section_name}.conductivity"),
        )


@dataclass(frozen=True)
class FactoryScenarioConfig:
    room_dimensions: list[float]
    num_machines: int
    machine_size_range: list[list[float]]
    materials: dict[str, MaterialConfig]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FactoryScenarioConfig":
        allowed = {"room_dimensions", "num_machines", "machine_size_range", "materials"}
        required = allowed
        _validate_keys("factory_scenario", raw, allowed)
        _require_keys("factory_scenario", raw, required)
        materials_raw = _ensure_dict(raw["materials"], "factory_scenario.materials")
        _validate_keys("factory_scenario.materials", materials_raw, {"metal", "concrete"})
        _require_keys("factory_scenario.materials", materials_raw, {"metal", "concrete"})
        return cls(
            room_dimensions=_ensure_number_list(
                raw["room_dimensions"],
                "factory_scenario.room_dimensions",
                expected_length=3,
            ),
            num_machines=_ensure_int(raw["num_machines"], "factory_scenario.num_machines"),
            machine_size_range=_ensure_matrix(
                raw["machine_size_range"],
                "factory_scenario.machine_size_range",
                rows=3,
            ),
            materials={
                name: MaterialConfig.from_dict(f"factory_scenario.materials.{name}", _ensure_dict(value, f"factory_scenario.materials.{name}"))
                for name, value in materials_raw.items()
            },
        )


@dataclass(frozen=True)
class RayTracingConfig:
    max_depth: int
    samples_per_src: int
    max_paths: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RayTracingConfig":
        allowed = {"max_depth", "samples_per_src", "max_paths"}
        required = allowed
        _validate_keys("ray_tracing", raw, allowed)
        _require_keys("ray_tracing", raw, required)
        return cls(
            max_depth=_ensure_int(raw["max_depth"], "ray_tracing.max_depth"),
            samples_per_src=_ensure_int(raw["samples_per_src"], "ray_tracing.samples_per_src"),
            max_paths=_ensure_int(raw["max_paths"], "ray_tracing.max_paths"),
        )


@dataclass(frozen=True)
class Factory6GConfig:
    simulation: SimulationConfig
    monte_carlo: MonteCarloConfig
    estimators: EstimatorsConfig
    resource_managers: ResourceManagersConfig
    system: SystemConfig
    transceiver: TransceiverConfig
    factory_scenario: FactoryScenarioConfig
    ray_tracing: RayTracingConfig
    # Optional Industry 5.0 scenario layer (factory6g.sim.industry); None keeps the thesis scenario.
    industry: Any = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Factory6GConfig":
        allowed = {
            "simulation",
            "monte_carlo",
            "estimators",
            "resource_managers",
            "system",
            "transceiver",
            "factory_scenario",
            "ray_tracing",
            "jidd_scma",   # optional: Dr. Athirah's JIDD-SCMA simulation config
            "industry",    # optional: Industry 5.0 scenario layer
        }
        required = allowed - {"jidd_scma", "industry"}
        _validate_keys("root", raw, allowed)
        _require_keys("root", raw, required)
        return cls(
            simulation=SimulationConfig.from_dict(_ensure_dict(raw["simulation"], "simulation")),
            monte_carlo=MonteCarloConfig.from_dict(_ensure_dict(raw["monte_carlo"], "monte_carlo")),
            estimators=EstimatorsConfig.from_dict(_ensure_dict(raw["estimators"], "estimators")),
            resource_managers=ResourceManagersConfig.from_dict(
                _ensure_dict(raw["resource_managers"], "resource_managers")
            ),
            system=SystemConfig.from_dict(_ensure_dict(raw["system"], "system")),
            transceiver=TransceiverConfig.from_dict(_ensure_dict(raw["transceiver"], "transceiver")),
            factory_scenario=FactoryScenarioConfig.from_dict(
                _ensure_dict(raw["factory_scenario"], "factory_scenario")
            ),
            ray_tracing=RayTracingConfig.from_dict(_ensure_dict(raw["ray_tracing"], "ray_tracing")),
            industry=_industry_from(raw.get("industry")),
        )

    @property
    def system_runtime_config(self) -> dict[str, Any]:
        runtime = asdict(self.system)
        runtime.update(asdict(self.transceiver))
        return runtime

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _industry_from(raw: Any):
    if raw is None:
        return None
    from factory6g.sim.industry import IndustryConfig

    try:
        return IndustryConfig.from_dict(_ensure_dict(raw, "industry"))
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def load_config(config_path: str | Path = "config/config.json") -> Factory6GConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw_config = json.loads(_strip_json_comments(handle.read()))
    return Factory6GConfig.from_dict(_ensure_dict(raw_config, "root"))
