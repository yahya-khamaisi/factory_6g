#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import logging
import time
import random
import re
import sys
import warnings
from pathlib import Path
from typing import TextIO

from factory6g.sim.config import ConfigError, load_config
from factory6g.sim.env import configure_env, setup_gpu
from factory6g.sim.run_context import create_run_context


class _FilteredTeeStream:
    """Mirror stream output to console + log file while filtering noisy lines."""

    _NOISY_PATTERNS = [
        re.compile(r".*Allocation.*exceeds.*free system memory.*"),
        re.compile(r".*external/local_xla/xla/tsl/framework/cpu_allocator_impl.*"),
        re.compile(r".*No supported GPU was found.*"),
        re.compile(r".*Matplotlib created a temporary cache directory.*"),
        re.compile(r".*Matplotlib is building the font cache; this may take a moment.*"),
        re.compile(r".*\.matplotlib is not a writable directory.*"),
    ]

    def __init__(self, console_stream: TextIO, log_stream: TextIO):
        self.console_stream = console_stream
        self.log_stream = log_stream

    def write(self, message):
        if not message:
            return 0
        if any(pattern.search(message) for pattern in self._NOISY_PATTERNS):
            return len(message)
        self.console_stream.write(message)
        self.log_stream.write(message)
        return len(message)

    def flush(self):
        self.console_stream.flush()
        self.log_stream.flush()

    @property
    def encoding(self):
        return getattr(self.console_stream, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self.console_stream, "errors", "strict")

    def isatty(self):
        return bool(getattr(self.console_stream, "isatty", lambda: False)())

    def fileno(self):
        return self.console_stream.fileno()

    def __getattr__(self, name: str):
        return getattr(self.console_stream, name)


def _configure_root_logging(level: int, console_stream: TextIO, log_path: Path) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler(console_stream)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)


_MODULATION_MAP = {"low": 2, "mid": 4, "high": 6}
_MODULATION_LABEL_MAP = {2: "qpsk", 4: "16qam", 6: "64qam", 1: "bpsk"}
_VALID_CHANNELS = {"tr38901", "rayleigh", "rician", "raytraced", "awgn"}
_FACTORY_SIZE_PRESETS = {
    "s": {
        "room_dimensions": [15.0, 15.0, 5.0],
        "num_machines": 5,
        "machine_size_range": [[0.5, 2.0], [0.5, 2.0], [0.5, 1.5]],
        "num_ut": 4,
    },
    "m": {
        "room_dimensions": [25.0, 25.0, 6.0],
        "num_machines": 10,
        "machine_size_range": [[1.0, 3.0], [1.0, 3.0], [1.0, 2.5]],
        "num_ut": 8,
    },
    "l": {
        "room_dimensions": [40.0, 40.0, 8.0],
        "num_machines": 20,
        "machine_size_range": [[1.5, 4.0], [1.5, 4.0], [1.0, 3.0]],
        "num_ut": 16,
    },
    "apple": {
        # Consumer electronics precision assembly hall (iPhone-style)
        # 60×35m floor, 8m ceiling — rectangular assembly line layout
        # Dense compact workstations: SMT placers, robotic arms, test stations
        # num_ut=8: fft_size(128) must be divisible by num_ut for Kronecker pilots
        "room_dimensions": [60.0, 35.0, 8.0],
        "num_machines": 22,
        "machine_size_range": [[0.8, 2.5], [0.8, 2.0], [1.0, 2.5]],
        "num_ut": 8,
    },
}


def _parse_modulation_list(raw: str) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in _MODULATION_MAP:
            raise ValueError(
                f"Unknown modulation level '{token}'. Choose from: {', '.join(sorted(_MODULATION_MAP))}."
            )
        result.append((token, _MODULATION_MAP[token]))
    if not result:
        raise ValueError("--modulation requires at least one level.")
    return result


def _parse_channel_list(raw: str) -> list[str]:
    result: list[str] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in _VALID_CHANNELS:
            raise ValueError(
                f"Unknown channel type '{token}'. Choose from: {', '.join(sorted(_VALID_CHANNELS))}."
            )
        result.append(token)
    if not result:
        raise ValueError("--channel requires at least one type.")
    return result


def _parse_factory_size_list(raw: str) -> list[str]:
    result: list[str] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in _FACTORY_SIZE_PRESETS:
            raise ValueError(
                f"Unknown factory size '{token}'. Choose from: {', '.join(sorted(_FACTORY_SIZE_PRESETS))}."
            )
        result.append(token)
    if not result:
        raise ValueError("--factory-size requires at least one size.")
    return result


def _build_run_suffix(config, args: argparse.Namespace) -> str:
    requested_stages = []
    if getattr(args, "stage", None) is not None:
        requested_stages = [s.strip().lower() for s in args.stage.split(",") if s.strip()]

    if requested_stages == ["jidd_scma"]:
        return "jidd_scma"

    all_methods = list(config.estimators.enabled) + list(config.resource_managers.enabled)
    methods_part = "_".join(all_methods) if all_methods else "run"
    if "coupled" in requested_stages:
        methods_part = "coupled_" + methods_part
    if "jidd_scma" in requested_stages:
        methods_part = "jidd_scma_" + methods_part if methods_part != "run" else "jidd_scma"

    channel_labels = []
    for ch in args.channel_list:
        channel_labels.append(config.system.scenario if ch == "tr38901" else ch)
    channel_part = "_".join(channel_labels)

    mod_labels = [_MODULATION_LABEL_MAP.get(bits, f"{bits}bps") for _, bits in args.modulation_list]
    modulation_part = "_".join(mod_labels)

    size_part = "_".join(args.factory_size_list)

    return f"{methods_part}_{channel_part}_{modulation_part}_{size_part}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Factory6G simulations.")
    parser.add_argument("--config", default="config/config.json", help="Path to the simulation config JSON file.")
    parser.add_argument(
        "--resume",
        metavar="RUN_DIR",
        default=None,
        help="Resume an interrupted run from an existing run directory (e.g. results/20260315_092517_simulation).",
    )
    parser.add_argument(
        "--estimators",
        metavar="METHODS",
        default=None,
        help="Comma-separated estimator methods to run, e.g. --estimators ls,pso. Overrides config. Skips resource-manager stage unless --resource-managers is also given.",
    )
    parser.add_argument(
        "--resource-managers",
        metavar="METHODS",
        default=None,
        help="Comma-separated resource-manager methods to run, e.g. --resource-managers wmmse,drl. Overrides config. Skips estimator stage unless --estimators is also given. Default (when omitted): max_throughput.",
    )
    parser.add_argument(
        "--modulation",
        metavar="LEVELS",
        default="low",
        help="Comma-separated modulation levels: low=QPSK(2), mid=16-QAM(4), high=64-QAM(6). Default: low.",
    )
    parser.add_argument(
        "--channel",
        metavar="TYPES",
        default=None,
        help="Comma-separated channel types: rayleigh, rician, tr38901, raytraced, awgn. Default: from config.",
    )
    parser.add_argument(
        "--factory-size",
        metavar="SIZES",
        default="s",
        help="Comma-separated factory sizes: s (small), m (medium), l (large). Default: s.",
    )
    parser.add_argument(
        "--stage",
        metavar="STAGES",
        default=None,
        help=(
            "Comma-separated stages to run: estimators, resource_managers, coupled, jidd_scma. "
            "'coupled' crosses every --estimators method with every --resource-managers method. "
            "Use 'jidd_scma' to run Dr. Athirah's SCMA-OFDM JIDD simulation. "
            "Default: determined by --estimators / --resource-managers flags."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle: TextIO | None = None

    try:
        args.modulation_list = _parse_modulation_list(args.modulation)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    try:
        config = load_config(args.config)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"Error loading configuration: {exc}")
        return 1

    # Load raw config dict (for sections not modelled by Factory6GConfig, e.g. jidd_scma)
    import json as _json
    try:
        with open(args.config, encoding="utf-8") as _fh:
            raw_config: dict = _json.load(_fh)
    except Exception:
        raw_config = {}

    # CLI method overrides: --estimators / --resource-managers filter which methods run.
    # If only one flag is given, the other stage is skipped (enabled=[]).
    if args.estimators is not None or args.resource_managers is not None:
        est_enabled = (
            [m.strip().lower() for m in args.estimators.split(",") if m.strip()]
            if args.estimators is not None
            else []
        )
        rm_enabled = (
            [m.strip().lower() for m in args.resource_managers.split(",") if m.strip()]
            if args.resource_managers is not None
            else []
        )
        config = dataclasses.replace(
            config,
            estimators=dataclasses.replace(config.estimators, enabled=est_enabled),
            resource_managers=dataclasses.replace(config.resource_managers, enabled=rm_enabled),
        )
    else:
        # No CLI overrides — default resource managers to max_throughput
        config = dataclasses.replace(
            config,
            resource_managers=dataclasses.replace(
                config.resource_managers, enabled=["max_throughput"]
            ),
        )

    # Parse channel list; default to channel from config if not specified
    if args.channel is not None:
        try:
            args.channel_list = _parse_channel_list(args.channel)
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1
    else:
        args.channel_list = [config.system.channel_model_type]

    try:
        args.factory_size_list = _parse_factory_size_list(args.factory_size)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    try:
        sim_config = config.simulation
        if args.resume:
            run_dir = Path(args.resume)
            # Extract run_id as the timestamp prefix (everything before the first _ after date)
            parts = run_dir.name.split("_")
            run_id = "_".join(parts[:2]) if len(parts) >= 2 else run_dir.name
            log_mode = "a"
        else:
            run_id, run_dir = create_run_context(
                sim_config.output_dir,
                suffix=_build_run_suffix(config, args),
            )
            log_mode = "w"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "simulation.log"

        try:
            log_handle = log_path.open(log_mode, encoding="utf-8")
        except OSError as exc:
            original_stderr.write(f"Error opening log file '{log_path}': {exc}\n")
            original_stderr.flush()
            return 1

        sys.stdout = _FilteredTeeStream(original_stdout, log_handle)
        sys.stderr = _FilteredTeeStream(original_stderr, log_handle)
        print(f"Log file: {log_path.resolve()}")

        configure_env(force_cpu=sim_config.force_cpu, gpu_num=sim_config.gpu_id)

        import numpy as np
        import sionna.phy
        import tensorflow as tf

        from factory6g.sim.flow import run_jidd_scma_flow, run_simulation_flow

        level = getattr(logging, sim_config.log_level, logging.INFO)
        _configure_root_logging(level, original_stdout, log_path)
        tf.get_logger().setLevel("ERROR" if level > logging.DEBUG else "INFO")
        warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
        logging.getLogger("tensorflow").setLevel(logging.ERROR if level > logging.DEBUG else logging.INFO)
        logging.getLogger("absl").setLevel(logging.ERROR if level > logging.DEBUG else logging.INFO)

        if not sim_config.force_cpu:
            setup_gpu(sim_config.gpu_id, force_cpu=sim_config.force_cpu)

        random.seed(sim_config.seed)
        np.random.seed(sim_config.seed)
        tf.random.set_seed(sim_config.seed)
        sionna.phy.config.seed = sim_config.seed

        from factory6g.sim.stages.common import fmt_elapsed

        # Parse --stage flag to determine which simulation flows to run
        requested_stages: list[str] = []
        if args.stage is not None:
            requested_stages = [s.strip().lower() for s in args.stage.split(",") if s.strip()]

        run_jidd = "jidd_scma" in requested_stages
        run_coupled = "coupled" in requested_stages
        run_standard = (
            "estimators" in requested_stages
            or "resource_managers" in requested_stages
            or run_coupled
            or not requested_stages  # default: run standard flow unless only jidd_scma
        ) and not (run_jidd and len(requested_stages) == 1)

        print(f"Loaded configuration for scenario: {config.system.scenario}")
        _wall_start = time.perf_counter()

        if run_jidd:
            run_jidd_scma_flow(config, run_id=run_id, run_dir=run_dir, raw_config=raw_config)

        if run_standard:
            run_simulation_flow(config, run_id=run_id, run_dir=run_dir, modulations=args.modulation_list, channels=args.channel_list, factory_sizes=args.factory_size_list, coupled=run_coupled)

        print(f"Wall-clock time (incl. setup): {fmt_elapsed(time.perf_counter() - _wall_start)}")
        return 0
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_handle is not None:
            log_handle.flush()
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
