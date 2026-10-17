from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from factory6g.sim.config import Factory6GConfig
from factory6g.sim.output import write_overview_plots, write_stage_outputs, write_summary_outputs
from factory6g.sim.run_context import build_run_dir, create_run_context, generate_run_id
from factory6g.sim.stages.common import fmt_elapsed
from factory6g.sim.stages.coupled import run_coupled_stage
from factory6g.sim.stages.estimators import run_estimator_stage
from factory6g.sim.stages.jidd_scma import run_jidd_scma_stage
from factory6g.sim.stages.resource_managers import run_resource_manager_stage

_MODULATION_DISPLAY = {2: "QPSK", 4: "16-QAM", 6: "64-QAM", 1: "BPSK"}
_FACTORY_SIZE_DISPLAY = {"s": "Small", "m": "Medium", "l": "Large", "apple": "Apple Factory"}
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


def _load_completed_stage(stage_dir: Path) -> dict[str, Any] | None:
    """Return saved stage result dict if the stage already finished, else None."""
    result_path = stage_dir / "stage_results_v2.json"
    if not result_path.exists():
        return None
    with result_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def run_jidd_scma_flow(
    config: Factory6GConfig,
    *,
    run_id: str | None = None,
    run_dir: Path | None = None,
    raw_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run only the JIDD-SCMA stage and write results to run_dir/jidd_scma/."""
    if run_id is None and run_dir is None:
        run_id, run_dir = create_run_context(
            config.simulation.output_dir,
            run_id=run_id,
        )
    elif run_id is not None and run_dir is None:
        run_dir = build_run_dir(config.simulation.output_dir, run_id)
    elif run_id is None and run_dir is not None:
        run_dir = Path(run_dir)
        run_id = generate_run_id()
    else:
        run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  JIDD-SCMA Simulation (Dr. Athirah Mohd Ramly, UKM)")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print("-" * 70)

    flow_start = time.perf_counter()

    stage_dir = run_dir / "jidd_scma"
    saved = _load_completed_stage(stage_dir)
    if saved is not None:
        print("[checkpoint] JIDD-SCMA stage already complete — skipping.")
        jidd_result = {
            "stage": "jidd_scma",
            "ebno_db_range": saved["ebno_db_range"],
            "methods": saved["methods"],
            "runtime_totals_sec": saved["runtime_totals_sec"],
        }
        stage_paths = {
            "dir": str(stage_dir),
            "json": str(stage_dir / "stage_results_v2.json"),
            "csv": str(stage_dir / "stage_results_v2.csv"),
        }
    else:
        _t = time.perf_counter()
        jidd_result = run_jidd_scma_stage(raw_config or {})
        print(f"[jidd_scma] Done in {fmt_elapsed(time.perf_counter() - _t)}")
        stage_paths = write_stage_outputs(
            run_id=run_id,
            stage_name="jidd_scma",
            stage_result=jidd_result,
            stage_dir=stage_dir,
            config_snapshot=raw_config or {},
            confidence_level=config.monte_carlo.confidence_level,
            plot_results=config.simulation.plot_results,
        )

    summary = write_summary_outputs(
        run_id=run_id,
        run_dir=run_dir,
        stage_order=["jidd_scma"],
        stage_paths={"jidd_scma": stage_paths},
        stage_payloads={"jidd_scma": {
            "methods": jidd_result["methods"],
            "runtime_totals_sec": jidd_result["runtime_totals_sec"],
        }},
    )

    print(f"\nSimulation complete. Total time: {fmt_elapsed(time.perf_counter() - flow_start)}")
    return {
        "schema_version": "2.0",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stage_order": ["jidd_scma"],
        "stage_paths": {"jidd_scma": stage_paths},
        "summary": summary["summary_payload"],
        "summary_paths": {
            "json": summary["summary_json"],
            "csv": summary["summary_csv"],
        },
    }


def run_simulation_flow(
    config: Factory6GConfig,
    *,
    run_id: str | None = None,
    run_dir: Path | None = None,
    modulations: list[tuple[str, int]] | None = None,
    channels: list[str] | None = None,
    factory_sizes: list[str] | None = None,
    coupled: bool = False,
) -> dict[str, Any]:
    if run_id is None and run_dir is None:
        run_id, run_dir = create_run_context(
            config.simulation.output_dir,
            run_id=run_id,
        )
    elif run_id is not None and run_dir is None:
        run_dir = build_run_dir(config.simulation.output_dir, run_id)
    elif run_id is None and run_dir is not None:
        run_dir = Path(run_dir)
        suffix = "_simulation"
        if run_dir.name.endswith(suffix) and len(run_dir.name) > len(suffix):
            run_id = run_dir.name[: -len(suffix)]
        else:
            run_id = generate_run_id()
    else:
        run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Defaults
    if not modulations:
        modulations = [("default", config.system.num_bits_per_symbol)]
    if not channels:
        channels = [config.system.channel_model_type]
    if not factory_sizes:
        factory_sizes = ["s"]

    # The coupled stage crosses the estimator and scheduler rosters itself, so
    # the two separate stages are not run alongside it.
    run_coupled = bool(coupled)
    run_estimators = bool(config.estimators.enabled) and not run_coupled
    run_rms = bool(config.resource_managers.enabled) and not run_coupled

    active_stages = []
    if run_estimators:
        active_stages.append("estimators")
    if run_rms:
        active_stages.append("resource_managers")
    if run_coupled:
        active_stages.append("coupled")

    multi_size = len(factory_sizes) > 1
    multi_mod = len(modulations) > 1
    multi_ch = len(channels) > 1

    mod_display_names = [_MODULATION_DISPLAY.get(bits, f"{bits}bps") for _, bits in modulations]
    size_display_names = [_FACTORY_SIZE_DISPLAY.get(s, s.upper()) for s in factory_sizes]

    print("=" * 70)
    print("  6G Smart Factory Simulation Flow")
    print("=" * 70)
    print(f"Stages: {' -> '.join(active_stages) if active_stages else '(none)'}")
    if run_estimators:
        print(f"Estimators: {config.estimators.enabled}")
    if run_rms or run_coupled:
        print(f"Resource managers: {config.resource_managers.enabled}")
    if run_coupled:
        print(f"Coupled estimators: {config.estimators.enabled}")
    print(f"Factory sizes: {' | '.join(size_display_names)}")
    print(f"Modulations: {' | '.join(mod_display_names)}")
    print(f"Channels: {' | '.join(channels)}")
    print(f"Scenario: {config.system.scenario}")
    print(f"Eb/No Range: {config.monte_carlo.ebno_db_range}")
    print(f"Run directory: {run_dir}")
    print("-" * 70)

    flow_start = time.perf_counter()

    combined_plot_entries: list[tuple[str, str, dict[str, Any], list[float]]] = []
    all_stage_payloads: dict[str, dict[str, Any]] = {}
    all_stage_paths: dict[str, dict[str, str]] = {}

    for size_label in factory_sizes:
        preset = _FACTORY_SIZE_PRESETS[size_label]
        size_display = _FACTORY_SIZE_DISPLAY.get(size_label, size_label.upper())
        size_config = dataclasses.replace(
            config,
            system=dataclasses.replace(config.system, num_ut=preset["num_ut"]),
            factory_scenario=dataclasses.replace(
                config.factory_scenario,
                room_dimensions=preset["room_dimensions"],
                num_machines=preset["num_machines"],
                machine_size_range=preset["machine_size_range"],
            ),
            resource_managers=dataclasses.replace(
                config.resource_managers,
                num_active_users=min(config.resource_managers.num_active_users, preset["num_ut"]),
            ),
        )

        for mod_label, bits_per_symbol in modulations:
            mod_display = _MODULATION_DISPLAY.get(bits_per_symbol, f"{bits_per_symbol}bps")
            mod_config = dataclasses.replace(
                size_config,
                system=dataclasses.replace(size_config.system, num_bits_per_symbol=bits_per_symbol),
            )

            for channel in channels:
                combo_config = dataclasses.replace(
                    mod_config,
                    system=dataclasses.replace(mod_config.system, channel_model_type=channel),
                )

                # Subdir naming: join non-singleton dimension labels
                parts = []
                if multi_size:
                    parts.append(size_label)
                if multi_mod:
                    parts.append(mod_label)
                if multi_ch:
                    parts.append(channel)
                combo_key = "_".join(parts) if parts else ""

                combo_dir = run_dir / combo_key if combo_key else run_dir
                combo_dir.mkdir(parents=True, exist_ok=True)

                # Label for combined plot
                label_parts = []
                if multi_size:
                    label_parts.append(size_display)
                if multi_mod:
                    label_parts.append(mod_display)
                if multi_ch:
                    label_parts.append(channel)
                combo_label = " – ".join(label_parts) if label_parts else size_display

                print(f"\n{'=' * 70}")
                print(f"  {combo_label}")
                print(f"{'=' * 70}")

                stage_payloads: dict[str, dict[str, Any]] = {}
                stage_paths: dict[str, dict[str, str]] = {}

                if run_estimators:
                    estimator_stage_dir = combo_dir / "estimators"
                    saved_estimators = _load_completed_stage(estimator_stage_dir)
                    if saved_estimators is not None:
                        print("[checkpoint] Estimator stage already complete — skipping.")
                        estimator_result = {
                            "stage": "estimators",
                            "ebno_db_range": saved_estimators["ebno_db_range"],
                            "methods": saved_estimators["methods"],
                            "runtime_totals_sec": saved_estimators["runtime_totals_sec"],
                        }
                        estimator_paths = {
                            "dir": str(estimator_stage_dir),
                            "json": str(estimator_stage_dir / "stage_results_v2.json"),
                            "csv": str(estimator_stage_dir / "stage_results_v2.csv"),
                        }
                    else:
                        print("[estimators] Starting...")
                        _t = time.perf_counter()
                        estimator_result = run_estimator_stage(combo_config, checkpoint_dir=estimator_stage_dir)
                        print(f"[estimators] Done in {fmt_elapsed(time.perf_counter() - _t)}")
                        estimator_paths = write_stage_outputs(
                            run_id=run_id,
                            stage_name="estimators",
                            stage_result=estimator_result,
                            stage_dir=estimator_stage_dir,
                            config_snapshot=combo_config.to_dict(),
                            confidence_level=combo_config.monte_carlo.confidence_level,
                            plot_results=combo_config.simulation.plot_results,
                        )
                    stage_payloads["estimators"] = {
                        "methods": estimator_result["methods"],
                        "runtime_totals_sec": estimator_result["runtime_totals_sec"],
                    }
                    stage_paths["estimators"] = estimator_paths
                    combined_plot_entries.append((
                        combo_label, "estimators",
                        estimator_result["methods"],
                        estimator_result["ebno_db_range"],
                        estimator_result["runtime_totals_sec"],
                    ))

                if run_rms:
                    rm_stage_dir = combo_dir / "resource_managers"
                    saved_rm = _load_completed_stage(rm_stage_dir)
                    if saved_rm is not None:
                        print("[checkpoint] Resource-manager stage already complete — skipping.")
                        rm_result = {
                            "stage": "resource_managers",
                            "ebno_db_range": saved_rm["ebno_db_range"],
                            "methods": saved_rm["methods"],
                            "runtime_totals_sec": saved_rm["runtime_totals_sec"],
                        }
                        rm_paths = {
                            "dir": str(rm_stage_dir),
                            "json": str(rm_stage_dir / "stage_results_v2.json"),
                            "csv": str(rm_stage_dir / "stage_results_v2.csv"),
                        }
                    else:
                        print("[resource_managers] Starting...")
                        _t = time.perf_counter()
                        rm_result = run_resource_manager_stage(combo_config, checkpoint_dir=rm_stage_dir)
                        print(f"[resource_managers] Done in {fmt_elapsed(time.perf_counter() - _t)}")
                        rm_paths = write_stage_outputs(
                            run_id=run_id,
                            stage_name="resource_managers",
                            stage_result=rm_result,
                            stage_dir=rm_stage_dir,
                            config_snapshot=combo_config.to_dict(),
                            confidence_level=combo_config.monte_carlo.confidence_level,
                            plot_results=combo_config.simulation.plot_results,
                        )
                    stage_payloads["resource_managers"] = {
                        "methods": rm_result["methods"],
                        "runtime_totals_sec": rm_result["runtime_totals_sec"],
                    }
                    stage_paths["resource_managers"] = rm_paths
                    combined_plot_entries.append((
                        combo_label, "resource_managers",
                        rm_result["methods"],
                        rm_result["ebno_db_range"],
                        rm_result["runtime_totals_sec"],
                    ))

                if run_coupled:
                    coupled_stage_dir = combo_dir / "coupled"
                    saved_coupled = _load_completed_stage(coupled_stage_dir)
                    if saved_coupled is not None:
                        print("[checkpoint] Coupled stage already complete — skipping.")
                        coupled_result = {
                            "stage": "coupled",
                            "ebno_db_range": saved_coupled["ebno_db_range"],
                            "methods": saved_coupled["methods"],
                            "runtime_totals_sec": saved_coupled["runtime_totals_sec"],
                        }
                        coupled_paths = {
                            "dir": str(coupled_stage_dir),
                            "json": str(coupled_stage_dir / "stage_results_v2.json"),
                            "csv": str(coupled_stage_dir / "stage_results_v2.csv"),
                        }
                    else:
                        print("[coupled] Starting...")
                        _t = time.perf_counter()
                        coupled_result = run_coupled_stage(combo_config, checkpoint_dir=coupled_stage_dir)
                        print(f"[coupled] Done in {fmt_elapsed(time.perf_counter() - _t)}")
                        # 24+ methods per panel would be unreadable in the default stage plots.
                        coupled_paths = write_stage_outputs(
                            run_id=run_id,
                            stage_name="coupled",
                            stage_result=coupled_result,
                            stage_dir=coupled_stage_dir,
                            config_snapshot=combo_config.to_dict(),
                            confidence_level=combo_config.monte_carlo.confidence_level,
                            plot_results=False,
                        )
                    stage_payloads["coupled"] = {
                        "methods": coupled_result["methods"],
                        "runtime_totals_sec": coupled_result["runtime_totals_sec"],
                    }
                    stage_paths["coupled"] = coupled_paths

                # Merge into top-level dicts keyed by "{combo_key}/{stage}" or just "{stage}"
                for stage_key, payload in stage_payloads.items():
                    dict_key = f"{combo_key}/{stage_key}" if combo_key else stage_key
                    all_stage_payloads[dict_key] = payload
                for stage_key, paths in stage_paths.items():
                    dict_key = f"{combo_key}/{stage_key}" if combo_key else stage_key
                    all_stage_paths[dict_key] = paths

    # Overview plots when multiple combinations exist
    if config.simulation.plot_results and len(combined_plot_entries) > 1:
        dim_labels = []
        if multi_size:
            dim_labels.append("Factory Sizes")
        if multi_ch:
            dim_labels.append("Channels")
        if multi_mod:
            dim_labels.append("Modulations")
        title_prefix = " & ".join(dim_labels) if dim_labels else "Overview"
        write_overview_plots(run_dir, combined_plot_entries, title_prefix=title_prefix)

    print(f"\nSimulation complete. Total time: {fmt_elapsed(time.perf_counter() - flow_start)}")
    summary = write_summary_outputs(
        run_id=run_id,
        run_dir=run_dir,
        stage_order=list(all_stage_paths.keys()),
        stage_paths=all_stage_paths,
        stage_payloads=all_stage_payloads,
    )

    return {
        "schema_version": "2.0",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stage_order": active_stages,
        "stage_paths": all_stage_paths,
        "summary": summary["summary_payload"],
        "summary_paths": {
            "json": summary["summary_json"],
            "csv": summary["summary_csv"],
        },
    }
