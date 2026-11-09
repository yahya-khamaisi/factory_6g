# Factory6G

Factory6G is a 6G and Beyond-5G smart-factory simulation project focused on
physical-layer reliability and AI/ML-assisted resource management. The current
runtime uses Sionna/TensorFlow components to run Monte Carlo Eb/No sweeps for
two main benchmark families:

- channel estimator comparison
- resource manager comparison

The project is Docker-first. Run scripts, simulations, training utilities, and
tests inside the Docker container so the TensorFlow, Sionna, Dr.Jit, and
plotting environment stays consistent.

## Project Map

| Path | Purpose |
|---|---|
| `pyproject.toml` | Packaging metadata. `pip install -e .` exposes the `factory6g` package plus the `factory6g-run` / `factory6g-train` / `factory6g-visualize` console entrypoints. |
| `src/factory6g/cli/run.py` | CLI entrypoint for simulation runs (`python -m factory6g.cli.run`). Loads `config/config.json`, creates a timestamped run directory, configures the environment, and dispatches the simulation flow. |
| `src/factory6g/cli/train.py` | NeuralChannelEstimator training entrypoint (`python -m factory6g.cli.train`). |
| `src/factory6g/cli/visualize.py` | Factory ray-tracing visualization entrypoint (`python -m factory6g.cli.visualize`). |
| `config/config.json` | Main simulation configuration: Monte Carlo policy, enabled estimators/resource managers, system numerology, factory geometry, and output settings. |
| `config/factory_size_profiles.json` | Factory size profile definitions (S/M/L) for dataset generation. |
| `docs/ARCHITECTURE.md` | Main architecture reference for runtime flow, stage data flow, interfaces, and output schema. Start there when reading the implementation deeply. |
| `docs/CONTEXT.md` | Repo-wide working context and lean-git zone policy. |
| `docs/assets/system_design/` | Static system-design images. |
| `src/factory6g/sim/` | Simulation orchestration, config loading, run context creation, stage execution, checkpointing, output writing, and plotting. |
| `src/factory6g/models/` | PHY model composition plus resource-manager implementations, learned CNN/DRL policy wrappers, and scheduling directives. |
| `src/factory6g/components/` | Signal-processing building blocks: antenna arrays, channel model, transmitter, receiver, and custom channel estimators. |
| `src/factory6g/athirah/` | Python port of the MATLAB polar-coded SCMA-OFDM (JIDD-SCMA) reference. |
| `scripts/` | Dataset generation, model training, reporting, visualization, and maintenance utilities. Use Docker with a repo bind mount for these scripts. |
| `data/` | Generated training datasets and dataset documentation. See `data/README.md`. |
| `models/` | Trained channel-estimator and resource-manager artifacts. |
| `results/` | Local full simulation runs (gitignored). See `results/README.md`. |
| `reports/` | Progress reports and curated evidence summaries. |
| `reports/evidence/` | Cross-cutting promoted plots, tables, and stage summaries. |
| `tests/` | Unit and integration tests for config, CLI flow, estimators, resource managers, plotting, datasets, and DRL policy loading. |
| `reference/dr_athirah_simulation/` | Reference MATLAB-origin PHY/MAC/APP layer material and JIDD-SCMA assets. |
| `thesis/` | LaTeX thesis sources, figures, and notes (gitignored; local only). |
| `archive/` | Superseded drafting workflows kept for reference (gitignored). |

## Docker Setup

The code is a proper Python package (`factory6g`, see `pyproject.toml`). The
Docker image runs `pip install -e .` during build, so image-based runs work out
of the box. For bind-mounted runs against a live checkout, run `pip install -e .`
inside the container first (shown in the examples below).

Build the CPU simulation image:

```bash
docker compose build simulation
```

Run CPU simulations with the `simulation` service:

```bash
docker compose run --rm simulation --config config/config.json
```

Run GPU simulations with the `simulation-gpu` service when the host has NVIDIA
Docker support and `config/config.json` is set for GPU execution:

```bash
docker compose run --rm simulation-gpu --config config/config.json
```

Use `--build` when source code, config defaults, model files, or dependencies
may have changed and you want Docker Compose to rebuild before the run:

```bash
docker compose run --rm --build simulation --config config/config.json
```

Use a repo bind mount when running tests or scripts that are not copied into the
image, or when you need the container to see the exact checkout without a
rebuild:

```bash
docker compose run --rm --entrypoint python -v "$PWD:/app" simulation -m factory6g.cli.run --help
```

## CLI Behavior

The CLI is implemented in `src/factory6g/cli/run.py` (run it with
`python -m factory6g.cli.run` or the installed `factory6g-run` console script;
the `simulation` Docker service uses it as its entrypoint).

- `--estimators ls,dft,adaptive` runs only the estimator stage unless
  `--resource-managers` is also passed.
- `--resource-managers static,pf,drl` runs only the resource-manager stage
  unless `--estimators` is also passed.
- With no method override flags, the run uses estimators from `config/config.json` and
  defaults the resource-manager stage to `max_throughput`.
- `--channel` accepts `rayleigh`, `rician`, `tr38901`, or `awgn`.
- `--modulation` accepts `low` for QPSK, `mid` for 16-QAM, and `high` for
  64-QAM.
- `--factory-size` accepts `s`, `m`, `l`, and `apple`.
- `--stage jidd_scma` runs the JIDD-SCMA flow. For standard estimator or
  resource-manager experiments, prefer selecting methods with `--estimators`
  and/or `--resource-managers`.

## Reproduce Current Result Families

Full reproduction runs can take hours. Do not use these as quick validation
checks unless you intentionally want to regenerate research results.

### Baseline Resource Managers - Rayleigh

```bash
docker compose run --rm --build simulation --config config/config.json --resource-managers static,round_robin,max_throughput,pf,wmmse,queue_aware,drl --channel rayleigh --modulation low --factory-size s
```

### Baseline Resource Managers - TR 38.901 UMi

```bash
docker compose run --rm --build simulation --config config/config.json --resource-managers static,round_robin,max_throughput,pf,wmmse,queue_aware,drl --channel tr38901 --modulation low --factory-size s
```

### BER-First Learned Resource Manager - TR 38.901 UMi

```bash
docker compose run --rm --build simulation --config config/config.json --resource-managers reliability_drl --channel tr38901 --modulation low --factory-size s
```

### BER-First Learned Resource Manager - Rayleigh

```bash
docker compose run --rm --build simulation --config config/config.json --resource-managers reliability_drl --channel rayleigh --modulation low --factory-size s
```

### Resume An Interrupted Resource-Manager Run

Use the channel that matches the original run directory. For UMi runs, pass
`--channel tr38901`; the scenario label in output names is usually `umi`.

```bash
docker compose run -d --rm simulation --config config/config.json --resource-managers static,round_robin,max_throughput,pf,wmmse,queue_aware,drl --channel <rayleigh|tr38901> --modulation low --factory-size s --resume /app/results/<run_dir>
```

The resume path must be the in-container path under `/app/results/`. A resumed
resource-manager stage should log a checkpoint message such as:

```text
[checkpoint] Resuming resource_managers from batch ...
```

## Outputs And Result Interpretation

Every run creates a timestamped directory under `results/`.

The run root contains:

- `simulation.log`: console output and runtime progress.
- `summary_v2.json`: structured summary of stage paths and run metadata.
- `summary_v2.csv`: tabular summary for quick inspection.

Each stage directory, such as `estimators/` or `resource_managers/`, contains:

- `stage_results_v2.json`
- `stage_results_v2.csv`
- `ber_vs_ebno.png`
- `ber_raw_vs_ebno.png`
- `latency_vs_ebno.png`
- `throughput_vs_ebno.png`
- `power_vs_ebno.png`
- `runtime_by_method.png`

Key metrics:

- `ber`: measured bit error rate, computed as `bit_errors / total_bits`.
- `ber_upper_confidence`: conservative upper confidence bound for BER. This is
  important when few or zero bit errors are observed.
- `point_status`: evidence classification for each Eb/No point. `resolved`
  points have enough observed errors; `upper_bound_only` points should be read
  through the confidence bound.
- `throughput_bits_per_batch`: successfully delivered bits per simulated batch.
- `latency_ms`: average per-batch latency in milliseconds.
- `runtime_sec` and `runtime_totals_sec`: simulation runtime, not link-layer
  latency.

For publication-style reliability comparisons, prefer the confidence-aware BER
plot and inspect `ber_upper_confidence` alongside raw `ber`.

## Per-Batch Trace And Run Viewer

Set `"trace": true` in the `simulation` section to make every stage also write
`trace_v1.jsonl.gz` next to `stage_results_v2.json`. Per batch and Eb/No point
it records the shared hall draw (terminal positions, LOS flags, large-scale
gains); per method it records which terminals were scheduled, their power, and
their bit and block errors. The recorder only reads what the stage already
computed, so a traced run gives the same results as an untraced one
(`tests/test_trace_recorder.py` checks this on the real PHY). It is off by
default, so locked runs are unaffected.

The viewer runs in its own TensorFlow-free image (`Dockerfile.viz`, Sionna RT
2.2). Build it once, export a traced run, then serve it:

```bash
docker compose build viewer
```

```bash
docker compose run --rm viewer export results/<run> --radio-map
```

```bash
docker compose run --rm --service-ports viewer serve results/<run>/viewer
```

Open http://localhost:8765. `--radio-map` traces a Sionna RT path-gain map of
the hall at terminal height (a few seconds on CPU for the 15 m hall).
`--scene` also traces every path of one batch's terminals and renders two
stills of the hall (about 35 s on CPU); it uses the channel bank's settings
(depth 5, LOS and specular reflection only). The viewer has six views:

- **Floor:** the hall, terminals and scheduled links for one batch, plus the
  selected terminal's traced reflections on the traced batch.
- **Timeline:** who was scheduled in each batch, and how BER converges.
- **Compare:** BER with batch-level intervals, and the BLER difference paired
  against a baseline method.
- **Run and checks:** whether the trace sums to the stage results, and how
  much evidence the run has.
- **3D hall:** the Sionna RT renders with the radio map and paths.
- **Link:** strongest paths, power delay profile, K-factor, delay spread and
  |H(f)| per base-station port for each terminal.

The same image draws the thesis figure pack and a Rerun recording:

```bash
docker compose run --rm viewer figures results/<run>
```

```bash
docker compose run --rm viewer record results/<run>
```

`figures` writes PNG and PDF files plus `figure_pack_summary.csv` to
`results/<run>/figures/`: the floor with the radio map, traced path gain
against distance with the InF-SL curves, BER with batch-level intervals,
the CDF of delivered bits per scheduled terminal, and fairness against
reliability. `record` writes `results/<run>/recording.rrd`; open it with
`rerun` from the `rerun-sdk` package to scrub batches in 3D.

To see where a batch's time goes, profile the batch path in the simulation
image:

```bash
docker run --rm -v "$PWD":/app -w /app --entrypoint python factory6g:latest scripts/tools/profile_batch_path.py --batches 10
```

## Reliability Report And Provenance

Every run writes `provenance.json` next to `simulation.log`: the config hash,
a hash of the package source, the git commit, the command, library versions
and the seed. Set `"deterministic": true` in `simulation` to ask TensorFlow for
deterministic ops on final evidence runs (slower).

For traced runs, the reliability report gives batch-level intervals, paired
tests against a baseline and, over several seeds, the seed-level spread:

```bash
docker compose run --rm --entrypoint python simulation -m factory6g.analysis report results/<run_seed42> results/<run_seed43> --baseline round_robin
```

It writes `reliability_report.md`, a JSON file and CSV tables to
`<first run>/reliability/`. Per point it reports:

- BER with a batch-level 95 % interval, and BLER with a Clopper-Pearson
  interval over codewords (optimistic, because codewords share a batch).
- Status by bit errors and by block errors.
- The paired per-batch BLER difference against the baseline, with normal and
  bootstrap intervals and a paired-t p-value, Holm-corrected over Eb/N0.
- An exact McNemar test over the codewords both methods carried.

Across seeds it adds the mean and t-interval of per-seed BER, the seed
standard deviation, the interquartile mean, and the probability of
improvement over the baseline. `tests/test_analytic_anchors.py` checks the PHY
against closed-form AWGN and Rayleigh combining results.

## Industry 5.0 Scenario, Service Metrics And Constrained PPO

An optional `industry` config section turns on the Industry 5.0 layer
(`src/factory6g/sim/industry.py`). Without it nothing changes. It provides:

- Device classes per terminal, each with a latency budget, reliability target
  and survival time.
- Body blockage of body-worn devices, drawn from a separate seeded generator.
- Scripted disruptions: `bs_port_fault`, `ut_blockage` and
  `interference_burst`.
- Power caps for body-worn classes.

`config/industry50/config_industry50.json` is an example, and
`scripts/tools/run_industry50.sh` runs it.

The reliability report adds service metrics from the trace, treating each
batch as one scheduling interval:

- Reliability within a latency budget.
- Survival-time violations and availability.
- Uplink energy per delivered bit.
- Tail BLER.

For Industry 5.0 runs it also gives per-class verdicts (met, missed, or
inconclusive when the run is too short) and recovery after each disruption.
The viewer shows these in its Operations view.

`constrained_drl` is a PPO scheduler trained to maximise delivered bits subject
to E[BLER | Eb/N0] ≤ 1 %, with one Lagrange multiplier per Eb/N0 point
(`scripts/tools/train_constrained_ppo.sh`).

`scripts/tools/finetune_site_specific.py` pre-trains the neural estimator on
statistical channels, fine-tunes it on ray-traced samples, and scores it at
held-out hall positions. `python -m factory6g.analysis.bank_validation`
compares the channel bank with the TR 38.901 InF statistics.

## PyTorch Stack (Sionna 2.2)

The simulation also runs on Sionna 2.2 / PyTorch (`Dockerfile.torch`, image
`factory6g-torch:latest`, compose service `simulation-torch`). The TensorFlow
image (`Dockerfile`, Sionna 1.2.1) stays as it is: every locked thesis run was
made with it, and the PyTorch code lives on its own branch.

What changed in the port:

* every TensorFlow op in the PHY path, the estimators, the Industry 5.0 hook,
  PPO training and inference, and neural-estimator training is now PyTorch;
* one random source: all draws (channel, noise, positions, bits) use Sionna's
  `torch.Generator`, seeded by `factory6g.sim.rng.seed_all`;
* Kronecker pilots are drawn on a fixed seed in isolation
  (`factory6g.sim.rng.PILOT_SEED`), as Sionna 1.2.1 did. Sionna 2.2 draws them
  from the shared generator, which would give each model its own pilots and
  shift every later draw;
* TR 38.901 UMi is pinned to `spec_version="16.1"` (Sionna 2.2 defaults to
  19.2); set `tr38901_spec_version` in the system config to change it;
* the LDPC decoder has no early stopping in either version, so decoder
  iterations are reported as the configured `num_decoding_iter`;
* weights trained on the TensorFlow stack load unchanged: the neural estimator
  (`*.keras`) and PPO policies (`policy.keras`) are converted on load
  (`factory6g.torch_io`), and the converted networks match Keras outputs to
  about 1e-6. New training writes `*.pt` / `policy.pt`;
* the legacy supervised DRL and CNN schedulers (`drl_policy.py`,
  `cnn_resource_manager.py`) are used by no campaign and stay TensorFlow-only.

Equivalence gate: the same config on both stacks, with the PyTorch run checked
against the TensorFlow seed spread (Phase 1 seeds 42–46):

```bash
DATA_DIR=$PWD/data scripts/tools/run_phase4_gate.sh c     # Run C, estimators
DATA_DIR=$PWD/data scripts/tools/run_phase4_gate.sh b     # Run B, schedulers
python -m factory6g.analysis.equivalence --stage estimators \
  --reference <the five Phase 1 Run C folders> --candidate results/phase4/<run>/
```

Tests: `docker compose run --rm --entrypoint python simulation-torch -m pytest -q`.

## Dataset, Training, And Reports

Dataset and training workflows are documented in `data/README.md`. Keep those
commands Dockerized. Use a repo bind mount for script-based workflows because
the `scripts/` tree is intended to be executed from the checkout; run an
editable install first so the bind-mounted `factory6g` package resolves:

```bash
docker compose run --rm --entrypoint bash -v "$PWD:/app" simulation -lc "pip install -e . -q && python scripts/tools/generate_rm_ber_report.py"
```

Generated summaries and research-facing tables live under `reports/`. Some
older result documentation may mention legacy result filenames; the current
runtime writes `stage_results_v2.*` and `summary_v2.*`.

## Progress Report Log

Each dated weekly folder includes the stakeholder message draft and, when
available, a concise evidence deck. The table links one primary readable
progress report per date. These weekly packages are the evidence trail for
remote/Friday progress updates used in the Semester 5 Meeting Log (alongside
Microsoft Teams supervisory meetings on 10 Feb, 26 Feb, 16 Mar, and 2 Apr 2026).

| Date | Progress report | Focus |
|---|---|---|
| 2026-01-15 | [Factory6G weekly messages](reports/weekly/2026-01-15/factory6g_weekly_messages.md) | Sem-5 plan and Chapter 2 literature-review structure; transition toward learned estimators and AI resource management. |
| 2026-03-27 | [Factory6G weekly messages](reports/weekly/2026-03-27/factory6g_weekly_messages.md) | March PHY benchmark: Adaptive/PSO estimator evidence, TR 38.901 BER floor, modulation/factory-size impact, and corrected JIDD-SCMA result. |
| 2026-04-20 | [Factory6G weekly messages](reports/weekly/2026-04-20/factory6g_weekly_messages.md) | April neural-estimator update: retrained Neural vs LS Rayleigh evidence, corrected DFT comparison, TR 38.901 estimator tier, and remaining large-factory/JIDD findings. |
| 2026-04-27 | [Factory6G weekly messages](reports/weekly/2026-04-27/factory6g_weekly_messages.md) | AI resource-manager recovery, Docker workflow validation, and Rayleigh QPSK benchmark evidence. |
| 2026-05-01 | [Factory6G weekly messages](reports/weekly/2026-05-01/factory6g_weekly_messages.md) | BER-first DRL resource-manager workflow, Rayleigh/UMi benchmark runs, and Docker validation evidence. |
| 2026-05-15 | [Factory6G weekly messages](reports/weekly/2026-05-15/factory6g_weekly_messages.md) | Special Absence start (11 May 2026); remote Friday weekly cadence and absence milestone plan. |
| 2026-05-23 | [Factory6G weekly messages](reports/weekly/2026-05-23/factory6g_weekly_messages.md) | Cross-channel resource-manager BER screening across Rayleigh, Rician, and UMI/TR38901, with [evidence deck](reports/weekly/2026-05-23/factory6g_weekly_evidence.pptx), [BER comparison](reports/weekly/2026-05-23/resource_manager_ber_comparison.md), and [synthetic channel curves](reports/weekly/2026-05-23/resource_manager_channel_comparison_synthetic.md); `ber_drl` matched the Rayleigh best baseline and ranked second on Rician and UMI/TR38901 in the shortened run. |
| 2026-05-29 | [Factory6G weekly messages](reports/weekly/2026-05-29/factory6g_weekly_messages.md) | Remote thesis drafting (Chapters 3–6) and Chapter 4 evidence-family organization under Special Absence. |
| 2026-06-05 | [Factory6G weekly messages](reports/weekly/2026-06-05/factory6g_weekly_messages.md) | Softbound drafting status; remaining large-factory BER and longer UMi RM/BER-DRL validation gaps. |
| 2026-06-12 | [Factory6G weekly messages](reports/weekly/2026-06-12/factory6g_weekly_messages.md) | Softbound thesis package progress; Chapters 1–5 substantially in place. |
| 2026-06-19 | [Factory6G weekly messages](reports/weekly/2026-06-19/factory6g_weekly_messages.md) | Curated evidence promotion and lean-git zones for tracked thesis claims. |
| 2026-06-26 | [Factory6G weekly messages](reports/weekly/2026-06-26/factory6g_weekly_messages.md) | Sem-5 softbound package ready for supervisor review and signature. |

## Quick Validation

Validate the CLI syntax inside Docker (the bind-mounted checkout needs an
editable install so the `factory6g` package resolves):

```bash
docker compose run --rm --entrypoint bash -v "$PWD:/app" simulation -lc "pip install -e . -q && python -m factory6g.cli.run --help"
```

Run a lightweight test subset inside Docker:

```bash
docker compose run --rm --entrypoint bash -v "$PWD:/app" simulation -lc "pip install -e . -q && python -m pytest tests/test_config_loader.py tests/test_main_cli.py -q"
```

Run the DRL policy loader regression inside Docker:

```bash
docker compose run --rm --entrypoint bash -v "$PWD:/app" simulation -lc "pip install -e . -q && python -m pytest tests/test_drl_policy_pipeline.py -q"
```

Do not run full reproduction simulations as routine validation. They are meant
for regenerating research artifacts and may take several hours depending on the
channel model, enabled methods, hardware, and Monte Carlo stopping behavior.

## Reading Path For New Contributors

1. Read `docs/ARCHITECTURE.md` for the fixed simulation flow and data contracts.
2. Inspect `src/factory6g/cli/run.py` to understand CLI overrides and
   run-directory creation.
3. Follow `src/factory6g/sim/flow.py` into `src/factory6g/sim/stages/estimators.py`
   and `src/factory6g/sim/stages/resource_managers.py`.
4. Read `src/factory6g/models/model.py` for the PHY model composition.
5. Read `src/factory6g/models/resource_manager.py` for the scheduling and
   power-control interface used by all resource managers.
6. Use `data/README.md` only when you need dataset generation or model training
   workflows.
