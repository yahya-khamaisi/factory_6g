# Phase 1 evidence: reliability over five seeds

Generated 9–10 Nov 2026 on branch `phase1-reliability` (TensorFlow stack,
image `factory6g:latest`). New runs only; the locked Runs A–D are untouched.

`scripts/tools/run_phase1_seeds.sh b|c` re-ran the Rayleigh cells of Run B
(eight schedulers) and Run C (five estimators) at seeds 42–46 with the
per-batch trace on (`config/phase1/`). Everything else equals the locked
configs. Each folder holds the report from

    python -m factory6g.analysis report <5 runs> --baseline round_robin --stage resource_managers   # run_b
    python -m factory6g.analysis report <5 runs> --baseline ls --stage estimators                   # run_c

with `SOURCE_RUNS.txt` naming the five runs under `results/` and
`provenance_seed42.json` the hashes and versions of the seed-42 run.

## Seed 42 against the locked runs

- Run C, Rayleigh: all 55 method × Eb/N0 points have the same bit errors and
  bit counts as the locked run `20261015_073321`.
- Run B, Rayleigh: 87 of 88 points are identical to the locked run
  `20261015_073404`. The exception is `round_robin` at 15 dB (7170 against
  7160 bit errors, Jain fairness 1.00 against 0.81). This comes from the
  round-robin fix in `505e3ac` (one pointer per Eb/N0 point), which landed
  after Run B was locked and removes the fairness artefact described in
  `thesis/notes/ROUND_ROBIN_FAIRNESS_2026-10.md`.

So the trace, provenance and reliability code added in Phase 1 do not change
the simulation.

## What five seeds show

- Rankings hold across seeds. At 6 dB in Run B, `max_throughput` (mean BER
  8.1 × 10⁻⁴), `wmmse` (2.0 × 10⁻³), `reliability_drl` (4.6 × 10⁻³) and `drl`
  (8.2 × 10⁻³) beat `round_robin` (2.2 × 10⁻²) on all five seeds; `pf` and
  `round_robin` are not distinguishable (P(better) 0.54); `static` is worse on
  all five. In Run C every estimator beats LS on all five seeds from 0 to
  21 dB; `neural` is lowest at 6 and 12 dB (2.6 × 10⁻² and 4.1 × 10⁻³).
- The seed-to-seed spread is larger than single-run intervals suggest. At
  0–9 dB (points with BER above 10⁻³) the seed SD is 4–15 % of the mean BER
  in Run C and 5–43 % in Run B; at 12 dB and above, where few errors remain,
  it is 15–224 %. The per-batch interval of one run
  understates this, as the October seed-43 check found.
- Seed 42 tends to sit low: it is the lowest-BER seed at 10 of 20 Run C points
  and 8 of 32 Run B points between 0 and 9 dB (for example LS at 6 dB:
  5.4 × 10⁻² against 6.4–6.8 × 10⁻² for the other seeds). The shift is
  common to all methods, so comparisons inside the locked runs are fair, but
  absolute low-SNR BERs from seed 42 are slightly optimistic.
- Paired per-batch BLER tests at seed 42 (Holm-corrected) give the same
  picture: `max_throughput` and `wmmse` have fewer block errors than
  `round_robin` at 5 of 11 points, `drl` and `reliability_drl` at 4,
  `static` more at 5; every estimator has fewer block errors than LS at
  6–7 points.
