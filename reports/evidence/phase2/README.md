# Phase 2 evidence: constrained PPO and the Industry 5.0 scenario

Generated 9–10 Nov 2026 on branch `phase2-methods` (TensorFlow stack, image
`factory6g:latest`). These are new runs; the locked Runs A–D are untouched.

## Constrained PPO (`constrained_ppo/`)

`scripts/tools/train_constrained_ppo.sh`: 150 iterations × 256 steps in six
25-iteration segments, seed 42, the same budget as the reliability policy.
The objective maximises delivered bits subject to E[BLER | Eb/N0] ≤ 1 %, with
one Lagrange multiplier per Eb/N0 point, capped at 10. The policy is in
`models/constrained_drl_ppo_policy/`; `history.json` has every iteration.

Final multipliers:

| Eb/N0 (dB) | 0 | 3 | 6 | 9 | 12 | 15 | 18–30 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| λ | 10 (cap) | 10 (cap) | 10 (cap) | 10 (cap) | 6.24 | 1.24 | 0 |

A multiplier at the cap means the 1 % BLER target could not be met at that
point with this link (QPSK, 4 terminals): the constraint is infeasible at
0–9 dB, binding at 12–15 dB and slack from 18 dB. The rollout BLER at the last
iteration was 0.28 at 0 dB and about 0.10 at 6–9 dB.

## Industry 5.0 campaign (`industry50/rayleigh`, `industry50/raytraced`)

`scripts/tools/run_industry50.sh`: `config/industry50/config_industry50.json`,
nine schedulers including `constrained_drl`, 160 batches at 6, 12 and 18 dB,
seed 42, Rayleigh and the ray-traced bank. Four device classes (cobot motion,
AGV control, safety wearable, condition sensor); body blockage with
probability 0.3 and 20 dB; three scripted disruptions: terminal 1 blocked
(batches 30–40), all base-station ports down (60–80) and a 10 dB interference
burst (110–120). Each folder holds the reliability report
(`reliability_report.md`), service metrics, per-class verdicts, disruption
recovery, `summary_v2.json` and `provenance.json`. `SOURCE_RUN.txt` names the
full run under `results/`.

Packet delivery at 18 dB (share of all terminal packets delivered per interval):

| Scheduler | Rayleigh | Ray-traced |
|:--|--:|--:|
| static | 0.759 | 0.738 |
| max_throughput | 0.461 | 0.487 |
| wmmse | 0.452 | 0.458 |
| constrained_drl | 0.436 | 0.447 |
| drl | 0.416 | 0.397 |
| queue_aware | 0.417 | 0.420 |
| pf | 0.414 | 0.420 |
| round_robin | 0.408 | 0.420 |
| reliability_drl | 0.388 | 0.419 |

Findings:

- No scheduler meets any class target in this scenario. Of the 108
  class × scheduler × Eb/N0 verdicts per channel, 107 (Rayleigh) and 104
  (ray-traced) are "missed"; the rest are "inconclusive". Confirming a class
  target would need 3 × 10³ (condition sensor) to 3 × 10⁶ (cobot motion)
  packets against the run's 160, so a met verdict is out of reach here; the
  missed verdicts come from reliabilities far below the targets.
- `static` delivers the most because it serves every terminal every
  interval; the others schedule a subset, so unscheduled packets count as not
  delivered. Delivery here is a service measure, not a link BLER.
- At 18 dB `wmmse` uses the least energy per delivered bit (47–48 nJ/bit);
  the others use 53–67 nJ/bit. At 6 dB the gap closes (all 64–76 nJ/bit).
- During the base-station port fault, delivery falls to 0–0.21 (Rayleigh) and
  0.10–0.44 (ray-traced) of packets depending on scheduler and Eb/N0; it is
  back within 0–3 intervals after the fault ends.
- The constrained policy was trained without the Industry 5.0 layer; it does
  not see device classes or disruptions.

## Not done in this phase

- The site-specific fine-tuning of the neural estimator
  (`scripts/tools/finetune_site_specific.py`) was OOM-killed on 9 Nov when it
  shared the machine with three simulations; it is re-run separately.
- A neural receiver was scoped out: training one on this CPU-only machine is
  not feasible in the time available.
