# Phase 3 evidence: channel bank v2 (Sionna RT 2.2)

The v2 bank (`data/rt_channel_bank_v2.npz`, not tracked; manifest tracked) was
traced with sionna-rt 2.2.0 / Mitsuba 3.9.1 / Dr.Jit 1.5.0 from the same hall,
layout and position seeds as v1 (sionna-rt 1.2.1). Mechanisms: LOS and specular
reflection, depth 5, as in v1.

Findings (`bank_compare/`, percentiles 5 / 50 / 95 over positions):

* Link gain: median change about 0 dB, 5–95 % range about ±3 dB; 46 % of
  positions move by more than 1 dB. The floor two-ray interference pattern
  shifts slightly between tracers, which moves individual positions.
* Frequency-response correlation: median 0.97, 5th percentile 0.79–0.83.
* K-factor: median change −0.1 to −0.2 dB.
* RMS delay spread: about +1.2 ns (v1 median 5.2 ns, v2 6.5 ns).
* LOS flags agree at every position; median path count unchanged (202).

Against TR 38.901 InF (`bank_validation_v1/`, `bank_validation_v2/`), both
banks keep the same gaps: LOS delay spread well below the InF median (6.5 ns
vs 23.7 ns) because only specular reflections are traced in an empty hall;
LOS K-factor close to InF (7.8 dB vs 7 dB); LOS path-loss exponent about 5.2
because the floor two-ray pattern dominates the short distances. The NLOS fit
uses 51–166 positions and is not reliable in either bank.

Regenerate in the viz image:

    python -m factory6g.analysis.bank_compare --out results/phase3/bank_compare
    python -m factory6g.analysis.bank_validation --bank data/rt_channel_bank_v2.npz --out results/phase3/bank_validation_v2
