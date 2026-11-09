# Channel bank v1 (Sionna RT 1.2.1) against v2 (Sionna RT 2.x)

v1 tracer: sionna-rt 1.2.1 (not recorded). v2 tracer: {'sionna.rt': '2.2.0', 'mitsuba': '3.9.1', 'drjit': '1.5.0'}.

Same layout and position seeds; values are 5th / 50th / 95th percentiles over positions.

| Split | Positions | Gain change (dB) | |Δgain| > 1 dB | CFR correlation | K change (dB) | Delay-spread change (ns) | LOS flags agree | Paths (v1 → v2) |
|:--|--:|:--|--:|:--|:--|:--|--:|:--|
| train | 8192 | -3.04 / -0.0264 / 3.2 | 47.3 % | 0.832 / 0.972 / 0.998 | -0.799 / -0.0974 / 3.15 | 0.291 / 1.18 / 1.65 | 100.0 % | 202 → 202 |
| eval | 4096 | -3.1 / -0.0595 / 3.03 | 45.7 % | 0.787 / 0.973 / 0.998 | -0.851 / -0.201 / 2.53 | 0.253 / 1.15 / 1.54 | 100.0 % | 202 → 202 |
