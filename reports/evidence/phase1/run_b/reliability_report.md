# Reliability report

Runs: 20261109_143552_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_reliability_drl_rayleigh_qpsk_s, 20261109_165120_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_reliability_drl_rayleigh_qpsk_s, 20261109_190346_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_reliability_drl_rayleigh_qpsk_s, 20261109_210420_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_reliability_drl_rayleigh_qpsk_s, 20261109_231101_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_reliability_drl_rayleigh_qpsk_s

## Stage `resource_managers`

Seeds: 42, 43, 44, 45, 46. Baseline: `round_robin`.

### Paired tests against the baseline, seed 42

Per Eb/N0 point, Holm-corrected over the points of each method; the paired difference is per-batch BLER.

| Method | Fewer block errors | More block errors | No clear difference | Not testable |
|:--|--:|--:|--:|--:|
| static | 0 | 5 | 6 | 0 |
| max_throughput | 5 | 0 | 6 | 0 |
| pf | 0 | 0 | 11 | 0 |
| wmmse | 5 | 0 | 6 | 0 |
| queue_aware | 0 | 0 | 11 | 0 |
| drl | 4 | 0 | 7 | 0 |
| reliability_drl | 4 | 0 | 7 | 0 |

### Across seeds

BER mean over seeds with a 95 % t-interval; P(better) is the probability that a seed of the method has lower BER than a seed of the baseline.

| Method | Eb/N0 | Mean BER | 95 % over seeds | Seed SD | P(better) | Seeds better |
|:--|--:|--:|:--|--:|--:|--:|
| static | 0 | 0.1174 | 0.1088 – 0.126 | 6.914e-03 | 0.12 | 0/5 |
| static | 3 | 0.0721 | 0.06777 – 0.07643 | 3.489e-03 | 0 | 0/5 |
| static | 6 | 0.03602 | 0.03211 – 0.03993 | 3.150e-03 | 0 | 0/5 |
| static | 9 | 0.01745 | 0.01531 – 0.01959 | 1.725e-03 | 0.04 | 0/5 |
| static | 12 | 8.015e-03 | 5.411e-03 – 0.01062 | 2.097e-03 | 0.12 | 0/5 |
| static | 15 | 3.017e-03 | 1.638e-03 – 4.396e-03 | 1.111e-03 | 0 | 0/5 |
| static | 18 | 4.586e-04 | 0 – 9.256e-04 | 3.762e-04 | 0.12 | 0/5 |
| static | 21 | 2.238e-04 | 2.883e-05 – 4.187e-04 | 1.570e-04 | 0.08 | 1/5 |
| static | 24 | 6.849e-05 | 1.598e-05 – 1.210e-04 | 4.229e-05 | 0.22 | 0/5 |
| static | 27 | 1.283e-05 | 0 – 3.692e-05 | 1.940e-05 | 0.3 | 0/5 |
| static | 30 | 0 | 0 – 0 | 0 | 0.5 | 0/5 |
| round_robin | 0 | 0.103 | 0.09206 – 0.1139 | 8.809e-03 | – | –/– |
| round_robin | 3 | 0.04955 | 0.04544 – 0.05366 | 3.311e-03 | – | –/– |
| round_robin | 6 | 0.02165 | 0.01787 – 0.02544 | 3.045e-03 | – | –/– |
| round_robin | 9 | 0.01064 | 7.038e-03 – 0.01424 | 2.900e-03 | – | –/– |
| round_robin | 12 | 4.190e-03 | 9.334e-04 – 7.447e-03 | 2.623e-03 | – | –/– |
| round_robin | 15 | 9.278e-04 | 5.451e-04 – 1.311e-03 | 3.082e-04 | – | –/– |
| round_robin | 18 | 1.299e-04 | 0 – 4.664e-04 | 2.710e-04 | – | –/– |
| round_robin | 21 | 5.549e-05 | 0 – 1.236e-04 | 5.485e-05 | – | –/– |
| round_robin | 24 | 4.118e-05 | 6.656e-06 – 7.570e-05 | 2.780e-05 | – | –/– |
| round_robin | 27 | 0 | 0 – 0 | 0 | – | –/– |
| round_robin | 30 | 0 | 0 – 0 | 0 | – | –/– |
| max_throughput | 0 | 0.01971 | 0.01456 – 0.02485 | 4.146e-03 | 1 | 5/5 |
| max_throughput | 3 | 4.565e-03 | 3.885e-03 – 5.244e-03 | 5.470e-04 | 1 | 5/5 |
| max_throughput | 6 | 8.068e-04 | 2.542e-04 – 1.359e-03 | 4.451e-04 | 1 | 5/5 |
| max_throughput | 9 | 5.107e-05 | 0 – 1.252e-04 | 5.973e-05 | 1 | 5/5 |
| max_throughput | 12 | 0 | 0 – 0 | 0 | 1 | 5/5 |
| max_throughput | 15 | 0 | 0 – 0 | 0 | 1 | 5/5 |
| max_throughput | 18 | 0 | 0 – 0 | 0 | 0.7 | 2/5 |
| max_throughput | 21 | 0 | 0 – 0 | 0 | 0.8 | 3/5 |
| max_throughput | 24 | 0 | 0 – 0 | 0 | 0.9 | 4/5 |
| max_throughput | 27 | 0 | 0 – 0 | 0 | 0.5 | 0/5 |
| max_throughput | 30 | 0 | 0 – 0 | 0 | 0.5 | 0/5 |
| pf | 0 | 0.09826 | 0.08932 – 0.1072 | 7.202e-03 | 0.84 | 5/5 |
| pf | 3 | 0.04844 | 0.04234 – 0.05453 | 4.909e-03 | 0.5 | 1/5 |
| pf | 6 | 0.02151 | 0.01917 – 0.02385 | 1.888e-03 | 0.54 | 2/5 |
| pf | 9 | 8.578e-03 | 6.236e-03 – 0.01092 | 1.886e-03 | 0.68 | 3/5 |
| pf | 12 | 4.536e-03 | 1.785e-03 – 7.287e-03 | 2.216e-03 | 0.46 | 0/5 |
| pf | 15 | 1.525e-03 | 4.039e-04 – 2.647e-03 | 9.032e-04 | 0.28 | 1/5 |
| pf | 18 | 1.424e-04 | 0 – 4.459e-04 | 2.444e-04 | 0.36 | 1/5 |
| pf | 21 | 8.555e-05 | 0 – 1.866e-04 | 8.139e-05 | 0.4 | 1/5 |
| pf | 24 | 6.852e-06 | 0 – 2.565e-05 | 1.514e-05 | 0.84 | 3/5 |
| pf | 27 | 0 | 0 – 0 | 0 | 0.5 | 0/5 |
| pf | 30 | 0 | 0 – 0 | 0 | 0.5 | 0/5 |
| wmmse | 0 | 0.02414 | 0.0181 – 0.03018 | 4.861e-03 | 1 | 5/5 |
| wmmse | 3 | 7.722e-03 | 6.108e-03 – 9.336e-03 | 1.300e-03 | 1 | 5/5 |
| wmmse | 6 | 2.018e-03 | 1.009e-03 – 3.027e-03 | 8.124e-04 | 1 | 5/5 |
| wmmse | 9 | 3.129e-04 | 1.664e-04 – 4.594e-04 | 1.180e-04 | 1 | 5/5 |
| wmmse | 12 | 1.388e-04 | 0 – 3.561e-04 | 1.750e-04 | 1 | 5/5 |
| wmmse | 15 | 4.136e-05 | 0 – 1.455e-04 | 8.389e-05 | 1 | 5/5 |
| wmmse | 18 | 5.501e-06 | 0 – 2.078e-05 | 1.230e-05 | 0.64 | 2/5 |
| wmmse | 21 | 1.932e-05 | 0 – 7.296e-05 | 4.320e-05 | 0.68 | 3/5 |
| wmmse | 24 | 5.013e-05 | 0 – 1.684e-04 | 9.527e-05 | 0.7 | 3/5 |
| wmmse | 27 | 3.662e-06 | 0 – 1.383e-05 | 8.189e-06 | 0.4 | 0/5 |
| wmmse | 30 | 0 | 0 – 0 | 0 | 0.5 | 0/5 |
| queue_aware | 0 | 0.1047 | 0.09252 – 0.1168 | 9.778e-03 | 0.4 | 1/5 |
| queue_aware | 3 | 0.0546 | 0.04935 – 0.05984 | 4.225e-03 | 0.16 | 1/5 |
| queue_aware | 6 | 0.0267 | 0.02379 – 0.02961 | 2.341e-03 | 0.08 | 0/5 |
| queue_aware | 9 | 0.01183 | 8.276e-03 – 0.01539 | 2.865e-03 | 0.44 | 3/5 |
| queue_aware | 12 | 6.009e-03 | 3.159e-03 – 8.859e-03 | 2.295e-03 | 0.28 | 0/5 |
| queue_aware | 15 | 1.958e-03 | 1.021e-03 – 2.896e-03 | 7.550e-04 | 0.08 | 0/5 |
| queue_aware | 18 | 3.329e-04 | 0 – 9.749e-04 | 5.170e-04 | 0.16 | 0/5 |
| queue_aware | 21 | 2.159e-04 | 0 – 5.254e-04 | 2.493e-04 | 0.24 | 0/5 |
| queue_aware | 24 | 2.713e-05 | 0 – 7.360e-05 | 3.743e-05 | 0.62 | 2/5 |
| queue_aware | 27 | 0 | 0 – 0 | 0 | 0.5 | 0/5 |
| queue_aware | 30 | 0 | 0 – 0 | 0 | 0.5 | 0/5 |
| drl | 0 | 0.04336 | 0.03361 – 0.0531 | 7.849e-03 | 1 | 5/5 |
| drl | 3 | 0.01769 | 0.01366 – 0.02172 | 3.247e-03 | 1 | 5/5 |
| drl | 6 | 8.228e-03 | 5.959e-03 – 0.0105 | 1.827e-03 | 1 | 5/5 |
| drl | 9 | 2.906e-03 | 1.649e-03 – 4.163e-03 | 1.012e-03 | 1 | 5/5 |
| drl | 12 | 1.208e-03 | 3.147e-04 – 2.102e-03 | 7.197e-04 | 0.92 | 5/5 |
| drl | 15 | 9.010e-04 | 0 – 2.014e-03 | 8.966e-04 | 0.64 | 3/5 |
| drl | 18 | 2.229e-04 | 0 – 5.934e-04 | 2.984e-04 | 0.3 | 0/5 |
| drl | 21 | 1.003e-04 | 5.064e-06 – 1.956e-04 | 7.671e-05 | 0.28 | 0/5 |
| drl | 24 | 1.772e-05 | 0 – 4.029e-05 | 1.818e-05 | 0.8 | 3/5 |
| drl | 27 | 5.062e-06 | 0 – 1.912e-05 | 1.132e-05 | 0.4 | 0/5 |
| drl | 30 | 0 | 0 – 0 | 0 | 0.5 | 0/5 |
| reliability_drl | 0 | 0.02733 | 0.02 – 0.03467 | 5.905e-03 | 1 | 5/5 |
| reliability_drl | 3 | 0.01337 | 9.316e-03 – 0.01743 | 3.266e-03 | 1 | 5/5 |
| reliability_drl | 6 | 4.632e-03 | 3.404e-03 – 5.860e-03 | 9.891e-04 | 1 | 5/5 |
| reliability_drl | 9 | 2.123e-03 | 9.982e-04 – 3.248e-03 | 9.060e-04 | 1 | 5/5 |
| reliability_drl | 12 | 2.540e-03 | 9.948e-04 – 4.086e-03 | 1.245e-03 | 0.72 | 4/5 |
| reliability_drl | 15 | 7.241e-04 | 3.184e-04 – 1.130e-03 | 3.267e-04 | 0.72 | 4/5 |
| reliability_drl | 18 | 1.430e-04 | 0 – 3.356e-04 | 1.552e-04 | 0.34 | 1/5 |
| reliability_drl | 21 | 3.208e-05 | 0 – 8.673e-05 | 4.401e-05 | 0.6 | 2/5 |
| reliability_drl | 24 | 1.636e-05 | 0 – 4.447e-05 | 2.264e-05 | 0.74 | 3/5 |
| reliability_drl | 27 | 5.062e-06 | 0 – 1.912e-05 | 1.132e-05 | 0.4 | 0/5 |
| reliability_drl | 30 | 0 | 0 – 0 | 0 | 0.5 | 0/5 |

