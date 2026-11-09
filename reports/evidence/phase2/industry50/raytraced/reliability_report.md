# Reliability report

Runs: 20261109_215935_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_reliability_drl_constrained_drl_raytraced_qpsk_s

## Stage `resource_managers`

Seeds: 42. Baseline: `round_robin`.

### Paired tests against the baseline, seed 42

Per Eb/N0 point, Holm-corrected over the points of each method; the paired difference is per-batch BLER.

| Method | Fewer block errors | More block errors | No clear difference | Not testable |
|:--|--:|--:|--:|--:|
| static | 0 | 3 | 0 | 0 |
| max_throughput | 3 | 0 | 0 | 0 |
| pf | 0 | 0 | 3 | 0 |
| wmmse | 3 | 0 | 0 | 0 |
| queue_aware | 0 | 0 | 3 | 0 |
| drl | 1 | 0 | 2 | 0 |
| reliability_drl | 2 | 0 | 1 | 0 |
| constrained_drl | 3 | 0 | 0 | 0 |

### Device classes, seed 42

Reliability within the class latency budget against the class target. *met*: the lower 95 % bound clears the target; *missed*: the upper bound is below it; *inconclusive*: the run has too few packets to decide (see *packets needed*).

| Method | Eb/N0 | Class | Budget | Target | Reliability | 95 % | Verdict | Packets | Packets needed | Survival violations |
|:--|--:|:--|--:|--:|--:|:--|:--|--:|--:|--:|
| static | 6 | Cobot motion control | 1 | 0.999999 | 0.55 | 0.4695 – 0.6286 | missed | 160 | 3000000 | 19 |
| static | 6 | Mobile robot control | 2 | 0.99999 | 0.6164 | 0.536 – 0.6923 | missed | 159 | 300001 | 13 |
| static | 6 | Worker safety wearable | 1 | 0.99999 | 0.25 | 0.185 – 0.3245 | missed | 160 | 300001 | 23 |
| static | 6 | Condition monitoring | 4 | 0.999 | 0.949 | 0.9021 – 0.9777 | missed | 157 | 3000 | 3 |
| static | 12 | Cobot motion control | 1 | 0.999999 | 0.7 | 0.6226 – 0.7698 | missed | 160 | 3000000 | 11 |
| static | 12 | Mobile robot control | 2 | 0.99999 | 0.7987 | 0.7279 – 0.8581 | missed | 159 | 300001 | 10 |
| static | 12 | Worker safety wearable | 1 | 0.99999 | 0.4125 | 0.3354 – 0.4929 | missed | 160 | 300001 | 21 |
| static | 12 | Condition monitoring | 4 | 0.999 | 0.9554 | 0.9103 – 0.9819 | missed | 157 | 3000 | 2 |
| static | 18 | Cobot motion control | 1 | 0.999999 | 0.8125 | 0.7433 – 0.8698 | missed | 160 | 3000000 | 5 |
| static | 18 | Mobile robot control | 2 | 0.99999 | 0.8742 | 0.8124 – 0.9214 | missed | 159 | 300001 | 5 |
| static | 18 | Worker safety wearable | 1 | 0.99999 | 0.5375 | 0.457 – 0.6165 | missed | 160 | 300001 | 16 |
| static | 18 | Condition monitoring | 4 | 0.999 | 0.9873 | 0.9547 – 0.9985 | missed | 157 | 3000 | 0 |
| round_robin | 6 | Cobot motion control | 1 | 0.999999 | 0.3312 | 0.259 – 0.4099 | missed | 160 | 3000000 | 19 |
| round_robin | 6 | Mobile robot control | 2 | 0.99999 | 0.4843 | 0.4044 – 0.5648 | missed | 159 | 300001 | 16 |
| round_robin | 6 | Worker safety wearable | 1 | 0.99999 | 0.175 | 0.1195 – 0.2429 | missed | 160 | 300001 | 15 |
| round_robin | 6 | Condition monitoring | 4 | 0.999 | 0.8726 | 0.8101 – 0.9204 | missed | 157 | 3000 | 8 |
| round_robin | 12 | Cobot motion control | 1 | 0.999999 | 0.4 | 0.3235 – 0.4803 | missed | 160 | 3000000 | 10 |
| round_robin | 12 | Mobile robot control | 2 | 0.99999 | 0.7358 | 0.6602 – 0.8025 | missed | 159 | 300001 | 10 |
| round_robin | 12 | Worker safety wearable | 1 | 0.99999 | 0.275 | 0.2075 – 0.3511 | missed | 160 | 300001 | 22 |
| round_robin | 12 | Condition monitoring | 4 | 0.999 | 0.9363 | 0.886 – 0.969 | missed | 157 | 3000 | 4 |
| round_robin | 18 | Cobot motion control | 1 | 0.999999 | 0.45 | 0.3714 – 0.5305 | missed | 160 | 3000000 | 6 |
| round_robin | 18 | Mobile robot control | 2 | 0.99999 | 0.8491 | 0.7838 – 0.9008 | missed | 159 | 300001 | 8 |
| round_robin | 18 | Worker safety wearable | 1 | 0.99999 | 0.325 | 0.2532 – 0.4034 | missed | 160 | 300001 | 17 |
| round_robin | 18 | Condition monitoring | 4 | 0.999 | 1 | 0.9768 – 1 | inconclusive | 157 | 3000 | 0 |
| max_throughput | 6 | Cobot motion control | 1 | 0.999999 | 0.5 | 0.42 – 0.58 | missed | 160 | 3000000 | 24 |
| max_throughput | 6 | Mobile robot control | 2 | 0.99999 | 0.5094 | 0.4291 – 0.5895 | missed | 159 | 300001 | 18 |
| max_throughput | 6 | Worker safety wearable | 1 | 0.99999 | 0.2313 | 0.1683 – 0.3044 | missed | 160 | 300001 | 22 |
| max_throughput | 6 | Condition monitoring | 4 | 0.999 | 0.9172 | 0.8626 – 0.9552 | missed | 157 | 3000 | 4 |
| max_throughput | 12 | Cobot motion control | 1 | 0.999999 | 0.5188 | 0.4385 – 0.5983 | missed | 160 | 3000000 | 16 |
| max_throughput | 12 | Mobile robot control | 2 | 0.99999 | 0.7421 | 0.6669 – 0.8082 | missed | 159 | 300001 | 17 |
| max_throughput | 12 | Worker safety wearable | 1 | 0.99999 | 0.3187 | 0.2474 – 0.397 | missed | 160 | 300001 | 23 |
| max_throughput | 12 | Condition monitoring | 4 | 0.999 | 0.8854 | 0.8249 – 0.9306 | missed | 157 | 3000 | 5 |
| max_throughput | 18 | Cobot motion control | 1 | 0.999999 | 0.5625 | 0.482 – 0.6407 | missed | 160 | 3000000 | 16 |
| max_throughput | 18 | Mobile robot control | 2 | 0.99999 | 0.7736 | 0.7006 – 0.8361 | missed | 159 | 300001 | 14 |
| max_throughput | 18 | Worker safety wearable | 1 | 0.99999 | 0.3312 | 0.259 – 0.4099 | missed | 160 | 300001 | 24 |
| max_throughput | 18 | Condition monitoring | 4 | 0.999 | 0.9108 | 0.8549 – 0.9504 | missed | 157 | 3000 | 4 |
| pf | 6 | Cobot motion control | 1 | 0.999999 | 0.3312 | 0.259 – 0.4099 | missed | 160 | 3000000 | 19 |
| pf | 6 | Mobile robot control | 2 | 0.99999 | 0.4843 | 0.4044 – 0.5648 | missed | 159 | 300001 | 16 |
| pf | 6 | Worker safety wearable | 1 | 0.99999 | 0.175 | 0.1195 – 0.2429 | missed | 160 | 300001 | 15 |
| pf | 6 | Condition monitoring | 4 | 0.999 | 0.8726 | 0.8101 – 0.9204 | missed | 157 | 3000 | 8 |
| pf | 12 | Cobot motion control | 1 | 0.999999 | 0.4 | 0.3235 – 0.4803 | missed | 160 | 3000000 | 9 |
| pf | 12 | Mobile robot control | 2 | 0.99999 | 0.7484 | 0.6736 – 0.8138 | missed | 159 | 300001 | 9 |
| pf | 12 | Worker safety wearable | 1 | 0.99999 | 0.225 | 0.1628 – 0.2976 | missed | 160 | 300001 | 17 |
| pf | 12 | Condition monitoring | 4 | 0.999 | 0.9618 | 0.9187 – 0.9858 | missed | 157 | 3000 | 2 |
| pf | 18 | Cobot motion control | 1 | 0.999999 | 0.45 | 0.3714 – 0.5305 | missed | 160 | 3000000 | 6 |
| pf | 18 | Mobile robot control | 2 | 0.99999 | 0.8491 | 0.7838 – 0.9008 | missed | 159 | 300001 | 8 |
| pf | 18 | Worker safety wearable | 1 | 0.99999 | 0.325 | 0.2532 – 0.4034 | missed | 160 | 300001 | 17 |
| pf | 18 | Condition monitoring | 4 | 0.999 | 1 | 0.9768 – 1 | inconclusive | 157 | 3000 | 0 |
| wmmse | 6 | Cobot motion control | 1 | 0.999999 | 0.5188 | 0.4385 – 0.5983 | missed | 160 | 3000000 | 21 |
| wmmse | 6 | Mobile robot control | 2 | 0.99999 | 0.522 | 0.4415 – 0.6017 | missed | 159 | 300001 | 18 |
| wmmse | 6 | Worker safety wearable | 1 | 0.99999 | 0.2375 | 0.1739 – 0.3111 | missed | 160 | 300001 | 23 |
| wmmse | 6 | Condition monitoring | 4 | 0.999 | 0.9363 | 0.886 – 0.969 | missed | 157 | 3000 | 3 |
| wmmse | 12 | Cobot motion control | 1 | 0.999999 | 0.55 | 0.4695 – 0.6286 | missed | 160 | 3000000 | 15 |
| wmmse | 12 | Mobile robot control | 2 | 0.99999 | 0.7107 | 0.6336 – 0.7798 | missed | 159 | 300001 | 19 |
| wmmse | 12 | Worker safety wearable | 1 | 0.99999 | 0.3187 | 0.2474 – 0.397 | missed | 160 | 300001 | 20 |
| wmmse | 12 | Condition monitoring | 4 | 0.999 | 0.8599 | 0.7956 – 0.9101 | missed | 157 | 3000 | 6 |
| wmmse | 18 | Cobot motion control | 1 | 0.999999 | 0.5062 | 0.4262 – 0.5861 | missed | 160 | 3000000 | 18 |
| wmmse | 18 | Mobile robot control | 2 | 0.99999 | 0.7044 | 0.627 – 0.774 | missed | 159 | 300001 | 16 |
| wmmse | 18 | Worker safety wearable | 1 | 0.99999 | 0.3812 | 0.3057 – 0.4613 | missed | 160 | 300001 | 23 |
| wmmse | 18 | Condition monitoring | 4 | 0.999 | 0.9172 | 0.8626 – 0.9552 | missed | 157 | 3000 | 3 |
| queue_aware | 6 | Cobot motion control | 1 | 0.999999 | 0.3063 | 0.2359 – 0.3839 | missed | 160 | 3000000 | 21 |
| queue_aware | 6 | Mobile robot control | 2 | 0.99999 | 0.4906 | 0.4105 – 0.5709 | missed | 159 | 300001 | 13 |
| queue_aware | 6 | Worker safety wearable | 1 | 0.99999 | 0.1625 | 0.109 – 0.229 | missed | 160 | 300001 | 16 |
| queue_aware | 6 | Condition monitoring | 4 | 0.999 | 0.8726 | 0.8101 – 0.9204 | missed | 157 | 3000 | 4 |
| queue_aware | 12 | Cobot motion control | 1 | 0.999999 | 0.4062 | 0.3294 – 0.4866 | missed | 160 | 3000000 | 9 |
| queue_aware | 12 | Mobile robot control | 2 | 0.99999 | 0.7358 | 0.6602 – 0.8025 | missed | 159 | 300001 | 10 |
| queue_aware | 12 | Worker safety wearable | 1 | 0.99999 | 0.25 | 0.185 – 0.3245 | missed | 160 | 300001 | 17 |
| queue_aware | 12 | Condition monitoring | 4 | 0.999 | 0.9363 | 0.886 – 0.969 | missed | 157 | 3000 | 3 |
| queue_aware | 18 | Cobot motion control | 1 | 0.999999 | 0.4625 | 0.3835 – 0.543 | missed | 160 | 3000000 | 4 |
| queue_aware | 18 | Mobile robot control | 2 | 0.99999 | 0.8428 | 0.7767 – 0.8956 | missed | 159 | 300001 | 10 |
| queue_aware | 18 | Worker safety wearable | 1 | 0.99999 | 0.325 | 0.2532 – 0.4034 | missed | 160 | 300001 | 19 |
| queue_aware | 18 | Condition monitoring | 4 | 0.999 | 1 | 0.9768 – 1 | inconclusive | 157 | 3000 | 0 |
| drl | 6 | Cobot motion control | 1 | 0.999999 | 0.375 | 0.2998 – 0.4549 | missed | 160 | 3000000 | 27 |
| drl | 6 | Mobile robot control | 2 | 0.99999 | 0.4843 | 0.4044 – 0.5648 | missed | 159 | 300001 | 21 |
| drl | 6 | Worker safety wearable | 1 | 0.99999 | 0.2375 | 0.1739 – 0.3111 | missed | 160 | 300001 | 24 |
| drl | 6 | Condition monitoring | 4 | 0.999 | 0.8917 | 0.8323 – 0.9356 | missed | 157 | 3000 | 5 |
| drl | 12 | Cobot motion control | 1 | 0.999999 | 0.4375 | 0.3593 – 0.518 | missed | 160 | 3000000 | 20 |
| drl | 12 | Mobile robot control | 2 | 0.99999 | 0.7107 | 0.6336 – 0.7798 | missed | 159 | 300001 | 22 |
| drl | 12 | Worker safety wearable | 1 | 0.99999 | 0.2938 | 0.2245 – 0.3708 | missed | 160 | 300001 | 27 |
| drl | 12 | Condition monitoring | 4 | 0.999 | 0.949 | 0.9021 – 0.9777 | missed | 157 | 3000 | 2 |
| drl | 18 | Cobot motion control | 1 | 0.999999 | 0.4062 | 0.3294 – 0.4866 | missed | 160 | 3000000 | 18 |
| drl | 18 | Mobile robot control | 2 | 0.99999 | 0.7673 | 0.6938 – 0.8306 | missed | 159 | 300001 | 16 |
| drl | 18 | Worker safety wearable | 1 | 0.99999 | 0.2812 | 0.2131 – 0.3577 | missed | 160 | 300001 | 24 |
| drl | 18 | Condition monitoring | 4 | 0.999 | 0.9618 | 0.9187 – 0.9858 | missed | 157 | 3000 | 2 |
| reliability_drl | 6 | Cobot motion control | 1 | 0.999999 | 0.4562 | 0.3774 – 0.5367 | missed | 160 | 3000000 | 25 |
| reliability_drl | 6 | Mobile robot control | 2 | 0.99999 | 0.3836 | 0.3077 – 0.464 | missed | 159 | 300001 | 19 |
| reliability_drl | 6 | Worker safety wearable | 1 | 0.99999 | 0.2313 | 0.1683 – 0.3044 | missed | 160 | 300001 | 22 |
| reliability_drl | 6 | Condition monitoring | 4 | 0.999 | 0.949 | 0.9021 – 0.9777 | missed | 157 | 3000 | 3 |
| reliability_drl | 12 | Cobot motion control | 1 | 0.999999 | 0.3438 | 0.2706 – 0.4228 | missed | 160 | 3000000 | 22 |
| reliability_drl | 12 | Mobile robot control | 2 | 0.99999 | 0.4403 | 0.3617 – 0.5211 | missed | 159 | 300001 | 23 |
| reliability_drl | 12 | Worker safety wearable | 1 | 0.99999 | 0.3438 | 0.2706 – 0.4228 | missed | 160 | 300001 | 22 |
| reliability_drl | 12 | Condition monitoring | 4 | 0.999 | 0.9554 | 0.9103 – 0.9819 | missed | 157 | 3000 | 2 |
| reliability_drl | 18 | Cobot motion control | 1 | 0.999999 | 0.2938 | 0.2245 – 0.3708 | missed | 160 | 3000000 | 25 |
| reliability_drl | 18 | Mobile robot control | 2 | 0.99999 | 0.5283 | 0.4477 – 0.6079 | missed | 159 | 300001 | 21 |
| reliability_drl | 18 | Worker safety wearable | 1 | 0.99999 | 0.4562 | 0.3774 – 0.5367 | missed | 160 | 300001 | 23 |
| reliability_drl | 18 | Condition monitoring | 4 | 0.999 | 0.9873 | 0.9547 – 0.9985 | missed | 157 | 3000 | 0 |
| constrained_drl | 6 | Cobot motion control | 1 | 0.999999 | 0.5312 | 0.4508 – 0.6105 | missed | 160 | 3000000 | 20 |
| constrained_drl | 6 | Mobile robot control | 2 | 0.99999 | 0.3711 | 0.2959 – 0.4512 | missed | 159 | 300001 | 18 |
| constrained_drl | 6 | Worker safety wearable | 1 | 0.99999 | 0.1625 | 0.109 – 0.229 | missed | 160 | 300001 | 20 |
| constrained_drl | 6 | Condition monitoring | 4 | 0.999 | 0.949 | 0.9021 – 0.9777 | missed | 157 | 3000 | 3 |
| constrained_drl | 12 | Cobot motion control | 1 | 0.999999 | 0.4875 | 0.4078 – 0.5677 | missed | 160 | 3000000 | 20 |
| constrained_drl | 12 | Mobile robot control | 2 | 0.99999 | 0.4151 | 0.3376 – 0.4958 | missed | 159 | 300001 | 20 |
| constrained_drl | 12 | Worker safety wearable | 1 | 0.99999 | 0.2313 | 0.1683 – 0.3044 | missed | 160 | 300001 | 21 |
| constrained_drl | 12 | Condition monitoring | 4 | 0.999 | 0.9745 | 0.9361 – 0.993 | missed | 157 | 3000 | 1 |
| constrained_drl | 18 | Cobot motion control | 1 | 0.999999 | 0.4875 | 0.4078 – 0.5677 | missed | 160 | 3000000 | 16 |
| constrained_drl | 18 | Mobile robot control | 2 | 0.99999 | 0.2893 | 0.2202 – 0.3664 | missed | 159 | 300001 | 18 |
| constrained_drl | 18 | Worker safety wearable | 1 | 0.99999 | 0.2313 | 0.1683 – 0.3044 | missed | 160 | 300001 | 24 |
| constrained_drl | 18 | Condition monitoring | 4 | 0.999 | 1 | 0.9768 – 1 | inconclusive | 157 | 3000 | 0 |

### Disruptions, seed 42

Delivery is the share of terminals whose packet got through in an interval; recovery counts intervals after the event ends until a 3-interval mean is back at 95 % of the pre-event level.

| Method | Eb/N0 | Event | Batches | Delivery before | Delivery during | Recovery (intervals) |
|:--|--:|:--|:--|--:|--:|--:|
| static | 6 | ut_blockage | 30–40 | 0.5083 | 0.4545 | 0 |
| static | 6 | bs_port_fault | 60–80 | 0.5292 | 0.1786 | 0 |
| static | 6 | interference_burst | 110–120 | 0.4705 | 0.1818 | 0 |
| static | 12 | ut_blockage | 30–40 | 0.6917 | 0.5455 | 0 |
| static | 12 | bs_port_fault | 60–80 | 0.6833 | 0.2857 | 3 |
| static | 12 | interference_burst | 110–120 | 0.6045 | 0.2955 | 0 |
| static | 18 | ut_blockage | 30–40 | 0.8333 | 0.6364 | 2 |
| static | 18 | bs_port_fault | 60–80 | 0.7875 | 0.3095 | 0 |
| static | 18 | interference_burst | 110–120 | 0.7227 | 0.4773 | 0 |
| round_robin | 6 | ut_blockage | 30–40 | 0.3417 | 0.1818 | 0 |
| round_robin | 6 | bs_port_fault | 60–80 | 0.3167 | 0.119 | 0 |
| round_robin | 6 | interference_burst | 110–120 | 0.2773 | 0.1364 | 0 |
| round_robin | 12 | ut_blockage | 30–40 | 0.4083 | 0.3409 | 0 |
| round_robin | 12 | bs_port_fault | 60–80 | 0.3958 | 0.1905 | 1 |
| round_robin | 12 | interference_burst | 110–120 | 0.3614 | 0.2045 | 0 |
| round_robin | 18 | ut_blockage | 30–40 | 0.475 | 0.3409 | 3 |
| round_robin | 18 | bs_port_fault | 60–80 | 0.4375 | 0.3095 | 0 |
| round_robin | 18 | interference_burst | 110–120 | 0.4205 | 0.25 | 0 |
| max_throughput | 6 | ut_blockage | 30–40 | 0.4417 | 0.4318 | 1 |
| max_throughput | 6 | bs_port_fault | 60–80 | 0.45 | 0.1905 | 0 |
| max_throughput | 6 | interference_burst | 110–120 | 0.4023 | 0.1818 | 0 |
| max_throughput | 12 | ut_blockage | 30–40 | 0.475 | 0.4545 | 0 |
| max_throughput | 12 | bs_port_fault | 60–80 | 0.4792 | 0.3333 | 0 |
| max_throughput | 12 | interference_burst | 110–120 | 0.4545 | 0.2727 | 0 |
| max_throughput | 18 | ut_blockage | 30–40 | 0.5 | 0.5 | 0 |
| max_throughput | 18 | bs_port_fault | 60–80 | 0.5 | 0.4405 | 0 |
| max_throughput | 18 | interference_burst | 110–120 | 0.4886 | 0.4318 | 0 |
| pf | 6 | ut_blockage | 30–40 | 0.3417 | 0.1818 | 0 |
| pf | 6 | bs_port_fault | 60–80 | 0.3167 | 0.119 | 0 |
| pf | 6 | interference_burst | 110–120 | 0.2773 | 0.1364 | 0 |
| pf | 12 | ut_blockage | 30–40 | 0.4 | 0.2273 | 1 |
| pf | 12 | bs_port_fault | 60–80 | 0.3667 | 0.2024 | 1 |
| pf | 12 | interference_burst | 110–120 | 0.3455 | 0.1364 | 0 |
| pf | 18 | ut_blockage | 30–40 | 0.475 | 0.3409 | 3 |
| pf | 18 | bs_port_fault | 60–80 | 0.4375 | 0.3095 | 0 |
| pf | 18 | interference_burst | 110–120 | 0.4205 | 0.25 | 0 |
| wmmse | 6 | ut_blockage | 30–40 | 0.45 | 0.4318 | 0 |
| wmmse | 6 | bs_port_fault | 60–80 | 0.4542 | 0.2143 | 0 |
| wmmse | 6 | interference_burst | 110–120 | 0.4068 | 0.1818 | 0 |
| wmmse | 12 | ut_blockage | 30–40 | 0.4667 | 0.4545 | 0 |
| wmmse | 12 | bs_port_fault | 60–80 | 0.4708 | 0.3095 | 0 |
| wmmse | 12 | interference_burst | 110–120 | 0.4477 | 0.2273 | 0 |
| wmmse | 18 | ut_blockage | 30–40 | 0.5 | 0.4318 | 4 |
| wmmse | 18 | bs_port_fault | 60–80 | 0.475 | 0.3452 | 0 |
| wmmse | 18 | interference_burst | 110–120 | 0.4568 | 0.3636 | 0 |
| queue_aware | 6 | ut_blockage | 30–40 | 0.325 | 0.2045 | 0 |
| queue_aware | 6 | bs_port_fault | 60–80 | 0.3042 | 0.09524 | 0 |
| queue_aware | 6 | interference_burst | 110–120 | 0.2727 | 0.04545 | 0 |
| queue_aware | 12 | ut_blockage | 30–40 | 0.3917 | 0.2955 | 0 |
| queue_aware | 12 | bs_port_fault | 60–80 | 0.375 | 0.2143 | 2 |
| queue_aware | 12 | interference_burst | 110–120 | 0.35 | 0.1364 | 0 |
| queue_aware | 18 | ut_blockage | 30–40 | 0.4667 | 0.3636 | 4 |
| queue_aware | 18 | bs_port_fault | 60–80 | 0.4292 | 0.3214 | 0 |
| queue_aware | 18 | interference_burst | 110–120 | 0.4182 | 0.2727 | 0 |
| drl | 6 | ut_blockage | 30–40 | 0.3667 | 0.2955 | 0 |
| drl | 6 | bs_port_fault | 60–80 | 0.3667 | 0.1667 | 0 |
| drl | 6 | interference_burst | 110–120 | 0.325 | 0.1136 | 0 |
| drl | 12 | ut_blockage | 30–40 | 0.4583 | 0.4091 | 3 |
| drl | 12 | bs_port_fault | 60–80 | 0.4375 | 0.2143 | 0 |
| drl | 12 | interference_burst | 110–120 | 0.3909 | 0.2045 | 0 |
| drl | 18 | ut_blockage | 30–40 | 0.4417 | 0.3409 | 4 |
| drl | 18 | bs_port_fault | 60–80 | 0.4083 | 0.2738 | 0 |
| drl | 18 | interference_burst | 110–120 | 0.3909 | 0.2273 | 0 |
| reliability_drl | 6 | ut_blockage | 30–40 | 0.4083 | 0.4318 | 0 |
| reliability_drl | 6 | bs_port_fault | 60–80 | 0.4208 | 0.1905 | 0 |
| reliability_drl | 6 | interference_burst | 110–120 | 0.3727 | 0.1591 | 0 |
| reliability_drl | 12 | ut_blockage | 30–40 | 0.4333 | 0.4091 | 0 |
| reliability_drl | 12 | bs_port_fault | 60–80 | 0.4375 | 0.2381 | 1 |
| reliability_drl | 12 | interference_burst | 110–120 | 0.4 | 0.2273 | 0 |
| reliability_drl | 18 | ut_blockage | 30–40 | 0.45 | 0.4545 | 2 |
| reliability_drl | 18 | bs_port_fault | 60–80 | 0.4458 | 0.2857 | 0 |
| reliability_drl | 18 | interference_burst | 110–120 | 0.4182 | 0.3182 | 0 |
| constrained_drl | 6 | ut_blockage | 30–40 | 0.4333 | 0.4091 | 0 |
| constrained_drl | 6 | bs_port_fault | 60–80 | 0.4292 | 0.1905 | 2 |
| constrained_drl | 6 | interference_burst | 110–120 | 0.3818 | 0.1818 | 0 |
| constrained_drl | 12 | ut_blockage | 30–40 | 0.45 | 0.4545 | 0 |
| constrained_drl | 12 | bs_port_fault | 60–80 | 0.4583 | 0.2619 | 1 |
| constrained_drl | 12 | interference_burst | 110–120 | 0.4159 | 0.2273 | 0 |
| constrained_drl | 18 | ut_blockage | 30–40 | 0.475 | 0.4773 | 0 |
| constrained_drl | 18 | bs_port_fault | 60–80 | 0.4708 | 0.3095 | 0 |
| constrained_drl | 18 | interference_burst | 110–120 | 0.4432 | 0.3409 | 0 |

### Service metrics, seed 42

Each batch is one scheduling interval with one packet per terminal; latency budgets count intervals. Availability uses a survival time of one interval.

| Method | Eb/N0 | Delivered | Within 1 | Within 2 | Within 4 | Availability | Survival violations | Energy (nJ/bit) | BLER p99 | Batches over 1 % BLER |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| static | 6 | 0.4453 | 0.4453 | 0.6714 | 0.8471 | 0.6734 | 73 | 75.82 | 1 | 0.9812 |
| static | 12 | 0.6078 | 0.6078 | 0.7909 | 0.9204 | 0.7922 | 53 | 70.58 | 1 | 0.8812 |
| static | 18 | 0.7375 | 0.7375 | 0.8648 | 0.9395 | 0.8656 | 32 | 67.27 | 0.7602 | 0.6375 |
| round_robin | 6 | 0.2672 | 0.2672 | 0.533 | 0.7452 | 0.5359 | 69 | 72.73 | 1 | 0.7188 |
| round_robin | 12 | 0.3609 | 0.3609 | 0.7217 | 0.8854 | 0.7234 | 53 | 67.73 | 1 | 0.4625 |
| round_robin | 18 | 0.4203 | 0.4203 | 0.8396 | 0.9459 | 0.8406 | 34 | 65.03 | 0.9557 | 0.3 |
| max_throughput | 6 | 0.3844 | 0.3844 | 0.6069 | 0.8089 | 0.6094 | 84 | 67.65 | 1 | 0.3563 |
| max_throughput | 12 | 0.45 | 0.45 | 0.6824 | 0.8758 | 0.6844 | 79 | 65.44 | 1 | 0.1625 |
| max_throughput | 18 | 0.4875 | 0.4875 | 0.7201 | 0.8981 | 0.7219 | 71 | 64.57 | 0.5 | 0.05 |
| pf | 6 | 0.2672 | 0.2672 | 0.533 | 0.7452 | 0.5359 | 69 | 72.73 | 1 | 0.7188 |
| pf | 12 | 0.3484 | 0.3484 | 0.695 | 0.8471 | 0.6969 | 49 | 67.79 | 1 | 0.4938 |
| pf | 18 | 0.4203 | 0.4203 | 0.8396 | 0.9459 | 0.8406 | 34 | 65.03 | 0.9557 | 0.3 |
| wmmse | 6 | 0.3891 | 0.3891 | 0.6116 | 0.8169 | 0.6141 | 85 | 65.81 | 1 | 0.375 |
| wmmse | 12 | 0.4391 | 0.4391 | 0.6682 | 0.8551 | 0.6703 | 76 | 54.31 | 1 | 0.2 |
| wmmse | 18 | 0.4578 | 0.4578 | 0.7013 | 0.8838 | 0.7031 | 76 | 47.17 | 0.5 | 0.1625 |
| queue_aware | 6 | 0.2594 | 0.2594 | 0.5126 | 0.7277 | 0.5156 | 70 | 69.45 | 1 | 0.7188 |
| queue_aware | 12 | 0.3531 | 0.3531 | 0.6965 | 0.8471 | 0.6984 | 50 | 63.59 | 1 | 0.4688 |
| queue_aware | 18 | 0.4203 | 0.4203 | 0.8302 | 0.9427 | 0.8313 | 40 | 60.69 | 0.941 | 0.3 |
| drl | 6 | 0.3141 | 0.3141 | 0.5613 | 0.8073 | 0.5641 | 96 | 70.62 | 1 | 0.5625 |
| drl | 12 | 0.3906 | 0.3906 | 0.6903 | 0.8997 | 0.6922 | 92 | 66.62 | 1 | 0.3625 |
| drl | 18 | 0.3969 | 0.3969 | 0.7186 | 0.8869 | 0.7203 | 76 | 66.46 | 1 | 0.3937 |
| reliability_drl | 6 | 0.3641 | 0.3641 | 0.5881 | 0.7978 | 0.5906 | 83 | 68.54 | 1 | 0.4313 |
| reliability_drl | 12 | 0.4031 | 0.4031 | 0.6022 | 0.8137 | 0.6047 | 78 | 63.86 | 1 | 0.3187 |
| reliability_drl | 18 | 0.4188 | 0.4188 | 0.6431 | 0.8471 | 0.6453 | 86 | 58.48 | 0.705 | 0.3125 |
| constrained_drl | 6 | 0.3641 | 0.3641 | 0.5692 | 0.7564 | 0.5719 | 74 | 71.93 | 1 | 0.4437 |
| constrained_drl | 12 | 0.4172 | 0.4172 | 0.6116 | 0.793 | 0.6141 | 71 | 67.63 | 1 | 0.2625 |
| constrained_drl | 18 | 0.4469 | 0.4469 | 0.6116 | 0.7659 | 0.6141 | 60 | 66.7 | 0.5 | 0.2062 |

