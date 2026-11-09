# Reliability report

Runs: 20261109_212012_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_reliability_drl_constrained_drl_rayleigh_qpsk_s

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
| queue_aware | 0 | 1 | 2 | 0 |
| drl | 1 | 0 | 2 | 0 |
| reliability_drl | 2 | 1 | 0 | 0 |
| constrained_drl | 2 | 0 | 1 | 0 |

### Device classes, seed 42

Reliability within the class latency budget against the class target. *met*: the lower 95 % bound clears the target; *missed*: the upper bound is below it; *inconclusive*: the run has too few packets to decide (see *packets needed*).

| Method | Eb/N0 | Class | Budget | Target | Reliability | 95 % | Verdict | Packets | Packets needed | Survival violations |
|:--|--:|:--|--:|--:|--:|:--|:--|--:|--:|--:|
| static | 6 | Cobot motion control | 1 | 0.999999 | 0.6188 | 0.5387 – 0.6943 | missed | 160 | 3000000 | 9 |
| static | 6 | Mobile robot control | 2 | 0.99999 | 0.6981 | 0.6204 – 0.7683 | missed | 159 | 300001 | 7 |
| static | 6 | Worker safety wearable | 1 | 0.99999 | 0.1938 | 0.1356 – 0.2636 | missed | 160 | 300001 | 15 |
| static | 6 | Condition monitoring | 4 | 0.999 | 0.8344 | 0.7668 – 0.8889 | missed | 157 | 3000 | 2 |
| static | 12 | Cobot motion control | 1 | 0.999999 | 0.725 | 0.6489 – 0.7925 | missed | 160 | 3000000 | 5 |
| static | 12 | Mobile robot control | 2 | 0.99999 | 0.7673 | 0.6938 – 0.8306 | missed | 159 | 300001 | 4 |
| static | 12 | Worker safety wearable | 1 | 0.99999 | 0.425 | 0.3473 – 0.5055 | missed | 160 | 300001 | 20 |
| static | 12 | Condition monitoring | 4 | 0.999 | 0.8726 | 0.8101 – 0.9204 | missed | 157 | 3000 | 1 |
| static | 18 | Cobot motion control | 1 | 0.999999 | 0.85 | 0.7851 – 0.9015 | missed | 160 | 3000000 | 2 |
| static | 18 | Mobile robot control | 2 | 0.99999 | 0.8113 | 0.7417 – 0.8689 | missed | 159 | 300001 | 3 |
| static | 18 | Worker safety wearable | 1 | 0.99999 | 0.5687 | 0.4882 – 0.6467 | missed | 160 | 300001 | 8 |
| static | 18 | Condition monitoring | 4 | 0.999 | 0.8854 | 0.8249 – 0.9306 | missed | 157 | 3000 | 1 |
| round_robin | 6 | Cobot motion control | 1 | 0.999999 | 0.3375 | 0.2648 – 0.4164 | missed | 160 | 3000000 | 10 |
| round_robin | 6 | Mobile robot control | 2 | 0.99999 | 0.5975 | 0.5169 – 0.6744 | missed | 159 | 300001 | 10 |
| round_robin | 6 | Worker safety wearable | 1 | 0.99999 | 0.1625 | 0.109 – 0.229 | missed | 160 | 300001 | 11 |
| round_robin | 6 | Condition monitoring | 4 | 0.999 | 0.7834 | 0.7107 – 0.8451 | missed | 157 | 3000 | 7 |
| round_robin | 12 | Cobot motion control | 1 | 0.999999 | 0.4313 | 0.3533 – 0.5118 | missed | 160 | 3000000 | 6 |
| round_robin | 12 | Mobile robot control | 2 | 0.99999 | 0.7358 | 0.6602 – 0.8025 | missed | 159 | 300001 | 7 |
| round_robin | 12 | Worker safety wearable | 1 | 0.99999 | 0.2437 | 0.1794 – 0.3178 | missed | 160 | 300001 | 19 |
| round_robin | 12 | Condition monitoring | 4 | 0.999 | 0.9745 | 0.9361 – 0.993 | missed | 157 | 3000 | 2 |
| round_robin | 18 | Cobot motion control | 1 | 0.999999 | 0.4437 | 0.3653 – 0.5243 | missed | 160 | 3000000 | 4 |
| round_robin | 18 | Mobile robot control | 2 | 0.99999 | 0.8365 | 0.7697 – 0.8903 | missed | 159 | 300001 | 5 |
| round_robin | 18 | Worker safety wearable | 1 | 0.99999 | 0.3125 | 0.2417 – 0.3904 | missed | 160 | 300001 | 13 |
| round_robin | 18 | Condition monitoring | 4 | 0.999 | 0.9873 | 0.9547 – 0.9985 | missed | 157 | 3000 | 1 |
| max_throughput | 6 | Cobot motion control | 1 | 0.999999 | 0.4562 | 0.3774 – 0.5367 | missed | 160 | 3000000 | 15 |
| max_throughput | 6 | Mobile robot control | 2 | 0.99999 | 0.6352 | 0.5553 – 0.71 | missed | 159 | 300001 | 13 |
| max_throughput | 6 | Worker safety wearable | 1 | 0.99999 | 0.2188 | 0.1573 – 0.2909 | missed | 160 | 300001 | 14 |
| max_throughput | 6 | Condition monitoring | 4 | 0.999 | 0.7962 | 0.7246 – 0.8562 | missed | 157 | 3000 | 2 |
| max_throughput | 12 | Cobot motion control | 1 | 0.999999 | 0.5625 | 0.482 – 0.6407 | missed | 160 | 3000000 | 15 |
| max_throughput | 12 | Mobile robot control | 2 | 0.99999 | 0.6352 | 0.5553 – 0.71 | missed | 159 | 300001 | 17 |
| max_throughput | 12 | Worker safety wearable | 1 | 0.99999 | 0.3125 | 0.2417 – 0.3904 | missed | 160 | 300001 | 21 |
| max_throughput | 12 | Condition monitoring | 4 | 0.999 | 0.8854 | 0.8249 – 0.9306 | missed | 157 | 3000 | 3 |
| max_throughput | 18 | Cobot motion control | 1 | 0.999999 | 0.4938 | 0.4139 – 0.5738 | missed | 160 | 3000000 | 21 |
| max_throughput | 18 | Mobile robot control | 2 | 0.99999 | 0.6226 | 0.5424 – 0.6982 | missed | 159 | 300001 | 21 |
| max_throughput | 18 | Worker safety wearable | 1 | 0.99999 | 0.3625 | 0.2881 – 0.4421 | missed | 160 | 300001 | 21 |
| max_throughput | 18 | Condition monitoring | 4 | 0.999 | 0.9236 | 0.8703 – 0.9599 | missed | 157 | 3000 | 3 |
| pf | 6 | Cobot motion control | 1 | 0.999999 | 0.3375 | 0.2648 – 0.4164 | missed | 160 | 3000000 | 10 |
| pf | 6 | Mobile robot control | 2 | 0.99999 | 0.5975 | 0.5169 – 0.6744 | missed | 159 | 300001 | 10 |
| pf | 6 | Worker safety wearable | 1 | 0.99999 | 0.1625 | 0.109 – 0.229 | missed | 160 | 300001 | 11 |
| pf | 6 | Condition monitoring | 4 | 0.999 | 0.7834 | 0.7107 – 0.8451 | missed | 157 | 3000 | 7 |
| pf | 12 | Cobot motion control | 1 | 0.999999 | 0.3812 | 0.3057 – 0.4613 | missed | 160 | 3000000 | 7 |
| pf | 12 | Mobile robot control | 2 | 0.99999 | 0.7484 | 0.6736 – 0.8138 | missed | 159 | 300001 | 6 |
| pf | 12 | Worker safety wearable | 1 | 0.99999 | 0.2687 | 0.2018 – 0.3445 | missed | 160 | 300001 | 14 |
| pf | 12 | Condition monitoring | 4 | 0.999 | 0.9363 | 0.886 – 0.969 | missed | 157 | 3000 | 3 |
| pf | 18 | Cobot motion control | 1 | 0.999999 | 0.4437 | 0.3653 – 0.5243 | missed | 160 | 3000000 | 3 |
| pf | 18 | Mobile robot control | 2 | 0.99999 | 0.8742 | 0.8124 – 0.9214 | missed | 159 | 300001 | 6 |
| pf | 18 | Worker safety wearable | 1 | 0.99999 | 0.3125 | 0.2417 – 0.3904 | missed | 160 | 300001 | 13 |
| pf | 18 | Condition monitoring | 4 | 0.999 | 0.9745 | 0.9361 – 0.993 | missed | 157 | 3000 | 2 |
| wmmse | 6 | Cobot motion control | 1 | 0.999999 | 0.4813 | 0.4017 – 0.5615 | missed | 160 | 3000000 | 16 |
| wmmse | 6 | Mobile robot control | 2 | 0.99999 | 0.6226 | 0.5424 – 0.6982 | missed | 159 | 300001 | 11 |
| wmmse | 6 | Worker safety wearable | 1 | 0.99999 | 0.1875 | 0.1302 – 0.2567 | missed | 160 | 300001 | 13 |
| wmmse | 6 | Condition monitoring | 4 | 0.999 | 0.7834 | 0.7107 – 0.8451 | missed | 157 | 3000 | 5 |
| wmmse | 12 | Cobot motion control | 1 | 0.999999 | 0.5625 | 0.482 – 0.6407 | missed | 160 | 3000000 | 18 |
| wmmse | 12 | Mobile robot control | 2 | 0.99999 | 0.5346 | 0.4539 – 0.614 | missed | 159 | 300001 | 19 |
| wmmse | 12 | Worker safety wearable | 1 | 0.99999 | 0.3063 | 0.2359 – 0.3839 | missed | 160 | 300001 | 23 |
| wmmse | 12 | Condition monitoring | 4 | 0.999 | 0.8599 | 0.7956 – 0.9101 | missed | 157 | 3000 | 2 |
| wmmse | 18 | Cobot motion control | 1 | 0.999999 | 0.5125 | 0.4323 – 0.5922 | missed | 160 | 3000000 | 21 |
| wmmse | 18 | Mobile robot control | 2 | 0.99999 | 0.4969 | 0.4167 – 0.5771 | missed | 159 | 300001 | 19 |
| wmmse | 18 | Worker safety wearable | 1 | 0.99999 | 0.375 | 0.2998 – 0.4549 | missed | 160 | 300001 | 22 |
| wmmse | 18 | Condition monitoring | 4 | 0.999 | 0.9236 | 0.8703 – 0.9599 | missed | 157 | 3000 | 3 |
| queue_aware | 6 | Cobot motion control | 1 | 0.999999 | 0.3187 | 0.2474 – 0.397 | missed | 160 | 3000000 | 12 |
| queue_aware | 6 | Mobile robot control | 2 | 0.99999 | 0.5849 | 0.5042 – 0.6624 | missed | 159 | 300001 | 11 |
| queue_aware | 6 | Worker safety wearable | 1 | 0.99999 | 0.1625 | 0.109 – 0.229 | missed | 160 | 300001 | 11 |
| queue_aware | 6 | Condition monitoring | 4 | 0.999 | 0.7834 | 0.7107 – 0.8451 | missed | 157 | 3000 | 7 |
| queue_aware | 12 | Cobot motion control | 1 | 0.999999 | 0.4062 | 0.3294 – 0.4866 | missed | 160 | 3000000 | 9 |
| queue_aware | 12 | Mobile robot control | 2 | 0.99999 | 0.7233 | 0.6468 – 0.7912 | missed | 159 | 300001 | 8 |
| queue_aware | 12 | Worker safety wearable | 1 | 0.99999 | 0.2562 | 0.1906 – 0.3312 | missed | 160 | 300001 | 16 |
| queue_aware | 12 | Condition monitoring | 4 | 0.999 | 0.9363 | 0.886 – 0.969 | missed | 157 | 3000 | 3 |
| queue_aware | 18 | Cobot motion control | 1 | 0.999999 | 0.45 | 0.3714 – 0.5305 | missed | 160 | 3000000 | 4 |
| queue_aware | 18 | Mobile robot control | 2 | 0.99999 | 0.8553 | 0.7909 – 0.906 | missed | 159 | 300001 | 6 |
| queue_aware | 18 | Worker safety wearable | 1 | 0.99999 | 0.3187 | 0.2474 – 0.397 | missed | 160 | 300001 | 15 |
| queue_aware | 18 | Condition monitoring | 4 | 0.999 | 0.9745 | 0.9361 – 0.993 | missed | 157 | 3000 | 2 |
| drl | 6 | Cobot motion control | 1 | 0.999999 | 0.3688 | 0.2939 – 0.4485 | missed | 160 | 3000000 | 18 |
| drl | 6 | Mobile robot control | 2 | 0.99999 | 0.6101 | 0.5296 – 0.6863 | missed | 159 | 300001 | 16 |
| drl | 6 | Worker safety wearable | 1 | 0.99999 | 0.2062 | 0.1464 – 0.2773 | missed | 160 | 300001 | 16 |
| drl | 6 | Condition monitoring | 4 | 0.999 | 0.7707 | 0.697 – 0.8339 | missed | 157 | 3000 | 4 |
| drl | 12 | Cobot motion control | 1 | 0.999999 | 0.425 | 0.3473 – 0.5055 | missed | 160 | 3000000 | 19 |
| drl | 12 | Mobile robot control | 2 | 0.99999 | 0.6226 | 0.5424 – 0.6982 | missed | 159 | 300001 | 20 |
| drl | 12 | Worker safety wearable | 1 | 0.99999 | 0.2875 | 0.2188 – 0.3643 | missed | 160 | 300001 | 23 |
| drl | 12 | Condition monitoring | 4 | 0.999 | 0.9108 | 0.8549 – 0.9504 | missed | 157 | 3000 | 3 |
| drl | 18 | Cobot motion control | 1 | 0.999999 | 0.45 | 0.3714 – 0.5305 | missed | 160 | 3000000 | 9 |
| drl | 18 | Mobile robot control | 2 | 0.99999 | 0.805 | 0.7348 – 0.8635 | missed | 159 | 300001 | 15 |
| drl | 18 | Worker safety wearable | 1 | 0.99999 | 0.3125 | 0.2417 – 0.3904 | missed | 160 | 300001 | 18 |
| drl | 18 | Condition monitoring | 4 | 0.999 | 0.9682 | 0.9272 – 0.9896 | missed | 157 | 3000 | 2 |
| reliability_drl | 6 | Cobot motion control | 1 | 0.999999 | 0.3937 | 0.3175 – 0.474 | missed | 160 | 3000000 | 18 |
| reliability_drl | 6 | Mobile robot control | 2 | 0.99999 | 0.5094 | 0.4291 – 0.5895 | missed | 159 | 300001 | 20 |
| reliability_drl | 6 | Worker safety wearable | 1 | 0.99999 | 0.2188 | 0.1573 – 0.2909 | missed | 160 | 300001 | 13 |
| reliability_drl | 6 | Condition monitoring | 4 | 0.999 | 0.8153 | 0.7456 – 0.8727 | missed | 157 | 3000 | 2 |
| reliability_drl | 12 | Cobot motion control | 1 | 0.999999 | 0.2562 | 0.1906 – 0.3312 | missed | 160 | 3000000 | 23 |
| reliability_drl | 12 | Mobile robot control | 2 | 0.99999 | 0.4025 | 0.3256 – 0.4831 | missed | 159 | 300001 | 22 |
| reliability_drl | 12 | Worker safety wearable | 1 | 0.99999 | 0.4062 | 0.3294 – 0.4866 | missed | 160 | 300001 | 22 |
| reliability_drl | 12 | Condition monitoring | 4 | 0.999 | 0.9618 | 0.9187 – 0.9858 | missed | 157 | 3000 | 1 |
| reliability_drl | 18 | Cobot motion control | 1 | 0.999999 | 0.1875 | 0.1302 – 0.2567 | missed | 160 | 3000000 | 20 |
| reliability_drl | 18 | Mobile robot control | 2 | 0.99999 | 0.3836 | 0.3077 – 0.464 | missed | 159 | 300001 | 22 |
| reliability_drl | 18 | Worker safety wearable | 1 | 0.99999 | 0.475 | 0.3956 – 0.5553 | missed | 160 | 300001 | 16 |
| reliability_drl | 18 | Condition monitoring | 4 | 0.999 | 0.9299 | 0.8781 – 0.9645 | missed | 157 | 3000 | 2 |
| constrained_drl | 6 | Cobot motion control | 1 | 0.999999 | 0.4562 | 0.3774 – 0.5367 | missed | 160 | 3000000 | 16 |
| constrained_drl | 6 | Mobile robot control | 2 | 0.99999 | 0.434 | 0.3557 – 0.5148 | missed | 159 | 300001 | 18 |
| constrained_drl | 6 | Worker safety wearable | 1 | 0.99999 | 0.1625 | 0.109 – 0.229 | missed | 160 | 300001 | 13 |
| constrained_drl | 6 | Condition monitoring | 4 | 0.999 | 0.828 | 0.7597 – 0.8835 | missed | 157 | 3000 | 2 |
| constrained_drl | 12 | Cobot motion control | 1 | 0.999999 | 0.3625 | 0.2881 – 0.4421 | missed | 160 | 3000000 | 25 |
| constrained_drl | 12 | Mobile robot control | 2 | 0.99999 | 0.4025 | 0.3256 – 0.4831 | missed | 159 | 300001 | 24 |
| constrained_drl | 12 | Worker safety wearable | 1 | 0.99999 | 0.2812 | 0.2131 – 0.3577 | missed | 160 | 300001 | 24 |
| constrained_drl | 12 | Condition monitoring | 4 | 0.999 | 0.9236 | 0.8703 – 0.9599 | missed | 157 | 3000 | 2 |
| constrained_drl | 18 | Cobot motion control | 1 | 0.999999 | 0.6687 | 0.5901 – 0.741 | missed | 160 | 3000000 | 10 |
| constrained_drl | 18 | Mobile robot control | 2 | 0.99999 | 0.05031 | 0.02197 – 0.09673 | missed | 159 | 300001 | 5 |
| constrained_drl | 18 | Worker safety wearable | 1 | 0.99999 | 0.125 | 0.07805 – 0.1864 | missed | 160 | 300001 | 20 |
| constrained_drl | 18 | Condition monitoring | 4 | 0.999 | 1 | 0.9768 – 1 | inconclusive | 157 | 3000 | 0 |

### Disruptions, seed 42

Delivery is the share of terminals whose packet got through in an interval; recovery counts intervals after the event ends until a 3-interval mean is back at 95 % of the pre-event level.

| Method | Eb/N0 | Event | Batches | Delivery before | Delivery during | Recovery (intervals) |
|:--|--:|:--|:--|--:|--:|--:|
| static | 6 | ut_blockage | 30–40 | 0.5667 | 0.4318 | 0 |
| static | 6 | bs_port_fault | 60–80 | 0.5458 | 0 | 1 |
| static | 6 | interference_burst | 110–120 | 0.4568 | 0.06818 | 0 |
| static | 12 | ut_blockage | 30–40 | 0.8417 | 0.6136 | 4 |
| static | 12 | bs_port_fault | 60–80 | 0.7667 | 0 | 0 |
| static | 12 | interference_burst | 110–120 | 0.6364 | 0.3636 | 2 |
| static | 18 | ut_blockage | 30–40 | 0.9083 | 0.7045 | 4 |
| static | 18 | bs_port_fault | 60–80 | 0.8583 | 0.0119 | 0 |
| static | 18 | interference_burst | 110–120 | 0.7114 | 0.5909 | 0 |
| round_robin | 6 | ut_blockage | 30–40 | 0.3083 | 0.2955 | 0 |
| round_robin | 6 | bs_port_fault | 60–80 | 0.325 | 0.0119 | 1 |
| round_robin | 6 | interference_burst | 110–120 | 0.275 | 0.04545 | 0 |
| round_robin | 12 | ut_blockage | 30–40 | 0.4417 | 0.3409 | 15 |
| round_robin | 12 | bs_port_fault | 60–80 | 0.4125 | 0.1071 | 0 |
| round_robin | 12 | interference_burst | 110–120 | 0.3568 | 0.2955 | 0 |
| round_robin | 18 | ut_blockage | 30–40 | 0.475 | 0.3864 | 5 |
| round_robin | 18 | bs_port_fault | 60–80 | 0.4417 | 0.131 | 0 |
| round_robin | 18 | interference_burst | 110–120 | 0.3886 | 0.3409 | 0 |
| max_throughput | 6 | ut_blockage | 30–40 | 0.4833 | 0.4091 | 0 |
| max_throughput | 6 | bs_port_fault | 60–80 | 0.4667 | 0 | 1 |
| max_throughput | 6 | interference_burst | 110–120 | 0.3795 | 0.1136 | 0 |
| max_throughput | 12 | ut_blockage | 30–40 | 0.5 | 0.5 | 0 |
| max_throughput | 12 | bs_port_fault | 60–80 | 0.5 | 0.08333 | 0 |
| max_throughput | 12 | interference_burst | 110–120 | 0.4205 | 0.3636 | 0 |
| max_throughput | 18 | ut_blockage | 30–40 | 0.5 | 0.5 | 0 |
| max_throughput | 18 | bs_port_fault | 60–80 | 0.5 | 0.2143 | 0 |
| max_throughput | 18 | interference_burst | 110–120 | 0.4455 | 0.4773 | 0 |
| pf | 6 | ut_blockage | 30–40 | 0.3083 | 0.2955 | 0 |
| pf | 6 | bs_port_fault | 60–80 | 0.325 | 0.0119 | 1 |
| pf | 6 | interference_burst | 110–120 | 0.275 | 0.04545 | 0 |
| pf | 12 | ut_blockage | 30–40 | 0.4417 | 0.3182 | 3 |
| pf | 12 | bs_port_fault | 60–80 | 0.4125 | 0.08333 | 0 |
| pf | 12 | interference_burst | 110–120 | 0.3523 | 0.1591 | 0 |
| pf | 18 | ut_blockage | 30–40 | 0.4667 | 0.3636 | 5 |
| pf | 18 | bs_port_fault | 60–80 | 0.4333 | 0.1905 | 0 |
| pf | 18 | interference_burst | 110–120 | 0.3955 | 0.3636 | 0 |
| wmmse | 6 | ut_blockage | 30–40 | 0.4583 | 0.3864 | 0 |
| wmmse | 6 | bs_port_fault | 60–80 | 0.4417 | 0.0119 | 2 |
| wmmse | 6 | interference_burst | 110–120 | 0.3636 | 0.09091 | 0 |
| wmmse | 12 | ut_blockage | 30–40 | 0.4917 | 0.5 | 0 |
| wmmse | 12 | bs_port_fault | 60–80 | 0.4917 | 0.05952 | 0 |
| wmmse | 12 | interference_burst | 110–120 | 0.4091 | 0.2727 | 0 |
| wmmse | 18 | ut_blockage | 30–40 | 0.5 | 0.4773 | 0 |
| wmmse | 18 | bs_port_fault | 60–80 | 0.4958 | 0.1905 | 0 |
| wmmse | 18 | interference_burst | 110–120 | 0.4386 | 0.4091 | 0 |
| queue_aware | 6 | ut_blockage | 30–40 | 0.3083 | 0.2955 | 0 |
| queue_aware | 6 | bs_port_fault | 60–80 | 0.325 | 0.0119 | 1 |
| queue_aware | 6 | interference_burst | 110–120 | 0.2727 | 0.04545 | 0 |
| queue_aware | 12 | ut_blockage | 30–40 | 0.4417 | 0.3182 | 4 |
| queue_aware | 12 | bs_port_fault | 60–80 | 0.4083 | 0.08333 | 0 |
| queue_aware | 12 | interference_burst | 110–120 | 0.35 | 0.2955 | 0 |
| queue_aware | 18 | ut_blockage | 30–40 | 0.475 | 0.3636 | 5 |
| queue_aware | 18 | bs_port_fault | 60–80 | 0.4458 | 0.1905 | 0 |
| queue_aware | 18 | interference_burst | 110–120 | 0.4045 | 0.3182 | 0 |
| drl | 6 | ut_blockage | 30–40 | 0.4 | 0.3182 | 0 |
| drl | 6 | bs_port_fault | 60–80 | 0.3833 | 0 | 1 |
| drl | 6 | interference_burst | 110–120 | 0.3159 | 0.09091 | 0 |
| drl | 12 | ut_blockage | 30–40 | 0.475 | 0.3864 | 0 |
| drl | 12 | bs_port_fault | 60–80 | 0.4458 | 0.07143 | 0 |
| drl | 12 | interference_burst | 110–120 | 0.3795 | 0.2273 | 1 |
| drl | 18 | ut_blockage | 30–40 | 0.4667 | 0.3864 | 4 |
| drl | 18 | bs_port_fault | 60–80 | 0.45 | 0.1667 | 0 |
| drl | 18 | interference_burst | 110–120 | 0.3977 | 0.3636 | 0 |
| reliability_drl | 6 | ut_blockage | 30–40 | 0.45 | 0.3864 | 0 |
| reliability_drl | 6 | bs_port_fault | 60–80 | 0.4333 | 0 | 1 |
| reliability_drl | 6 | interference_burst | 110–120 | 0.3455 | 0.09091 | 0 |
| reliability_drl | 12 | ut_blockage | 30–40 | 0.475 | 0.4545 | 0 |
| reliability_drl | 12 | bs_port_fault | 60–80 | 0.4667 | 0.09524 | 0 |
| reliability_drl | 12 | interference_burst | 110–120 | 0.3909 | 0.25 | 0 |
| reliability_drl | 18 | ut_blockage | 30–40 | 0.45 | 0.4091 | 40 |
| reliability_drl | 18 | bs_port_fault | 60–80 | 0.4208 | 0.1786 | 0 |
| reliability_drl | 18 | interference_burst | 110–120 | 0.3773 | 0.2955 | 0 |
| constrained_drl | 6 | ut_blockage | 30–40 | 0.4417 | 0.4318 | 0 |
| constrained_drl | 6 | bs_port_fault | 60–80 | 0.4375 | 0 | 3 |
| constrained_drl | 6 | interference_burst | 110–120 | 0.3545 | 0.1136 | 0 |
| constrained_drl | 12 | ut_blockage | 30–40 | 0.5 | 0.5 | 0 |
| constrained_drl | 12 | bs_port_fault | 60–80 | 0.4958 | 0.05952 | 0 |
| constrained_drl | 12 | interference_burst | 110–120 | 0.4068 | 0.25 | 0 |
| constrained_drl | 18 | ut_blockage | 30–40 | 0.4833 | 0.5 | 0 |
| constrained_drl | 18 | bs_port_fault | 60–80 | 0.4917 | 0.2024 | 0 |
| constrained_drl | 18 | interference_burst | 110–120 | 0.4295 | 0.3636 | 0 |

### Service metrics, seed 42

Each batch is one scheduling interval with one packet per terminal; latency budgets count intervals. Availability uses a survival time of one interval.

| Method | Eb/N0 | Delivered | Within 1 | Within 2 | Within 4 | Availability | Survival violations | Energy (nJ/bit) | BLER p99 | Batches over 1 % BLER |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| static | 6 | 0.4641 | 0.4641 | 0.6399 | 0.7436 | 0.6422 | 41 | 69.95 | 0.9727 | 0.9688 |
| static | 12 | 0.6469 | 0.6469 | 0.7673 | 0.8455 | 0.7688 | 33 | 66.9 | 0.8176 | 0.7438 |
| static | 18 | 0.7594 | 0.7594 | 0.8255 | 0.8583 | 0.8266 | 14 | 65.26 | 0.7102 | 0.5125 |
| round_robin | 6 | 0.2766 | 0.2766 | 0.5519 | 0.6879 | 0.5547 | 45 | 67.95 | 1 | 0.6375 |
| round_robin | 12 | 0.3703 | 0.3703 | 0.7406 | 0.8631 | 0.7422 | 40 | 65.23 | 0.5955 | 0.4375 |
| round_robin | 18 | 0.4078 | 0.4078 | 0.8145 | 0.9013 | 0.8156 | 28 | 63.47 | 0.5102 | 0.3 |
| max_throughput | 6 | 0.3781 | 0.3781 | 0.5755 | 0.7325 | 0.5781 | 63 | 66.29 | 0.9307 | 0.3187 |
| max_throughput | 12 | 0.4359 | 0.4359 | 0.6478 | 0.836 | 0.65 | 73 | 64.38 | 0.3852 | 0.1562 |
| max_throughput | 18 | 0.4609 | 0.4609 | 0.6855 | 0.8726 | 0.6875 | 77 | 63.64 | 0.1352 | 0.1125 |
| pf | 6 | 0.2766 | 0.2766 | 0.5519 | 0.6879 | 0.5547 | 45 | 67.95 | 1 | 0.6375 |
| pf | 12 | 0.3609 | 0.3609 | 0.7201 | 0.828 | 0.7219 | 35 | 65 | 0.8025 | 0.4 |
| pf | 18 | 0.4141 | 0.4141 | 0.827 | 0.9076 | 0.8281 | 26 | 63.32 | 0.5 | 0.2938 |
| wmmse | 6 | 0.3672 | 0.3672 | 0.5582 | 0.7118 | 0.5609 | 61 | 65.6 | 0.9455 | 0.3625 |
| wmmse | 12 | 0.4172 | 0.4172 | 0.6226 | 0.8169 | 0.625 | 78 | 53.56 | 0.5852 | 0.2062 |
| wmmse | 18 | 0.4516 | 0.4516 | 0.6588 | 0.8392 | 0.6609 | 74 | 47.75 | 0.5102 | 0.1562 |
| queue_aware | 6 | 0.2703 | 0.2703 | 0.5393 | 0.6847 | 0.5422 | 48 | 65.24 | 1 | 0.6625 |
| queue_aware | 12 | 0.3609 | 0.3609 | 0.7186 | 0.8455 | 0.7203 | 41 | 62.19 | 0.8127 | 0.425 |
| queue_aware | 18 | 0.4172 | 0.4172 | 0.827 | 0.9124 | 0.8281 | 29 | 60.45 | 0.5 | 0.2875 |
| drl | 6 | 0.3234 | 0.3234 | 0.5456 | 0.7182 | 0.5484 | 72 | 66.61 | 0.9852 | 0.4875 |
| drl | 12 | 0.3812 | 0.3812 | 0.6572 | 0.8408 | 0.6594 | 82 | 64.91 | 1 | 0.3563 |
| drl | 18 | 0.4156 | 0.4156 | 0.7814 | 0.9061 | 0.7828 | 53 | 63.77 | 0.525 | 0.275 |
| reliability_drl | 6 | 0.35 | 0.35 | 0.5503 | 0.7086 | 0.5531 | 65 | 65.75 | 0.9307 | 0.4 |
| reliability_drl | 12 | 0.4016 | 0.4016 | 0.5959 | 0.7898 | 0.5984 | 75 | 60.15 | 0.625 | 0.3 |
| reliability_drl | 18 | 0.3875 | 0.3875 | 0.5582 | 0.7452 | 0.5609 | 69 | 53.73 | 0.5 | 0.4062 |
| constrained_drl | 6 | 0.3594 | 0.3594 | 0.5377 | 0.6847 | 0.5406 | 58 | 68.11 | 0.9852 | 0.3937 |
| constrained_drl | 12 | 0.4156 | 0.4156 | 0.5896 | 0.7834 | 0.5922 | 78 | 64.89 | 0.5 | 0.225 |
| constrained_drl | 18 | 0.4359 | 0.4359 | 0.5362 | 0.629 | 0.5391 | 38 | 67.34 | 0.5 | 0.2188 |

