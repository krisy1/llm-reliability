| Model | greedy_acc | mean_acc | reliable_acc | worst_wording_acc | flip_seed | flip_wording | flip_order | parse_fail_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-0.5b | 60.0 [40.0, 80.0] | 60.0 [42.5, 77.5] | 45.0 [25.0, 70.0] | 60.0 [42.5, 77.5] | n/a | n/a | 25.0 [10.0, 45.0] | 1.2 [0.0, 3.8] |

All values in %, with 95% bootstrap CIs over 20 items (2000 resamples). Flip rates: share of run pairs that differ only in that factor whose correctness differs (wording/order: greedy cells; seed: sampled runs in the same cell).
