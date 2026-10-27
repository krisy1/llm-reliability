| Model | greedy_acc | mean_acc | reliable_acc | worst_wording_acc | flip_seed | flip_wording | flip_order | parse_fail_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-0.5b | 12.0 [4.0, 22.0] | 17.0 [9.5, 25.5] | 4.0 [0.0, 10.0] | 17.0 [9.5, 25.5] | 12.0 [6.0, 19.0] | n/a | 20.0 [10.0, 32.0] | 46.3 [34.0, 59.0] |
| smollm2-360m | 14.0 [4.0, 24.0] | 13.0 [7.5, 19.0] | 2.0 [0.0, 6.0] | 13.0 [7.5, 19.0] | 14.0 [8.0, 20.0] | n/a | 32.0 [20.0, 46.0] | 0.0 [0.0, 0.0] |

All values in %, with 95% bootstrap CIs over 50 items (2000 resamples). Flip rates: share of run pairs that differ only in that factor whose correctness differs (wording/order: greedy cells; seed: sampled runs in the same cell).
