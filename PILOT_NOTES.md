# Pilot notes — CPU smoke test (27 Oct 2026)

**Purpose:** check that the pipeline works end to end, not to produce research results.
Both models are tiny (0.36–0.5B) and close to chance on MMLU-Pro, so none of these numbers
belongs in the paper's results section. They can go in the mid-term PPT as "progress so far".

Setup: 50 stratified MMLU-Pro items, original wording only (rewordings not generated yet),
2 option orders × (2 sampled seeds + 1 greedy) = 6 runs per item per model, 600 generations in
total. CPU (2 cores, float32), transformers backend, batch size 16. About 8 minutes per model.

## Reliability card (smoke test)

| Model | Greedy acc | Mean acc | Reliable acc | Seed flip | Order flip | Parse fail |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 12.0 [4, 22] | 17.0 [9.5, 25.5] | 4.0 [0, 10] | 12.0 [6, 19] | 20.0 [10, 32] | 46.3 |
| SmolLM2-360M-Instruct | 14.0 [4, 24] | 13.0 [7.5, 19] | 2.0 [0, 6] | 14.0 [8, 20] | 32.0 [20, 46] | 0.0 |

%, 95% bootstrap CI over items. Chance is about 11%.

## What the pilot caught

1. **"Explain first" behaviour.** Qwen2.5-0.5B ignores "letter only" in 44% of greedy runs and
   starts with "To solve this problem, we need to…". With `max_new_tokens = 8` this is a parse
   failure. The proposal flagged this as a pilot risk.
2. **An assistant prefill helps one model and hurts another.** Starting the assistant turn with
   `The answer is` (same text for every model) gave:

   | Model | Greedy acc, no prefill | with prefill | Parse fail, no prefill | with prefill |
   |---|---:|---:|---:|---:|
   | Qwen2.5-0.5B | 14% | 21% | 44% | 2% |
   | SmolLM2-360M | 16% | 6% | 0% | 0% |

   With the prefill, SmolLM2 answers "C" in 74% of runs (without it, "B" in 42%). The prefill
   moves its **label preference** rather than its knowledge — the effect Zheng et al. (2024)
   describe. One harmless-looking format choice reverses the order of the two models, which
   supports the paper's motivation.
3. **Label bias is visible already:** SmolLM2's order flip rate (32%) is more than twice its
   seed noise floor (14%).
4. **Parser safety:** a test caught that "I think…" would have been read as option I. Fixed:
   "letter + space + text" is not accepted.

## Decisions for the Kaggle pilot (M1) with the real 3B models

* Run `pilot_prefill_greedy` (50 items, greedy, with and without prefill) for all 5 ranking models.
* Rule, decided *before* seeing the main results: use no prefill unless at least one model has
  a parse-failure rate above 10%; in that case use the prefill for **all** models. Write the
  rule and the choice in the paper's method section, and report the comparison in an appendix.
* Time the largest model (Phi-3.5-mini) on 50 items × 30 runs to confirm the compute budget.
