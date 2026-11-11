# Beyond Single-Run Accuracy: Measuring Reliability and Ranking Stability of Small Open-Weight Language Models Under Seed, Wording, and Order Variations

**Krish · 24csu320 · CSC301 · The Northcap University · 2026**  
**Repository:** `krisy1/llm-reliability` (`commit 1dd5bb8`)

Standard LLM leaderboards report one accuracy number per model, usually obtained from a single greedy run (`w0/o0`). That score hides whether a model will give the same answer when the sampling seed, prompt wording, or ordering of options or tools changes. 

This project evaluates five open-weight instruction-tuned models in the 2–4B parameter range—**Qwen2.5-3B**, **Llama-3.2-3B**, **Phi-3.5-mini**, **SmolLM3-3B**, and **Granite-3.3-2B**—on **300 stratified MMLU-Pro questions** and **100 BFCL v4 "multiple" function-calling items**. Each item is evaluated 12 times per model in a fully crossed `s2` factorial design: 8 sampled runs ($2\text{ seeds} \times 2\text{ wordings} \times 2\text{ orders}$) plus 4 greedy runs (one per wording–order cell), totaling **24,000 main-run generations** (24,500 including the M1 prefill pilot).

---

## Key Empirical Findings

* **Reliability Gap (RQ1 / H1):** On MMLU-Pro, **reliable accuracy** (answering correctly in all 8 sampled conditions) is **17.4–24.2 percentage points (pp)** below mean sampled accuracy for every model. Even the strongest model answers only 1 in 5 MMLU-Pro items consistently across all conditions. On BFCL, four models exceed 89% mean accuracy (near ceiling), while SmolLM3-3B drops by 18.0 pp (from 89.0% mean to 71.0% reliable accuracy).
* **Factor Attribution (RQ2 / H2):** On MMLU-Pro, changing option order flips correctness on **22.0%–26.0%** of items—more than twice the rate caused by rewording (**8.0%–11.0%**) with non-overlapping 95% CIs. Furthermore, moving the gold option changes the selected answer letter on **43.2%–61.7%** of greedy questions. On BFCL, wording flips (3.5%–8.0%) and tool-order flips (2.5%–11.0%) have overlapping intervals.
* **Ranking Instability & Reversals (RQ3 / H3):** Replacing greedy accuracy with reliable accuracy on MMLU-Pro changes **4 of 10 pairwise model orderings** (Kendall's $\tau = 0.20$). **Llama-3.2-3B** ranks 2nd by greedy accuracy (40.7%) but drops to **last place** by reliable accuracy (11.3%), whereas **Qwen2.5-3B** rises from 3rd (37.0%) to **1st** (20.0%), leading Llama by **+8.7 pp** (95% paired bootstrap CI `[4.7, 13.0]`). On BFCL, three models tie at 94.0% greedy accuracy, but reliable accuracy separates **Granite-3.3-2B** (86.0%) from **SmolLM3-3B** (71.0%) by **+15.0 pp** (`[7.0, 23.0]`).
* **Evaluation Budget (RQ4 / H4):** A lightweight $1 \times 2 \times 2$ crossed design (1 seed, 2 wordings, 2 orders; 4 runs/item) agrees with a held-out reference ranking on **85.9%** of MMLU-Pro model pairs and **92.8%** of BFCL model pairs (compared to 76.9% and 84.2% for a single run).

---

## Evaluated Models

All five models were evaluated in `float16` using the `transformers` backend on Kaggle GPUs, with each model's native chat template, no system prompt, and pinned Hugging Face revisions:

| Short Name | Hugging Face Model | Pinned Revision | Notes |
| :--- | :--- | :---: | :--- |
| `qwen2.5-3b` | `Qwen/Qwen2.5-3B-Instruct` | `aa8e725` | Triggered M1 prefill rule in pilot |
| `llama3.2-3b` | `unsloth/Llama-3.2-3B-Instruct` | `006f5dc` | Ungated mirror of Meta release |
| `phi3.5-mini` | `microsoft/Phi-3.5-mini-instruct` | `2fe1924` | Standard chat template |
| `smollm3-3b` | `HuggingFaceTB/SmolLM3-3B` | `a07cc9a` | `enable_thinking: false` |
| `granite3.3-2b` | `ibm-granite/granite-3.3-2b-instruct` | `707f574` | Standard chat template |

*(Two sub-billion-parameter models, `Qwen2.5-0.5B-Instruct` and `SmolLM2-360M-Instruct`, were used strictly for CPU end-to-end smoke testing and are excluded from all reported rankings.)*

---

## Summary Reliability Cards

Values are percentages with 95% percentile intervals from 2,000 paired bootstrap resamples over items.

### Table 1: MMLU-Pro Reliability Card ($N = 300$, 12 runs/item)

| Model | Greedy Acc (`w0/o0`) | Mean Sampled Acc | Reliable Acc (All 8) | Gap (Mean $-$ Rel.) | Worst-Wording Acc | Seed Flip % | Wording Flip % | Order Flip % | Parse-Failure % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phi-3.5-mini** | **41.7** `[36.3, 47.3]` | **41.1** `[36.7, 45.5]` | 18.7 `[14.3, 23.3]` | 22.5 `[19.1, 26.1]` | **40.7** `[35.8, 44.9]` | 10.9 `[8.8, 13.1]` | **8.0** `[5.8, 10.5]` | **22.0** `[17.8, 26.5]` | 4.3 `[2.9, 5.8]` |
| **Llama-3.2-3B** | 40.7 `[35.3, 46.3]` | 35.6 `[31.7, 39.3]` | 11.3 `[7.7, 14.7]` | 24.2 `[21.4, 27.2]` | 35.3 `[31.1, 38.8]` | 20.3 `[17.8, 22.8]` | 10.0 `[7.7, 12.7]` | 24.7 `[20.7, 29.0]` | 0.0 `[0.0, 0.1]` |
| **Qwen2.5-3B** | 37.0 `[31.3, 42.7]` | 37.4 `[32.8, 41.7]` | **20.0** `[15.7, 24.3]` | **17.4** `[14.6, 20.2]` | 37.0 `[32.2, 41.3]` | **4.3** `[3.1, 5.8]` | 9.0 `[6.7, 11.3]` | 24.7 `[20.3, 29.0]` | **0.0** `[0.0, 0.0]` |
| **SmolLM3-3B** | 34.0 `[28.7, 39.3]` | 34.8 `[30.8, 39.1]` | 15.0 `[11.0, 19.0]` | 19.8 `[16.8, 22.9]` | 34.3 `[30.1, 38.3]` | 11.6 `[9.7, 13.5]` | 9.7 `[7.2, 12.2]` | 22.7 `[18.8, 27.0]` | 3.5 `[2.4, 4.9]` |
| **Granite-3.3-2B** | 30.7 `[25.7, 36.0]` | 33.3 `[29.1, 37.6]` | 13.7 `[9.7, 17.3]` | 19.7 `[16.6, 22.8]` | 32.0 `[27.7, 36.4]` | 7.3 `[5.8, 8.9]` | 11.0 `[8.3, 14.0]` | 26.0 `[21.5, 30.7]` | 0.4 `[0.1, 0.9]` |

### Table 2: BFCL v4 "Multiple" Reliability Card ($N = 100$, 12 runs/item)

| Model | Greedy Acc (`w0/o0`) | Mean Sampled Acc | Reliable Acc (All 8) | Gap (Mean $-$ Rel.) | Worst-Wording Acc | Seed Flip % | Wording Flip % | Order Flip % | Parse-Failure % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-3B** | **97.0** `[93.0, 100.0]` | **95.2** `[91.5, 98.1]` | **89.0** `[83.0, 94.0]` | **6.2** `[2.6, 10.6]` | **94.8** `[90.2, 97.8]` | **1.0** `[0.2, 2.0]` | **3.5** `[1.0, 7.0]` | **2.5** `[0.5, 5.5]` | **0.0** `[0.0, 0.0]` |
| **Granite-3.3-2B** | 94.0 `[89.0, 98.0]` | 92.5 `[87.9, 96.6]` | 86.0 `[79.0, 93.0]` | 6.5 `[3.0, 10.6]` | 92.5 `[87.0, 96.0]` | 2.0 `[0.8, 3.8]` | 4.0 `[1.0, 8.0]` | 4.0 `[1.0, 8.0]` | **0.0** `[0.0, 0.0]` |
| **Phi-3.5-mini** | 94.0 `[89.0, 98.0]` | 92.0 `[87.6, 95.9]` | 82.0 `[75.0, 89.0]` | 10.0 `[5.4, 14.7]` | 90.8 `[85.5, 95.2]` | 3.0 `[1.0, 5.5]` | 5.0 `[1.5, 9.0]` | 5.0 `[2.0, 9.0]` | **0.0** `[0.0, 0.0]` |
| **SmolLM3-3B** | 94.0 `[89.0, 98.0]` | 89.0 `[84.5, 93.3]` | 71.0 `[62.0, 80.0]` | 18.0 `[11.7, 24.5]` | 89.0 `[83.2, 92.5]` | 6.0 `[3.5, 8.8]` | 5.5 `[2.0, 9.5]` | 7.5 `[3.5, 12.5]` | 0.7 `[0.0, 1.8]` |
| **Llama-3.2-3B** | 47.0 `[38.0, 57.0]` | 43.8 `[35.4, 52.2]` | 30.0 `[21.0, 39.0]` | 13.8 `[9.1, 18.6]` | 40.5 `[31.8, 49.0]` | 9.0 `[5.5, 13.0]` | 8.0 `[4.0, 12.5]` | 11.0 `[6.0, 16.5]` | 1.1 `[0.0, 3.2]` |

*(Note: On BFCL, Llama-3.2-3B fails primarily due to argument type mismatches—passing stringified primitives such as `"5"` instead of `5`—producing `wrong_type` in 49.2% of greedy runs and 48.9% of sampled runs.)*

---

## Pipeline Architecture & Reproduction

```text
sample_mmlu / sample_bfcl  ->  make_variants orders  ->  make_variants reword  ->  [human check]  ->  make_variants finalize
                                                                                                           |
       analyse  <-  score_runs  <-  run_generation (per model)  <-  build_manifest  <----------------------+
