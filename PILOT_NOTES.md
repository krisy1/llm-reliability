---

## 2. `PILOT_NOTES.md`

```markdown
# Pilot & Execution Notes (Completed Record)

This file documents the pre-main-run verification pilots, the M1 prefill trigger decision, and the final `s2` factorial execution parameters reported in the paper (`commit 1dd5bb8`).

---

## 1. CPU Smoke Pilot (Pipeline Verification Only)

Before running GPU experiments on Kaggle, the end-to-end pipeline was verified on CPU (`float32`) using two sub-billion-parameter models (`Qwen/Qwen2.5-0.5B-Instruct` and `HuggingFaceTB/SmolLM2-360M-Instruct`) across 50 MMLU-Pro items (`smoke_n50_w0_o2_s2.jsonl`) and 20 BFCL items (`smoke_bfcl20.jsonl`).

* **MMLU-Pro Smoke Outcome ($N=50$, 6 runs/item):**
  * `qwen2.5-0.5b`: 12.0% greedy accuracy, 17.0% mean accuracy, 4.0% reliable accuracy, 46.3% parse failure rate (conversational preambles truncated at 8 tokens).
  * `smollm2-360m`: 14.0% greedy accuracy, 13.0% mean accuracy, 2.0% reliable accuracy, 0.0% parse failure rate, and a **90.0% order answer-change rate** (exhibiting strong positional bias toward option letter `"B"` in 42% of unprefilled runs).
* **Exclusion from Main Results:** Because both sub-billion models perform near the ~10% random-guess baseline on 10-option MMLU-Pro, they are excluded from all main rankings (`--ranking-only`).

---

## 2. M1 Prefill Pilot & Global Trigger Decision

Per Section 6.4 of the paper, we pre-specified rule **M1**: if any 2–4B ranking model exceeded a **10% parse-failure rate** on a 50-item greedy MMLU-Pro pilot (`pilot_prefill_greedy.jsonl`, 100 runs per model comparing no prefill vs. assistant prefill `"The answer is"`), the prefill `"The answer is"` would be applied uniformly to all models in the main runs.

### Empirical Pilot Outcome (Kaggle GPU, `float16`)
* **Without prefill:** `qwen2.5-3b` failed to parse on **100.0%** of pilot items because its conversational preamble (`"To find the correct..."`) exceeded the 8-token generation budget before naming an option letter.
* **With prefill (`"The answer is"`):** Parse failures dropped to **0.0%–5.0%** across all five models:
  * `qwen2.5-3b`: 39.0% greedy acc, 0.0% parse fail (wall time: 103.4s)
  * `phi3.5-mini`: 38.0% greedy acc, 5.0% parse fail (wall time: 84.9s)
  * `smollm3-3b`: 38.0% greedy acc, 4.0% parse fail (wall time: 70.5s)
  * `llama3.2-3b`: 36.0% greedy acc, 0.0% parse fail (wall time: 60.0s)
  * `granite3.3-2b`: 33.0% greedy acc, 0.0% parse fail (wall time: 62.8s)
* **Decision Locked:** Rule M1 was triggered. All main MMLU-Pro evaluations were executed with `--prefill "The answer is"`.

---

## 3. Main Evaluation Matrix (`s2` Balanced Factorial Design)

To remain within Kaggle's ~30-hour weekly GPU quota while preserving a fully crossed factorial design across all five models and both benchmarks, the main evaluation used the balanced **`s2` design**:
* **Factors:** 2 sampled seeds (`0, 1` at $T=0.7, p=0.95$) + greedy $\times$ 2 wordings (`w0, w1`) $\times$ 2 orders (`o0, o1`) = **12 runs per item per model** (8 sampled + 4 greedy).
* **Total Generations Executed:**
  * MMLU-Pro ($N=300$): $300 \times 12 \times 5 = 18,000$ runs
  * BFCL v4 "multiple" ($N=100$): $100 \times 12 \times 5 = 6,000$ runs
  * M1 Prefill Pilot ($N=50$): $50 \times 2 \times 5 = 500$ runs
  * **Grand Total:** **24,500 GPU generations**
