# Beyond Single-Run Accuracy — reliability of small LLMs under seed, wording and order

CSC301 Seminar project. Every test item is run under every combination of
**seed × wording × option order**, so each item's answer flips can be attributed to a factor,
and we can check whether model rankings survive when "accuracy" is replaced by "reliability".

Current status: **MMLU-Pro pipeline complete and tested end-to-end** (CPU smoke pilot with two
tiny models, 600 generations; see `PILOT_NOTES.md`). BFCL function calling is the next module.
Prefill comparison: `python -m src.run_generation ... --prefill "The answer is" --tag _prefill`.

## Pipeline

```
sample_mmlu  ->  make_variants orders  ->  make_variants reword  ->  [human check]  ->  make_variants finalize
                                                                                              |
analyse  <-  score_runs  <-  run_generation (one call per model)  <-  build_manifest  <-------+
```

| Step | Command | Output |
|---|---|---|
| 1. Sample items | `python -m src.sample_mmlu --n 300` | `data/sampled/mmlu_pro_n300.jsonl` + exclusion report |
| 2. Option reshuffles | `python -m src.make_variants orders --items data/sampled/mmlu_pro_n300.jsonl` | one fixed reshuffle per item (recorded) |
| 3. Rewordings | `python -m src.make_variants reword --items ... --n-rewordings 2` | `data/reworded/*_review.csv` |
| 4. Human check | open the CSV, set `human_ok` to 1 or 0 for every row | |
| 5. Finalize | `python -m src.make_variants finalize --items ...` | validated wordings; failed items dropped and counted |
| 6. Manifest | `python -m src.build_manifest --items ...` | one row per planned run |
| 7. Generate | `python -m src.run_generation --model qwen2.5-3b --manifest data/manifests/<name>.jsonl [--backend vllm]` | raw outputs (resumable) |
| 8. Score | `python -m src.score_runs --manifest data/manifests/<name>.jsonl` | `results/scored/<name>.csv` |
| 9. Analyse | `python -m src.analyse --scored results/scored/<name>.csv --ranking-only` | reliability card, pairwise table, budget curve, figures 1–4 |

Run the tests with `python -m pytest -q tests` (parser, reshuffles, rewording pipeline, and
every metric checked on synthetic models with known answers).

## Rewording API (step 3)

Any OpenAI-compatible free tier works. Keys are read from the environment only:

```bash
export REWORD_API_BASE=https://api.groq.com/openai/v1   # or Gemini / OpenRouter OpenAI-compatible URL
export REWORD_API_KEY=...                               # never commit this
export REWORD_MODEL=llama-3.3-70b-versatile
```

Automatic checks flag: changed numbers, changed blanks (`____`), length ratio outside 0.6–1.7,
identical text, answer-option text leaking into the question, mentions of option letters.
Flagged rows still go to the human reviewer. An item is dropped if **any** of its rewordings
fails, so the design stays fully crossed. Report the drop count in the paper.

## Design decisions to state in the paper

* **Item exclusion before sampling.** Items whose options refer to other options by position
  ("all of the above", "A and B", "option C") or contain duplicate option texts are excluded,
  because reshuffling would change or blur the correct answer. On the MMLU-Pro test split
  this removes 176 of 12,032 questions (158 positional, 18 duplicates). The rule is
  conservative: some logic/code questions that use capital letters as variables are also removed.
* **Reshuffle always moves the gold answer** to a different letter, so the order factor
  always changes the correct label.
* **One prompt template, no system prompt**, the model's own chat template, and
  `max_new_tokens = 8`. SmolLM3's thinking mode is switched off (`enable_thinking: false`).
* **Parser** accepts only formats that name one letter unambiguously (`C`, `(C)`, `C.`,
  `C. text`, `Answer: C`, `The answer is C`). `I think…` is *not* read as option I.
  Parse failures count as wrong and are reported separately. A diagnostic column flags outputs
  like `B. <text>` whose text belongs to a different option.
* **Seeds.** vLLM gives every request its own seed. The transformers backend sets
  `torch.manual_seed(seed)` per seed group with a fixed batch size; batch size, dtype, device
  and resolved model revision are logged with every output.
* **Statistics.** Every metric is reduced to one value per item and bootstrapped over
  **items** with shared resamples across models (paired differences). Never over runs.
* **Budget curve.** Seeds are split in half. Small designs draw seeds from the first half;
  the reference ranking uses only the second half (all wordings × orders), so the reference
  never contains the design's runs. Wording and order *levels* are shared, which the paper
  should state as a limitation.
* **Precision.** All models at float16 on GPU (T4 has no bfloat16). The CPU smoke test uses float32.

## Metrics

| Metric | Definition |
|---|---|
| Greedy accuracy | one greedy run, original wording and order (the leaderboard number) |
| Mean accuracy | mean correctness over all sampled runs |
| Reliable accuracy | share of items correct in **every** sampled run |
| Worst-wording accuracy | lowest of the per-wording mean accuracies |
| Flip rate (wording / order) | over greedy cells that differ only in that factor: how often correctness changes |
| Flip rate (seed) | over sampled runs in the same wording × order cell (noise floor) |
| Answer-change rate | as flip rate, but on the chosen option *text* (also catches wrong→different wrong) |
| Parse-failure rate | share of runs with no parseable answer |

## Compute plan

Full MMLU-Pro run: 300 items × 30 runs = 9,000 generations per model. Time your largest model
on Kaggle first (`--backend vllm` if it installs, else transformers with `--batch-size 32`).
`results/logs/timing.jsonl` records wall time per run.

## Licences

* MMLU-Pro (TIGER-Lab): MIT licence.
* BFCL (Gorilla, UC Berkeley): Apache 2.0.
* Models: each under its own licence (Qwen: Apache 2.0 / Qwen licence; Llama 3.2: Llama 3.2
  Community Licence; Phi-3.5: MIT; SmolLM: Apache 2.0; Granite: Apache 2.0). Verify on each
  model card and record the exact revision used.
