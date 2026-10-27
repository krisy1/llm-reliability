# Kaggle run — step by step

## Before Kaggle (on your laptop)
1. Rewordings (free Groq key; see README):
   ```bash
   export REWORD_API_BASE=https://api.groq.com/openai/v1 REWORD_API_KEY=... \
          REWORD_MODEL=openai/gpt-oss-120b REWORD_REASONING_EFFORT=low REWORD_MIN_INTERVAL=6
   python -m src.sample_mmlu --n 300
   python -m src.make_variants reword --items data/sampled/mmlu_pro_n300.jsonl --n-rewordings 2
   python -m src.sample_bfcl
   python -m src.make_variants reword --items data/sampled/bfcl_multiple_n200.jsonl --n-rewordings 2
   ```
   If the daily limit is hit the script stops; rerun the same command the next day.
2. Open each `data/reworded/*_review.csv`, set `human_ok` to 1 or 0 for every row (look first at rows with `auto_issues`).
3. `python -m src.make_variants finalize --items <same items file>` for both files.
4. `git add data/ && git commit -m "finalized wordings" && git push`.

## One-time Kaggle setup
5. Kaggle account, phone verified (needed for GPU and internet).
6. Hugging Face: accept the Llama-3.2 licence on its model page; create a read token.
7. GitHub: fine-grained token, read-only, only for `krisy1/llm-reliability`.
8. New notebook → File → Import notebook → `notebooks/kaggle_runner.ipynb`.
   Right panel: Accelerator **GPU T4 x2**, Internet **on**.
   Add-ons → Secrets: `HF_TOKEN`, `GH_TOKEN` (tick "attach" for this notebook).

## Each session
9. Run cells 0–3 (secrets, clone, install, optional vLLM, sampling + full manifest).
10. Cell 4 (M1): timing of phi3.5-mini on 50 items, then the prefill pilot. Apply the rule:
    prefill for all models only if a 3B model's parse-failure rate is above 10%. If yes, set
    `assistant_prefill: "The answer is"` in `configs/experiment.yaml`, commit, and re-clone.
11. Cell 5: one model at a time. Long runs: **Save Version → Save & Run All** (runs in the
    background, up to 12 h). Generation is resumable.
12. Two GPUs: run two models at once in separate terminals/cells with
    `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`.
13. Cells 6, then BFCL (N=100 first, then 200).
14. Last cell zips `results/` + manifests. Download `results.zip` from the version's Output tab,
    or "New Dataset" from the output so the next session can continue from it.

## Budget
Weekly GPU quota is about 30 h; check the timing log after step 10 and plan per model. Order of
priority for the 18 Nov minimum: MMLU-Pro for 4 models (all factors) → BFCL 100-item pilot → rest.
