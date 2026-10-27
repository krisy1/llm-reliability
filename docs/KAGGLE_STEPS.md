# Kaggle run guide — LLM reliability study

This guide takes the study from "pipeline tested on CPU" to "all GPU runs finished and analysed".
It has seven parts: A (laptop: rewordings), B (accounts and tokens, once), C (notebook setup, once),
D (first GPU session: timing + prefill check), E (full MMLU-Pro runs, several sessions),
F (BFCL), G (final analysis and download). Troubleshooting and a record-keeping list are at the end.

Kaggle limits to plan around (check your own quota under Profile → Settings, it can change):
- GPU quota about 30 hours per week, sometimes more; resets weekly ([Kaggle GPU tips](https://www.kaggle.com/docs/efficient-gpu-usage)).
- One session runs at most 12 hours, interactive or "Save & Run All" ([Kaggle notebooks docs](https://www.kaggle.com/docs/notebooks)).
- Interactive sessions stop after about 20 minutes idle; only `/kaggle/working` (20 GB) is saved.
- An interactive session left open while a saved version runs uses quota too, so stop it.

Two kinds of session:
- **Interactive** (you click cells): only for short checks. Uses quota while open.
- **Save Version → Save & Run All** (a "version"/commit): runs top to bottom in the background,
  you can close the browser, Kaggle emails you when it ends. Use this for every real run.

---

## Part A — On your laptop: rewordings (start now; takes 3–5 days on the free API cap)

The GPU work in Part D does not need the rewordings, so do Part B–D while this runs.

A1. Get the code and install it:
```bash
git clone https://github.com/krisy1/llm-reliability.git
cd llm-reliability
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests                                # should say 90 passed
```

A2. Create a free Groq API key at console.groq.com → API Keys. No card needed. Never commit it.

A3. Set the rewording settings (once per terminal):
```bash
export REWORD_API_BASE=https://api.groq.com/openai/v1
export REWORD_API_KEY=gsk_...
export REWORD_MODEL=openai/gpt-oss-120b
export REWORD_REASONING_EFFORT=low
export REWORD_MIN_INTERVAL=6
```
Windows PowerShell: `$env:REWORD_API_KEY="gsk_..."` (same for the others).

A4. Sample the items (deterministic; Kaggle will produce the identical files):
```bash
python -m src.sample_mmlu --n 300
python -m src.sample_bfcl            # all 200; the 100-item pilot is a subset made on Kaggle
python -m src.sample_bfcl --n 100    # also make the pilot file locally, so you can reword it first
```

A5. Reword. Do the BFCL pilot file first (smallest, needed first), then MMLU-Pro, then the rest of BFCL:
```bash
python -m src.make_variants reword --items data/sampled/bfcl_multiple_n100.jsonl --n-rewordings 2
python -m src.make_variants reword --items data/sampled/mmlu_pro_n300.jsonl      --n-rewordings 2
python -m src.make_variants reword --items data/sampled/bfcl_multiple_n200.jsonl --n-rewordings 2
```
- About 200 + 600 + 400 calls. If the daily limit is reached the script prints
  "Daily free-tier limit reached … rerun tomorrow" and stops. Run the same command the next day;
  finished rows are skipped.
- Note: the 200-item BFCL file contains the 100 pilot items, but they are reworded again into a separate
  review file. If you want identical wordings for the overlap, copy those rows from the n100 review CSV
  into the n200 review CSV before running the n200 command (the script skips rows already present).

A6. Human check. Open each `data/reworded/*_review.csv` in a spreadsheet:
- Sort by `auto_issues`; look at flagged rows first (numbers changed, value missing, length ratio, …).
- For EVERY row put `human_ok` = 1 (same meaning, same correct answer, natural) or 0.
- Use `notes` for the reason when you put 0. Keep a count; the paper reports it.
- Save as CSV (keep the same columns and file name).

A7. Finalize (drops items with any rejected rewording and prints the count):
```bash
python -m src.make_variants finalize --items data/sampled/mmlu_pro_n300.jsonl
python -m src.make_variants finalize --items data/sampled/bfcl_multiple_n100.jsonl
python -m src.make_variants finalize --items data/sampled/bfcl_multiple_n200.jsonl
```
This writes `data/reworded/<stem>_wordings.jsonl`.

A8. Commit and push so Kaggle can get them:
```bash
git add data/ && git commit -m "Finalized wordings (MMLU-Pro n300, BFCL n100)" && git push
```
Kaggle checks for these files: until they exist, the notebook only runs the Part D pilot.

---

## Part B — Accounts and tokens (once)

B1. Kaggle: create an account and verify your phone number (Settings → Phone verification).
Without it the GPU and Internet options are greyed out.

B2. Hugging Face (needed for Llama-3.2, which is gated):
- Log in, open huggingface.co/meta-llama/Llama-3.2-3B-Instruct, accept the licence. Approval is
  usually quick, but wait for the "access granted" email before the first run.
- Settings → Access Tokens → Create new token → type "Read". Copy it.

B3. GitHub token for the private repo:
- GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate.
- Repository access: "Only select repositories" → `krisy1/llm-reliability`.
- Permissions: Repository → Contents → Read-only. Expiry: end of December.
- Copy the token (starts with `github_pat_`).

---

## Part C — Notebook setup (once)

C1. Download `notebooks/kaggle_runner.ipynb` from GitHub (open the file → "Download raw file").

C2. Kaggle → Create → New Notebook → File → Import Notebook → upload the file.
Rename the notebook (e.g. `llm-reliability-runner`).

C3. Right-hand panel → Session options:
- Accelerator: **GPU T4 x2**.
- Internet: **On**.

C4. Add-ons → Secrets → Add secret:
- Label `HF_TOKEN`, value = the Hugging Face token.
- Label `GH_TOKEN`, value = the GitHub token.
- Make sure both are ticked ("attached") for this notebook.

C5. Quick interactive test (about 5 minutes of quota): run cells 0, 1, 2. You should see the latest
commit hash and two "Tesla T4" lines. Then click the power button (Stop session).

---

## Part D — Session 1: timing + prefill check (no rewordings needed)

D1. In cell 1 (PLAN) set:
```python
HOURS_BUDGET = 11.0
BACKEND = 'transformers'
RUN_M1 = True
MMLU_MODELS = ['qwen2.5-3b', 'llama3.2-3b', 'phi3.5-mini', 'smollm3-3b', 'granite3.3-2b']
BFCL_MODELS = []
```

D2. Save Version → "Save & Run All (Commit)" → Save. Then stop the interactive session
(power button) so you don't use quota twice. You can close the browser.

What this version does:
- Clones the repo, records `pip freeze`, samples the 300 MMLU-Pro items, makes the reshuffles.
- Runs phi3.5-mini (the largest model) on 50 items × 2 orders × (4 seeds + greedy) = 500 runs, and logs
  the time per batch to `results/logs/timing.jsonl`.
- Runs every model greedy on 50 items, original wording, with and without the "The answer is" prefill.
- Uses both GPUs: jobs alternate between GPU 0 and GPU 1.
- Prints accuracy and parse-failure rate per model, then zips everything to `results.zip`.
Expected: well under an hour; the first model download takes a few minutes per model.

D3. When the email arrives, open the notebook → Output tab (or the version's Log) and note:
- **Speed:** seconds per generation for phi3.5-mini. Estimate per model:
  9,000 runs × that time (MMLU-Pro). If 5 models × 9,000 will not fit in about 2 weeks of
  quota, plan to drop to 4 models (keep the minimum: 4 models with all factors).
- **Prefill rule:** if any 3B model has `parse_fail` above 0.10 without the prefill, the prefill is
  used for ALL models. Then, on your laptop, set `assistant_prefill: "The answer is"` in
  `configs/experiment.yaml`, commit and push. Otherwise leave it as `""`. Write down the numbers;
  they go into the paper (Section 6.4 and 7.2).

D4. Optional vLLM test. Set `BACKEND = 'vllm'`, `RUN_M1 = True`, and `MMLU_MODELS = ['qwen2.5-3b']`, run a
version, and compare the speed in `timing.jsonl`. Use vLLM only if it installs and is clearly faster.
Use the same backend for every model in the main runs (state it in the paper).

---

## Part E — Full MMLU-Pro runs (several sessions)

Needs Part A finished for MMLU-Pro (`data/reworded/mmlu_pro_n300_wordings.jsonl` pushed).

E1. Plan cell:
```python
RUN_M1 = False
MMLU_MODELS = ['qwen2.5-3b', 'llama3.2-3b', 'phi3.5-mini', 'smollm3-3b', 'granite3.3-2b']
BFCL_MODELS = []
```
Put the most important models first: models 1 and 3 go to GPU 0, models 2 and 4 to GPU 1, and so on.

E2. Save & Run All, stop the interactive session.

E3. The version stops itself after `HOURS_BUDGET` hours: unfinished runs are killed (finished
batches are already saved), then cell 10 prints progress, e.g.
`mmlu300_full__qwen2.5-3b   9000/9000 COMPLETE`, and cell 12 zips everything.

E4. Next session, carry results forward:
- In the notebook editor: Add Input (right panel) → Your Work / Notebooks → this notebook →
  select its output (the latest version). It appears under `/kaggle/input/...`.
- Cell 4 finds `results.zip` there and unzips it without overwriting; `run_generation` then skips every
  run already done.
- Each zip contains everything so far, so attach only the latest version's output (remove older ones).
- Remove finished models from `MMLU_MODELS` (or leave them; they finish instantly).

E5. Repeat until every model shows COMPLETE. With about 30 h/week, check the remaining quota
before each version (Profile → Settings). Before the weekly reset, a short version still helps:
progress is never lost.

---

## Part F — BFCL "multiple"

F1. Pilot (needed for 18 Nov): after `data/reworded/bfcl_multiple_n100_wordings.jsonl` is pushed:
```python
RUN_M1 = False
MMLU_MODELS = []          # or the unfinished ones
BFCL_N = 100
BFCL_MODELS = ['qwen2.5-3b', 'llama3.2-3b', 'phi3.5-mini', 'smollm3-3b', 'granite3.3-2b']
```
100 items × 3 wordings × 2 orders × 5 = 3,000 runs per model, up to 128 tokens each.
Expect this to be noticeably slower per run than MMLU-Pro; check `timing.jsonl` after the first model.

F2. Check that 128 tokens was enough. Look at `unparseable` in `results/analysis/bfcl100/error_breakdown.csv`,
and count outputs that hit the limit:
```python
import pandas as pd; g = pd.read_json('results/generations/bfcl100__qwen2.5-3b.jsonl', lines=True)
print((g.n_new_tokens >= 128).mean())
```
If more than about 1% hit the limit, raise `max_new_tokens` in `configs/experiment_bfcl.yaml` and rerun that model
(delete its file in `results/generations/` first, so all of its runs use the same settings).

F3. Full BFCL (only if time remains after the minimum): `BFCL_N = 200`, all 200 wordings finalized. This is a
separate manifest (`bfcl200`), so pilot runs are not mixed in.

---

## Part G — Final analysis and download

G1. When everything is COMPLETE, run one short version with `RUN_M1 = False` and the lists unchanged.
The runs are skipped, and cell 11 scores and analyses everything.

G2. Output tab → download `results.zip`. Also: "New Dataset" from the output, so the results
are kept even if you delete the notebook.

G3. On your laptop, unzip into the repo and rerun the analysis (identical numbers, since bootstrap
seeds are fixed):
```bash
unzip -n results.zip -d .
python -m src.analyse --scored results/scored/mmlu300_full.csv --ranking-only
python -m src.analyse --config configs/experiment_bfcl.yaml --scored results/scored/bfcl100.csv --ranking-only
```
Commit `results/scored`, `results/analysis`, `results/logs` (generations stay out of git;
keep them in the Kaggle dataset).

G4. Fill in Section 8 of the paper from `reliability_card.md`, `pairwise.csv`, `budget_curve.csv`, and the figures.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `401` / "gated repo" for Llama | Licence not yet approved, or `HF_TOKEN` secret not attached to this notebook |
| `git clone` asks for a password / 403 | `GH_TOKEN` missing, expired, or not given access to `llm-reliability` |
| GPU option greyed out | phone not verified, or weekly quota used up |
| Waiting in a queue | normal at busy times; the version starts when a GPU is free |
| CUDA out of memory | lower `bs` in `job(...)` (e.g. 16 for MMLU-Pro, 8 for BFCL) |
| Disk full (model downloads) | `!rm -rf ~/.cache/huggingface/hub/models--<org>--<name>` for finished models |
| Version ended "failed" at the time limit | lower `HOURS_BUDGET` (e.g. 10.5) so the zip cell runs in time |
| Restore cell found nothing | attach the previous version's output via Add Input |
| `skipped: no finalized … wordings` | Part A not pushed yet, or the notebook didn't `git pull` (cell 2 does) |
| vLLM install fails | use `BACKEND = 'transformers'` for every model |
| Gemma or bfloat16 errors | not used; all models run in float16 on T4 |

## What to write down for the paper

- GPU (T4), backend (transformers/vLLM), batch size, the date of the Kaggle runs.
- `results/logs/pip_freeze.txt` (library versions) and the pinned model revisions in `configs/models.yaml`.
- The rewording model and settings (`openai/gpt-oss-120b`, reasoning effort low, temperature 0.8), and the
  number of rewordings rejected and items dropped.
- The prefill decision and the pilot numbers behind it.
- Total GPU hours used (sum of `timing.jsonl`, or the quota page).
- One caveat: a model whose run was stopped and resumed restarts its random generator at the
  beginning of the batch group, so its sampled outputs are valid but not bit-identical to an uninterrupted run.
