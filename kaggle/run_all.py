"""Kaggle script kernel: the whole GPU part of the study in one session (T4 x2).

1. copy the code+data dataset into /kaggle/working, restore generations from a previous version
2. M1 pilot: every ranking model, greedy, 50 items, with and without the "The answer is" prefill
3. apply the pre-registered prefill rule (prefill for ALL models iff any parse-failure rate > 10%)
4. full runs, one model at a time per GPU: MMLU-Pro (300 items x 30 runs) then BFCL (200 x 30)
5. score, analyse, zip everything to /kaggle/working/results.zip
Every run is resumable: rerun the kernel with the previous version attached and it continues.
"""
import glob, json, os, pathlib, queue, shutil, subprocess, sys, threading, time

T0 = time.time()
BUDGET = float(os.environ.get("HOURS_BUDGET", "11.2")) * 3600
W = pathlib.Path("/kaggle/working/llm-reliability")
MODELS = ["qwen2.5-3b", "llama3.2-3b", "phi3.5-mini", "smollm3-3b", "granite3.3-2b"]
NO_GQA = {"phi3.5-mini"}            # large KV cache per token -> smaller token budget per batch


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, cwd=W, **kw)


def left():
    return BUDGET - (time.time() - T0)


# ---------------------------------------------------------------- 1. code, restore
src = next(pathlib.Path(p).parent.parent for p in glob.glob("/kaggle/input/**/src/run_generation.py", recursive=True)
           if "results" not in p)
if not W.exists():
    shutil.copytree(src, W)
for f in glob.glob("/kaggle/input/**/results/generations/*.jsonl", recursive=True):
    dst = W / "results" / "generations" / pathlib.Path(f).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size < os.path.getsize(f):
        shutil.copy(f, dst); print("restored", dst.name, flush=True)
(W / "results" / "logs").mkdir(parents=True, exist_ok=True)
import transformers, torch
print("transformers", transformers.__version__, "torch", torch.__version__, flush=True)
if tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (4, 56):
    sh("pip install -q 'transformers>=4.56,<6' accelerate")
sh("pip freeze > results/logs/pip_freeze.txt; nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee results/logs/gpus.txt")

# manifests (deterministic; wordings were finalized and shipped in data/reworded/)
sh("python -m src.build_manifest --items data/sampled/mmlu_pro_n300.jsonl --seeds 0 1 2 3 --wordings w0 w1 w2 --name mmlu300_full")
sh("python -m src.build_manifest --config configs/experiment_bfcl.yaml --items data/sampled/bfcl_multiple_n200.jsonl --name bfcl200")


def gen_cmd(model, manifest, config=None, bs=32, extra=""):
    mbt = 12000 if model in NO_GQA else 24000
    c = f"python -m src.run_generation --model {model} --manifest {manifest} --batch-size {bs} --max-batch-tokens {mbt}"
    return c + (f" --config {config}" if config else "") + (" " + extra if extra else "")


def lanes(job_lists):
    """job_lists: shared queue of lists of commands; each GPU pulls the next list and runs it in order."""
    q = queue.Queue()
    for j in job_lists:
        q.put(j)

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        while True:
            try:
                cmds = q.get_nowait()
            except queue.Empty:
                return
            for cmd in cmds:
                if left() < 900:
                    print(f"GPU{gpu}: no time left, skipping: {cmd}", flush=True); continue
                print(f"GPU{gpu} start [{(time.time()-T0)/3600:.2f} h]: {cmd}", flush=True)
                with open(W / "results" / "logs" / f"gpu{gpu}.log", "a") as log:
                    p = subprocess.Popen(cmd, shell=True, cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        p.wait(timeout=max(60, left() - 600))
                    except subprocess.TimeoutExpired:
                        p.kill(); p.wait(); print(f"GPU{gpu}: deadline, stopped (resumable)", flush=True)
                print(f"GPU{gpu} done rc={p.returncode} [{(time.time()-T0)/3600:.2f} h]: {cmd}", flush=True)
    ts = [threading.Thread(target=worker, args=(g,)) for g in range(torch.cuda.device_count() or 1)]
    [t.start() for t in ts]; [t.join() for t in ts]


# ---------------------------------------------------------------- 2-3. M1 pilot + prefill rule
decision_f = W / "results" / "logs" / "prefill_decision.json"
if not decision_f.exists():
    lanes([[gen_cmd(m, "data/manifests/pilot_prefill_greedy.jsonl"),
            gen_cmd(m, "data/manifests/pilot_prefill_greedy.jsonl", extra="--prefill 'The answer is' --tag _prefill")]
           for m in MODELS])
    sh("python -m src.score_runs --manifest data/manifests/pilot_prefill_greedy.jsonl")
    import pandas as pd
    d = pd.read_csv(W / "results" / "scored" / "pilot_prefill_greedy.csv")
    d = d[d["model"].isin(MODELS + [m + "_prefill" for m in MODELS])]
    tab = d.groupby("model").agg(acc=("correct", "mean"), parse_fail=("parse_fail", "mean")).round(4)
    pf = {m: float(tab.loc[m, "parse_fail"]) for m in MODELS if m in tab.index}
    use = any(v > 0.10 for v in pf.values())
    decision = {"rule": "prefill 'The answer is' for ALL models iff any ranking model's no-prefill parse-failure rate > 10%",
                "parse_fail_no_prefill": pf, "use_prefill": use, "pilot_table": tab.reset_index().to_dict("records")}
    decision_f.write_text(json.dumps(decision, indent=2))
decision = json.loads(decision_f.read_text())
print(json.dumps(decision, indent=2), flush=True)
if decision["use_prefill"]:
    cfg = (W / "configs" / "experiment.yaml").read_text().replace('assistant_prefill: ""', 'assistant_prefill: "The answer is"')
    (W / "configs" / "experiment.yaml").write_text(cfg)

# ---------------------------------------------------------------- 4. full runs
# slowest model first so the two GPUs finish at about the same time
order = ["phi3.5-mini", "smollm3-3b", "llama3.2-3b", "qwen2.5-3b", "granite3.3-2b"]
lanes([[gen_cmd(m, "data/manifests/mmlu300_full.jsonl", bs=32),
        gen_cmd(m, "data/manifests/bfcl200.jsonl", config="configs/experiment_bfcl.yaml", bs=24)] for m in order])

# ---------------------------------------------------------------- 5. score + analyse + zip
sh("python -m src.score_runs --manifest data/manifests/mmlu300_full.jsonl")
sh("python -m src.analyse --scored results/scored/mmlu300_full.csv --ranking-only")
sh("python -m src.score_runs --manifest data/manifests/bfcl200.jsonl")
sh("python -m src.analyse --config configs/experiment_bfcl.yaml --scored results/scored/bfcl200.csv --ranking-only")
sh("cp configs/experiment.yaml results/logs/experiment_used.yaml")
sh("rm -f /kaggle/working/results.zip && zip -qr /kaggle/working/results.zip results data/manifests/*_orders.jsonl")
print(f"finished in {(time.time()-T0)/3600:.2f} h", flush=True)
