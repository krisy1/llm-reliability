"""Step 4 - run every manifest row for one model and append raw outputs to JSONL.

* Resumable: rows whose run_id is already in the output file are skipped, so a Kaggle
  session that dies can simply be restarted.
* Raw text is stored un-parsed; parsing/scoring is a separate step (src/score_runs.py).
* Seeds: with vLLM each request gets its own seed. With transformers, rows are grouped by
  seed, sorted by run_id and generated in fixed-size batches after torch.manual_seed(seed);
  this is reproducible for the same batch size and hardware, which are both logged.

Usage:
  python -m src.run_generation --model qwen2.5-0.5b --manifest data/manifests/<name>.jsonl
  python -m src.run_generation --model qwen2.5-3b --manifest ... --backend vllm
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from .common import RESULTS, append_jsonl, load_yaml, read_jsonl
from .prompts import to_chat


def model_cfg(name: str) -> tuple[dict, str]:
    cfg = load_yaml("configs/models.yaml")
    for m in cfg["models"]:
        if m["name"] == name:
            return m, cfg["dtype"]
    raise SystemExit(f"model {name} not in configs/models.yaml")


PREFILL = ""


def render(tok, prompt: str, chat_kwargs: dict) -> str:
    text = tok.apply_chat_template(to_chat(prompt), tokenize=False,
                                   add_generation_prompt=True, **chat_kwargs)
    return text + PREFILL


# ---------------------------------------------------------------- transformers backend

def run_transformers(m: dict, dtype: str, rows: list[dict], dec: dict, out: Path, batch_size: int, device: str,
                     max_batch_tokens: int = 24000):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    if device == "cpu" and torch_dtype == torch.float16:
        torch_dtype = torch.float32  # fp16 matmuls are not supported/slow on CPU; logged below
    tok = AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(m["hf_id"], revision=m["revision"], dtype=torch_dtype).to(device)
    model.eval()
    meta = {"backend": "transformers", "dtype_used": str(torch_dtype).replace("torch.", ""),
            "device": device, "batch_size": batch_size, "max_batch_tokens": max_batch_tokens,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "torch": torch.__version__, "python": platform.python_version()}
    chat_kwargs = m.get("chat_template_kwargs", {}) or {}

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["decoding"], r["seed"]), []).append(r)

    total = len(rows); done = 0; t_start = time.time()
    for (decoding, seed), grp in sorted(groups.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1])):
        p = dec[decoding]
        # Sort by prompt length (ties by run_id) so a batch holds prompts of similar length, and cap
        # each batch at max_batch_tokens (prompt + new tokens, times rows). Deterministic for a given
        # manifest, batch size and token cap, which are all logged.
        texts_all = {r["run_id"]: render(tok, r["prompt"], chat_kwargs) for r in grp}
        lens = {k: len(tok(v, add_special_tokens=False)["input_ids"]) for k, v in texts_all.items()}
        grp.sort(key=lambda r: (lens[r["run_id"]], r["run_id"]))
        batches, cur = [], []
        for r in grp:
            width = lens[r["run_id"]] + p["max_new_tokens"]
            if cur and (len(cur) >= batch_size or (len(cur) + 1) * width > max_batch_tokens):
                batches.append(cur); cur = []
            cur.append(r)
        if cur:
            batches.append(cur)
        if seed is not None:
            torch.manual_seed(seed)
        queue = list(batches)
        while queue:
            batch = queue.pop(0)
            texts = [texts_all[r["run_id"]] for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            gen_kwargs = dict(max_new_tokens=p["max_new_tokens"], pad_token_id=tok.pad_token_id)
            if decoding == "greedy":
                gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            else:
                gen_kwargs.update(do_sample=True, temperature=p["temperature"], top_p=p["top_p"], top_k=0)
            t0 = time.time()
            try:
                with torch.no_grad():
                    outs = model.generate(**enc, **gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                del enc
                torch.cuda.empty_cache()
                if len(batch) == 1:
                    raise
                half = len(batch) // 2
                queue[:0] = [batch[:half], batch[half:]]   # split and retry; logged via batch_n
                print(f"  OOM at batch of {len(batch)}; splitting", flush=True)
                continue
            dt = time.time() - t0
            new = outs[:, enc["input_ids"].shape[1]:]
            recs = []
            for r, ids in zip(batch, new):
                recs.append({
                    "run_id": r["run_id"], "model": m["name"], "hf_id": m["hf_id"],
                    "raw_output": tok.decode(ids, skip_special_tokens=True),
                    "n_new_tokens": int((ids != tok.pad_token_id).sum()),
                    "prompt_tokens": int(enc["attention_mask"].shape[1]),
                    "batch_seconds": round(dt, 3), "batch_n": len(batch), "prefill": PREFILL, **meta,
                })
            append_jsonl(out, recs)
            done += len(batch)
            el = time.time() - t_start
            print(f"[{done}/{total}] {decoding} seed={seed} batch {len(batch)} {dt:.1f}s  "
                  f"eta {el / done * (total - done) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- vLLM backend

def run_vllm(m: dict, dtype: str, rows: list[dict], dec: dict, out: Path):
    from vllm import LLM, SamplingParams

    llm = LLM(model=m["hf_id"], revision=m["revision"], dtype=dtype, seed=0,
              max_model_len=4096, gpu_memory_utilization=0.90)
    tok = llm.get_tokenizer()
    chat_kwargs = m.get("chat_template_kwargs", {}) or {}
    meta = {"backend": "vllm", "dtype_used": dtype}
    texts, params = [], []
    for r in rows:
        p = dec[r["decoding"]]
        texts.append(render(tok, r["prompt"], chat_kwargs))
        if r["decoding"] == "greedy":
            params.append(SamplingParams(temperature=0.0, max_tokens=p["max_new_tokens"]))
        else:
            params.append(SamplingParams(temperature=p["temperature"], top_p=p["top_p"],
                                         max_tokens=p["max_new_tokens"], seed=int(r["seed"])))
    chunk = 2000  # write to disk regularly so a crash loses little
    for i in range(0, len(rows), chunk):
        t0 = time.time()
        res = llm.generate(texts[i:i + chunk], params[i:i + chunk])
        dt = time.time() - t0
        recs = [{"run_id": r["run_id"], "model": m["name"], "hf_id": m["hf_id"],
                 "raw_output": o.outputs[0].text, "n_new_tokens": len(o.outputs[0].token_ids),
                 "prompt_tokens": len(o.prompt_token_ids), "batch_seconds": round(dt, 3),
                 "batch_n": len(res), "prefill": PREFILL, **meta}
                for r, o in zip(rows[i:i + chunk], res)]
        append_jsonl(out, recs)
        print(f"[{min(i + chunk, len(rows))}/{len(rows)}] {dt:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-batch-tokens", type=int, default=24000,
                    help="cap on rows x (prompt + new tokens) per batch; lower it for models without GQA")
    ap.add_argument("--prefill", default=None, help="override prompt.assistant_prefill (pilot comparisons)")
    ap.add_argument("--config", default=None, help="configs/experiment_bfcl.yaml for BFCL")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. _prefill")
    a = ap.parse_args()

    global PREFILL
    m, dtype = model_cfg(a.model)
    exp = load_yaml(a.config or "configs/experiment.yaml")
    dec = exp["decoding"]
    PREFILL = a.prefill if a.prefill is not None else (exp.get("prompt", {}) or {}).get("assistant_prefill", "")
    rows = read_jsonl(a.manifest)
    out = RESULTS / "generations" / f"{a.manifest.stem}__{m['name']}{a.tag}.jsonl"
    done = {r["run_id"] for r in read_jsonl(out)}
    todo = [r for r in rows if r["run_id"] not in done]
    print(f"{m['hf_id']}: {len(rows)} runs in manifest, {len(done)} done, {len(todo)} to go -> {out}")
    if not todo:
        return
    t0 = time.time()
    if a.backend == "vllm":
        run_vllm(m, dtype, todo, dec, out)
    else:
        import torch
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        run_transformers(m, dtype, todo, dec, out, a.batch_size, device, a.max_batch_tokens)
    log = {"model": m["name"], "manifest": str(a.manifest), "n_runs": len(todo),
           "wall_seconds": round(time.time() - t0, 1), "backend": a.backend}
    with open(RESULTS / "logs" / "timing.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")
    print(json.dumps(log))


if __name__ == "__main__":
    main()
