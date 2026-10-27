"""Step 2 - build the wording and order variants for every sampled item.

Sub-commands
------------
orders    one fixed reshuffle per item (o1) that always moves the gold answer to a new
          letter; o0 is the original order. Writes data/manifests/<stem>_orders.jsonl
reword    calls an OpenAI-compatible chat API (Groq, Gemini, OpenRouter, ...) once per
          item per rewording and runs automatic checks. Writes
          data/reworded/<stem>_review.csv for HUMAN checking (fill column human_ok = 1/0).
finalize  reads the reviewed CSV and writes data/reworded/<stem>_wordings.jsonl, keeping only
          items whose rewordings all passed both automatic and human checks.

Environment variables for `reword` (never hard-code keys):
  REWORD_API_BASE   e.g. https://api.groq.com/openai/v1
  REWORD_API_KEY
  REWORD_MODEL      e.g. llama-3.3-70b-versatile

Usage
-----
python -m src.make_variants orders   --items data/sampled/mmlu_pro_n50.jsonl
python -m src.make_variants reword   --items data/sampled/mmlu_pro_n50.jsonl --n-rewordings 2
python -m src.make_variants finalize --items data/sampled/mmlu_pro_n50.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from .common import DATA, LETTERS, item_seed, load_yaml, read_jsonl, write_jsonl

# ---------------------------------------------------------------- orders

def reshuffle(n_options: int, gold_index: int, seed: int) -> list[int]:
    """Return perm where new position j shows original option perm[j].

    Guarantees the gold answer lands on a different letter, so the order factor
    always changes the correct label (needed to separate content from label bias).
    """
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        perm = rng.permutation(n_options).tolist()
        if perm.index(gold_index) != gold_index:
            return perm
    raise RuntimeError("could not find a valid reshuffle")


def apply_order(options: list[str], gold_index: int, perm: list[int]) -> tuple[list[str], int]:
    new_options = [options[p] for p in perm]
    new_gold = perm.index(gold_index)
    assert new_options[new_gold] == options[gold_index]
    return new_options, new_gold


def cmd_orders(items_path: Path) -> None:
    cfg = load_yaml("configs/experiment.yaml")
    base = cfg["sampling"]["random_state"]
    rows = []
    for it in read_jsonl(items_path):
        n, g = it["n_options"], it["answer_index"]
        perm = reshuffle(n, g, item_seed(it["item_id"], "order_o1", base))
        rows.append({"item_id": it["item_id"], "order_id": "o0", "perm": list(range(n)),
                     "gold_index": g, "gold_letter": LETTERS[g]})
        _, g1 = apply_order(it["options"], g, perm)
        rows.append({"item_id": it["item_id"], "order_id": "o1", "perm": perm,
                     "gold_index": g1, "gold_letter": LETTERS[g1]})
    out = DATA / "manifests" / f"{items_path.stem}_orders.jsonl"
    write_jsonl(out, rows)
    moved = Counter(r["gold_letter"] for r in rows if r["order_id"] == "o1")
    print(f"wrote {out}  ({len(rows)} rows); gold letters after reshuffle: {dict(sorted(moved.items()))}")


# ---------------------------------------------------------------- rewording

REWORD_SYSTEM = (
    "You rewrite exam questions. Produce ONE paraphrase of the question that a careful "
    "examiner would accept as asking exactly the same thing. Rules: keep every number, unit, "
    "formula, name, quotation and blank (____) exactly as written; do not add hints, do not "
    "remove information, do not answer the question, do not mention answer options. "
    "Change sentence structure and wording, not meaning. Return only the rewritten question."
)

NUM = re.compile(r"-?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?")
BLANK = re.compile(r"_{2,}")


def call_chat(prompt: str, variant: int, retries: int = 4) -> str:
    base = os.environ["REWORD_API_BASE"].rstrip("/")
    body = {
        "model": os.environ["REWORD_MODEL"],
        "messages": [
            {"role": "system", "content": REWORD_SYSTEM},
            {"role": "user", "content": f"Paraphrase #{variant} (make it differ from other paraphrases).\n\nQuestion:\n{prompt}"},
        ],
        "temperature": 0.8,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['REWORD_API_KEY']}",
                 "Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception as e:  # rate limits on free tiers are common: back off
            wait = 2 ** attempt * 5
            print(f"  API error ({e}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError("rewording API failed repeatedly")


def auto_check(original: str, reworded: str, options: list[str]) -> list[str]:
    """Cheap automatic checks; anything flagged still goes to the human reviewer."""
    issues = []
    if Counter(NUM.findall(original)) != Counter(NUM.findall(reworded)):
        issues.append("numbers_changed")
    if len(BLANK.findall(original)) != len(BLANK.findall(reworded)):
        issues.append("blanks_changed")
    ratio = len(reworded) / max(1, len(original))
    if not 0.6 <= ratio <= 1.7:
        issues.append(f"length_ratio_{ratio:.2f}")
    if re.sub(r"\W", "", reworded.lower()) == re.sub(r"\W", "", original.lower()):
        issues.append("identical_to_original")
    low_o, low_r = original.lower(), reworded.lower()
    for o in options:  # answer text leaking into the question
        o_l = o.lower().strip()
        if len(o_l) > 12 and o_l in low_r and o_l not in low_o:
            issues.append("option_text_leaked")
            break
    if re.search(r"\b(answer|option)\s*[:(]?\s*[A-J]\b", reworded):
        issues.append("mentions_option_letter")
    return issues


def cmd_reword(items_path: Path, n_rewordings: int) -> None:
    out = DATA / "reworded" / f"{items_path.stem}_review.csv"
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r["item_id"], r["wording_id"]) for r in csv.DictReader(f)}
    new_file = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "wording_id", "original", "reworded",
                                          "auto_issues", "human_ok", "notes"])
        if new_file:
            w.writeheader()
        for it in read_jsonl(items_path):
            for k in range(1, n_rewordings + 1):
                wid = f"w{k}"
                if (it["item_id"], wid) in done:
                    continue
                text = call_chat(it["question"], k)
                issues = auto_check(it["question"], text, it["options"])
                w.writerow({"item_id": it["item_id"], "wording_id": wid,
                            "original": it["question"], "reworded": text,
                            "auto_issues": ";".join(issues), "human_ok": "", "notes": ""})
                f.flush()
                print(f"{it['item_id']} {wid}: {'OK' if not issues else issues}")
    print(f"review sheet: {out}\nFill human_ok with 1 (same meaning) or 0, then run `finalize`.")


def cmd_finalize(items_path: Path, n_rewordings: int) -> None:
    review = DATA / "reworded" / f"{items_path.stem}_review.csv"
    with open(review) as f:
        rows = list(csv.DictReader(f))
    by_item: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_item.setdefault(r["item_id"], {})[r["wording_id"]] = r

    out_rows, dropped = [], Counter()
    unreviewed = 0
    for it in read_jsonl(items_path):
        got = by_item.get(it["item_id"], {})
        out_rows.append({"item_id": it["item_id"], "wording_id": "w0", "question": it["question"]})
        ok = True
        for k in range(1, n_rewordings + 1):
            r = got.get(f"w{k}")
            if r is None:
                dropped["missing"] += 1; ok = False; continue
            if r["human_ok"].strip() == "":
                unreviewed += 1; ok = False; continue
            if r["human_ok"].strip() != "1":
                dropped["human_rejected"] += 1; ok = False; continue
            out_rows.append({"item_id": it["item_id"], "wording_id": f"w{k}", "question": r["reworded"]})
        if not ok:  # drop the whole item: the design must stay fully crossed
            out_rows = [x for x in out_rows if x["item_id"] != it["item_id"]]
            dropped["items_dropped"] += 1
    if unreviewed:
        print(f"WARNING: {unreviewed} rewordings have no human_ok value yet; their items were dropped.")
    out = DATA / "reworded" / f"{items_path.stem}_wordings.jsonl"
    write_jsonl(out, out_rows)
    n_items = len({r["item_id"] for r in out_rows})
    print(f"wrote {out}: {n_items} items kept; {dict(dropped)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["orders", "reword", "finalize"])
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--n-rewordings", type=int, default=2)
    a = ap.parse_args()
    items = a.items if a.items.is_absolute() else Path.cwd() / a.items
    {"orders": lambda: cmd_orders(items),
     "reword": lambda: cmd_reword(items, a.n_rewordings),
     "finalize": lambda: cmd_finalize(items, a.n_rewordings)}[a.cmd]()


if __name__ == "__main__":
    main()
