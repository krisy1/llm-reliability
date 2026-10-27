"""Step 1 (BFCL) - load the BFCL v4 "multiple" category (200 items, 2-4 candidate functions,
exactly one correct call each) together with its possible-answer file.

The raw files are downloaded from a pinned gorilla commit so the data cannot drift.

Usage:  python -m src.sample_bfcl            # all 200 items
        python -m src.sample_bfcl --n 100    # random pilot subset (seeded)
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter

import numpy as np

from .common import DATA, read_jsonl, write_jsonl

GORILLA_SHA = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BASE = (f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{GORILLA_SHA}/"
        "berkeley-function-call-leaderboard/bfcl_eval/data/")
FILES = {"BFCL_v4_multiple.json": "BFCL_v4_multiple.json",
         "possible_answer/BFCL_v4_multiple.json": "possible_answer_BFCL_v4_multiple.json"}
RAW = DATA / "raw" / "bfcl"


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for remote, local in FILES.items():
        if not (RAW / local).exists():
            urllib.request.urlretrieve(BASE + remote, RAW / local)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--random-state", type=int, default=2026)
    a = ap.parse_args()

    download()
    qs = {r["id"]: r for r in read_jsonl(RAW / "BFCL_v4_multiple.json")}
    ans = {r["id"]: r for r in read_jsonl(RAW / "possible_answer_BFCL_v4_multiple.json")}
    assert qs.keys() == ans.keys()

    rows = []
    for bid in sorted(qs, key=lambda s: int(s.split("_")[-1])):
        q, gt = qs[bid], ans[bid]["ground_truth"]
        turns = q["question"][0]
        assert len(q["question"]) == 1 and len(turns) == 1 and turns[0]["role"] == "user"
        assert len(gt) == 1 and len(gt[0]) == 1          # exactly one correct call
        gold_name = next(iter(gt[0]))
        names = [f["name"] for f in q["function"]]
        assert names.count(gold_name) == 1, bid
        rows.append({
            "item_id": f"bfcl_{bid}", "bfcl_id": bid, "category": "multiple",
            "question": turns[0]["content"],
            "functions": q["function"],
            "ground_truth": gt,
            "answer_index": names.index(gold_name),       # position of the correct function
            "n_options": len(names),                      # number of candidate functions
        })

    if a.n and a.n < len(rows):
        idx = np.random.default_rng(a.random_state).choice(len(rows), a.n, replace=False)
        rows = [rows[i] for i in sorted(idx)]

    out = DATA / "sampled" / f"bfcl_multiple_n{len(rows)}.jsonl"
    write_jsonl(out, rows)
    report = {"dataset": "BFCL v4 multiple", "gorilla_commit": GORILLA_SHA, "n_items": len(rows),
              "n_functions_hist": dict(Counter(r["n_options"] for r in rows)),
              "gold_position_hist": dict(Counter(r["answer_index"] for r in rows))}
    (DATA / "sampled" / f"{out.stem}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)); print(f"wrote {out}")


if __name__ == "__main__":
    main()
