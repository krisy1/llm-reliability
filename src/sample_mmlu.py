"""Step 1 - draw a stratified, reshuffle-safe sample of MMLU-Pro test questions.

Items are excluded *before* sampling if reordering their options could change the
correct answer or make it ambiguous:
  * an option refers to other options by position ("all of the above", "both A and B", ...)
  * two options have identical text (after normalising whitespace/case)
  * an option is empty
The number of excluded items is written to the sample report so it can be stated in the paper.

Usage:  python -m src.sample_mmlu [--n 50]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from datasets import load_dataset

from .common import DATA, LETTERS, load_yaml, write_jsonl

# Phrases that point at other options. Letter rules are case-sensitive on purpose:
# lower-case "(a)", "(b)" in MMLU-Pro options label sub-questions, not options.
POSITIONAL_WORDS = re.compile(
    r"\b(all|none|both|neither)\s+(of\s+)?(the\s+)?(above|below|options|choices|answers)\b",
    re.IGNORECASE,
)
POSITIONAL_LETTERS = re.compile(
    r"(?<![A-Za-z])[A-J]\s*(?:and|&|or|,)\s*[A-J](?![A-Za-z])"   # "A and B", "C or D"
    r"|\b(?:[Oo]ptions?|[Cc]hoices?)\s+[A-J]\b"                   # "option C"
)


class _Positional:
    """Single object with .search() so callers can treat the two regexes as one."""
    def search(self, text: str):
        return POSITIONAL_WORDS.search(text) or POSITIONAL_LETTERS.search(text)


POSITIONAL = _Positional()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def exclusion_reason(options: list[str]) -> str | None:
    if any(not str(o).strip() for o in options):
        return "empty_option"
    if len({norm(o) for o in options}) < len(options):
        return "duplicate_options"
    if any(POSITIONAL.search(str(o)) for o in options):
        return "positional_reference"
    return None


def stratified_sample(df: pd.DataFrame, n: int, by: str, rng: np.random.Generator) -> pd.DataFrame:
    groups = sorted(df[by].unique())
    base, extra = divmod(n, len(groups))
    quota = {g: base for g in groups}
    for g in rng.choice(groups, size=extra, replace=False):
        quota[g] += 1
    parts = []
    for g in groups:
        sub = df[df[by] == g]
        k = min(quota[g], len(sub))
        idx = rng.choice(len(sub), size=k, replace=False)
        parts.append(sub.iloc[sorted(idx)])
    return pd.concat(parts).reset_index(drop=True)


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=cfg["sampling"]["n_items"])
    args = ap.parse_args()
    seed = cfg["sampling"]["random_state"]

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    df = ds.to_pandas()
    df["exclusion"] = df["options"].apply(lambda o: exclusion_reason(list(o)))
    excluded = Counter(df["exclusion"].dropna())
    pool = df[df["exclusion"].isna()]

    rng = np.random.default_rng(seed)
    sample = stratified_sample(pool, args.n, cfg["sampling"]["stratify_by"], rng)

    rows = []
    for _, r in sample.iterrows():
        opts = [str(o) for o in r["options"]]
        assert LETTERS[int(r["answer_index"])] == r["answer"], r["question_id"]
        rows.append({
            "item_id": f"mmlu_{int(r['question_id'])}",
            "question_id": int(r["question_id"]),
            "category": r["category"],
            "src": r["src"],
            "question": r["question"],
            "options": opts,
            "answer_index": int(r["answer_index"]),
            "answer_letter": r["answer"],
            "n_options": len(opts),
        })

    out = DATA / "sampled" / f"mmlu_pro_n{args.n}.jsonl"
    write_jsonl(out, rows)
    report = {
        "dataset": "TIGER-Lab/MMLU-Pro (test split)",
        "n_total": int(len(df)),
        "n_excluded": int(sum(excluded.values())),
        "excluded_by_reason": dict(excluded),
        "n_pool": int(len(pool)),
        "n_sampled": len(rows),
        "random_state": seed,
        "per_category": dict(Counter(r["category"] for r in rows)),
        "n_options_hist": dict(Counter(r["n_options"] for r in rows)),
    }
    (DATA / "sampled" / f"mmlu_pro_n{args.n}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
