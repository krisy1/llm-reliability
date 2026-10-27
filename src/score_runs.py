"""Step 5 - join raw generations with the manifest, parse, and score.

Usage: python -m src.score_runs --manifest data/manifests/<name>.jsonl
Writes results/scored/<manifest>.csv with one row per (model, run).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import LETTERS, RESULTS, read_jsonl
from .parse_outputs import parse_letter


def score(manifest: Path) -> pd.DataFrame:
    man = pd.DataFrame(read_jsonl(manifest))
    frames = []
    for gen_file in sorted((RESULTS / "generations").glob(f"{manifest.stem}__*.jsonl")):
        gen = pd.DataFrame(read_jsonl(gen_file)).drop_duplicates("run_id", keep="last")
        tag = gen_file.stem.split("__")[-1]          # model name (+ optional tag, e.g. _prefill)
        gen["model"] = tag
        if "prefill" not in gen:
            gen["prefill"] = ""
        gen["prefill"] = gen["prefill"].fillna("")
        df = man.merge(gen, on="run_id", how="inner")
        missing = len(man) - len(df)
        if missing:
            print(f"  {gen_file.name}: {missing} runs not generated yet (excluded)")
        parsed = [parse_letter(pf + r, n) for pf, r, n in zip(df["prefill"], df["raw_output"], df["n_options"])]
        df["parsed_letter"] = [p[0] for p in parsed]
        df["parse_rule"] = [p[1] for p in parsed]
        df["parse_fail"] = df["parsed_letter"].isna()
        df["correct"] = (df["parsed_letter"] == df["gold_letter"]).astype(int)
        frames.append(df)
    if not frames:
        raise SystemExit("no generations found for this manifest")
    out = pd.concat(frames, ignore_index=True)
    # chosen option TEXT (needed to detect answer changes across orders, where letters move)
    out["chosen_text"] = [
        _option_text(p, l) if l else None for p, l in zip(out["prompt"], out["parsed_letter"])
    ]
    # diagnostic: model wrote "B. <text>" but <text> is not option B's text
    out["letter_text_mismatch"] = [
        _mismatch(raw, txt) if rule == "letter_then_text" else False
        for raw, txt, rule in zip(out["raw_output"], out["chosen_text"], out["parse_rule"])
    ]
    return out.drop(columns=["prompt"])


def _mismatch(raw: str, chosen: str | None) -> bool:
    if chosen is None:
        return False
    tail = raw.strip().replace("**", "").split("\n")[0][2:].strip().lower()
    head = chosen.lower()
    k = min(len(tail), len(head), 12)  # output is cut at 8 tokens; compare the first characters
    return k > 0 and tail[:k] != head[:k]


def _option_text(prompt: str, letter: str) -> str | None:
    for line in prompt.splitlines():
        if line.startswith(f"{letter}. "):
            return line[3:].strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    a = ap.parse_args()
    df = score(a.manifest)
    out = RESULTS / "scored" / f"{a.manifest.stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = df.groupby(["model", "decoding"]).agg(n=("correct", "size"), acc=("correct", "mean"),
                                                     parse_fail=("parse_fail", "mean"),
                                                     letter_text_mismatch=("letter_text_mismatch", "mean"))
    print(summary.round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
