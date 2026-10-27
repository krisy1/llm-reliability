"""Step 3 - write one row for EVERY planned run before any generation happens.

A manifest row is model-independent; the generation script runs it once per model.
Rows: items x wordings x orders x (seeds [sampled] + 1 [greedy]).

Usage:
  python -m src.build_manifest --items data/sampled/mmlu_pro_n50.jsonl
  python -m src.build_manifest --items ... --wordings w0 --seeds 0 1    # smoke test, no rewordings yet
"""
from __future__ import annotations

import argparse
import itertools
from collections import Counter
from pathlib import Path

from .common import DATA, LETTERS, load_yaml, read_jsonl, write_jsonl
from .make_variants import apply_order
from .prompts import mmlu_user_message


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--wordings", nargs="+", default=cfg["factors"]["wordings"])
    ap.add_argument("--orders", nargs="+", default=cfg["factors"]["orders"])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["factors"]["seeds"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N items (smoke tests)")
    ap.add_argument("--greedy-only", action="store_true", help="only the greedy cells (quick pilot checks)")
    ap.add_argument("--name", default=None, help="manifest name (default: derived from items file)")
    a = ap.parse_args()

    stem = a.items.stem
    items = read_jsonl(a.items)
    orders = {(r["item_id"], r["order_id"]): r for r in read_jsonl(DATA / "manifests" / f"{stem}_orders.jsonl")}
    if not orders:
        raise SystemExit("run `python -m src.make_variants orders` first")

    wordings = {(it["item_id"], "w0"): it["question"] for it in items}
    if any(w != "w0" for w in a.wordings):
        wpath = DATA / "reworded" / f"{stem}_wordings.jsonl"
        if not wpath.exists():
            raise SystemExit(f"{wpath} not found: run reword + human check + finalize, or use --wordings w0")
        for r in read_jsonl(wpath):
            wordings[(r["item_id"], r["wording_id"])] = r["question"]
        keep = {r["item_id"] for r in read_jsonl(wpath)}
        items = [it for it in items if it["item_id"] in keep]  # only fully validated items
    if a.limit:
        items = items[: a.limit]

    decodings = [] if a.greedy_only else [("sampled", s) for s in a.seeds]
    if cfg["factors"]["greedy_cells"]:
        decodings.append(("greedy", None))

    rows = []
    for it, w, o, (dec, seed) in itertools.product(items, a.wordings, a.orders, decodings):
        q = wordings[(it["item_id"], w)]
        ordr = orders[(it["item_id"], o)]
        opts, gold = apply_order(it["options"], it["answer_index"], ordr["perm"])
        run_id = f"{it['item_id']}|{w}|{o}|{dec}|{'g' if seed is None else seed}"
        rows.append({
            "run_id": run_id, "item_id": it["item_id"], "task": "mmlu_pro",
            "category": it["category"], "wording_id": w, "order_id": o,
            "decoding": dec, "seed": seed, "n_options": len(opts),
            "gold_letter": LETTERS[gold], "gold_text": opts[gold],
            "prompt": mmlu_user_message(q, opts),
        })

    name = a.name or f"{stem}__{'-'.join(a.wordings)}__{'-'.join(a.orders)}__s{len(a.seeds)}"
    out = DATA / "manifests" / f"{name}.jsonl"
    write_jsonl(out, rows)
    print(f"wrote {out}: {len(rows)} runs per model over {len(items)} items; "
          f"{dict(Counter(r['decoding'] for r in rows))}")


if __name__ == "__main__":
    main()
