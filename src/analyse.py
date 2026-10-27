"""Step 6 - metrics, item-level bootstrap CIs, ranking stability, budget curve, figures.

Every metric is first reduced to ONE number per item (per model); the bootstrap then
resamples items with replacement. Runs of the same item are never resampled
independently, because they are not independent (Miller, 2024).
The same resampled item indices are used for all models, so pairwise differences are paired.

Usage: python -m src.analyse --scored results/scored/<manifest>.csv [--ranking-only]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

# a factor with one level (e.g. no rewordings yet) gives all-NaN flip rates: expected, shown as n/a
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

from .common import RESULTS, load_yaml

METRICS = ["greedy_acc", "mean_acc", "reliable_acc", "worst_wording_acc"]


# ------------------------------------------------------------------ per-item tables

def item_tables(df: pd.DataFrame, model: str, items: list[str]) -> dict[str, np.ndarray]:
    """Return item-level arrays (aligned to `items`) for one model."""
    d = df[df["model"] == model]
    s = d[d["decoding"] == "sampled"]
    g = d[d["decoding"] == "greedy"]
    out: dict[str, np.ndarray] = {}

    g00 = g[(g["wording_id"] == "w0") & (g["order_id"] == "o0")].set_index("item_id")["correct"]
    out["greedy_acc"] = g00.reindex(items).to_numpy(float)
    per_item = s.groupby("item_id")["correct"]
    out["mean_acc"] = per_item.mean().reindex(items).to_numpy(float)
    out["reliable_acc"] = per_item.min().reindex(items).to_numpy(float)  # 1 only if correct in ALL runs
    # per-wording mean (items x wordings); worst wording is taken AFTER averaging over items
    pw = s.groupby(["item_id", "wording_id"])["correct"].mean().unstack()
    out["_per_wording"] = pw.reindex(items).to_numpy(float)
    out["parse_fail_rate"] = d.groupby("item_id")["parse_fail"].mean().reindex(items).to_numpy(float)

    # ---- flip rates (correctness changes), each item weighted equally
    out["flip_wording"] = _pair_flip(g, items, vary="wording_id", fixed=["order_id"], col="correct")
    out["flip_order"] = _pair_flip(g, items, vary="order_id", fixed=["wording_id"], col="correct")
    out["flip_seed"] = _pair_flip(s, items, vary="seed", fixed=["wording_id", "order_id"], col="correct")
    # ---- answer-change rates (the chosen option TEXT changes, even if both are wrong)
    gt = g.assign(ans=g["chosen_text"].fillna("<parse_fail>"))
    st = s.assign(ans=s["chosen_text"].fillna("<parse_fail>"))
    out["change_wording"] = _pair_flip(gt, items, "wording_id", ["order_id"], "ans")
    out["change_order"] = _pair_flip(gt, items, "order_id", ["wording_id"], "ans")
    out["change_seed"] = _pair_flip(st, items, "seed", ["wording_id", "order_id"], "ans")
    return out


def _pair_flip(d: pd.DataFrame, items: list[str], vary: str, fixed: list[str], col: str) -> np.ndarray:
    """Per item: share of run pairs that differ ONLY in `vary` whose `col` value differs."""
    res = {}
    for item, grp in d.groupby("item_id"):
        n_pairs = n_flip = 0
        for _, cell in grp.groupby(fixed):
            vals = cell[col].tolist()
            for a, b in itertools.combinations(vals, 2):
                n_pairs += 1
                n_flip += a != b
        res[item] = n_flip / n_pairs if n_pairs else np.nan
    return pd.Series(res).reindex(items).to_numpy(float)


def point_metrics(t: dict[str, np.ndarray], idx: np.ndarray | None = None) -> dict[str, float]:
    sel = (lambda a: a[idx]) if idx is not None else (lambda a: a)
    m = {k: float(np.nanmean(sel(v))) for k, v in t.items() if not k.startswith("_")}
    pw = sel(t["_per_wording"])
    m["worst_wording_acc"] = float(np.nanmin(np.nanmean(pw, axis=0)))
    return m


# ------------------------------------------------------------------ bootstrap

def bootstrap(tables: dict[str, dict], n_items: int, n_boot: int, rng: np.random.Generator):
    """Return {model: {metric: array(n_boot)}} using shared item resamples."""
    boots = {m: {} for m in tables}
    for b in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        for m, t in tables.items():
            for k, v in point_metrics(t, idx).items():
                boots[m].setdefault(k, []).append(v)
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in boots.items()}


def ci(a: np.ndarray, level: float) -> tuple[float, float]:
    lo, hi = np.nanpercentile(a, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


# ------------------------------------------------------------------ budget curve

def budget_curve(df: pd.DataFrame, models: list[str], items: list[str], n_boot: int,
                 rng: np.random.Generator, max_subsets: int = 12) -> pd.DataFrame:
    """Agreement of small designs with HELD-OUT runs.

    Seeds are split in half: designs draw seeds only from the first half (A); the reference
    ordering is mean accuracy over the second half (B), all wordings x orders. The reference
    therefore never contains the runs used by the design.
    """
    s = df[df["decoding"] == "sampled"]
    seeds = sorted(s["seed"].dropna().unique().astype(int))
    if len(seeds) < 2 or len(models) < 2:
        return pd.DataFrame()
    A, B = seeds[: len(seeds) // 2], seeds[len(seeds) // 2:]
    W = sorted(s["wording_id"].unique()); O = sorted(s["order_id"].unique())

    # cube[model] : items x seed x wording x order  (correctness)
    cube = {}
    for m in models:
        piv = s[s["model"] == m].set_index(["item_id", "seed", "wording_id", "order_id"])["correct"]
        full_idx = pd.MultiIndex.from_product([items, seeds, W, O])
        cube[m] = piv.reindex(full_idx).to_numpy(float).reshape(len(items), len(seeds), len(W), len(O))
    sidx = {sd: i for i, sd in enumerate(seeds)}
    ref = {m: np.nanmean(cube[m][:, [sidx[x] for x in B]], axis=(1, 2, 3)) for m in models}
    pairs = list(itertools.combinations(models, 2))

    rows = []
    for ns, nw, no in itertools.product(range(1, len(A) + 1), range(1, len(W) + 1), range(1, len(O) + 1)):
        subsets = list(itertools.product(itertools.combinations(A, ns),
                                         itertools.combinations(range(len(W)), nw),
                                         itertools.combinations(range(len(O)), no)))
        if len(subsets) > max_subsets:
            subsets = [subsets[i] for i in rng.choice(len(subsets), max_subsets, replace=False)]
        des = []  # per subset: {model: item-level mean on the design}
        for sd, ws, os_ in subsets:
            sub = {m: np.nanmean(cube[m][:, [sidx[x] for x in sd]][:, :, list(ws)][:, :, :, list(os_)],
                                 axis=(1, 2, 3)) for m in models}
            des.append(sub)
        agree = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(items), len(items))
            r_means = {m: ref[m][idx].mean() for m in models}
            for sub in des:
                d_means = {m: sub[m][idx].mean() for m in models}
                agree.append(np.mean([np.sign(d_means[a] - d_means[b]) == np.sign(r_means[a] - r_means[b])
                                      for a, b in pairs]))
        agree = np.array(agree)
        rows.append({"seeds": ns, "wordings": nw, "orders": no, "runs_per_item": ns * nw * no,
                     "mean_pair_agreement": agree.mean(),
                     "share_all_pairs_agree": float(np.mean(agree == 1.0))})
    return pd.DataFrame(rows).sort_values(["runs_per_item", "seeds", "wordings", "orders"])


# ------------------------------------------------------------------ figures

def figures(card: pd.DataFrame, flips: pd.DataFrame, ranks: pd.DataFrame, budget: pd.DataFrame,
            outdir: Path, title_note: str, chance: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: greedy vs mean vs reliable accuracy
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(card) + 2), 4.2))
    names = card["model"].tolist(); x = np.arange(len(names)); w = 0.26
    for i, (k, lab) in enumerate([("greedy_acc", "Greedy (leaderboard)"), ("mean_acc", "Mean (sampled)"),
                                  ("reliable_acc", "Reliable (all runs correct)")]):
        y = card[k].to_numpy() * 100
        err = np.vstack([y - card[f"{k}_lo"] * 100, card[f"{k}_hi"] * 100 - y])
        ax.bar(x + (i - 1) * w, y, w, yerr=err, capsize=3, label=lab)
    if chance:
        ax.axhline(chance * 100, ls="--", lw=1, color="grey", label=f"Chance ({chance * 100:.0f}%)")
    ax.set_xticks(x, names); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Fig. 1  Accuracy vs reliability{title_note}", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(outdir / "fig1_accuracy_vs_reliability.png", dpi=200); plt.close(fig)

    # Fig 2: flip rate by factor
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(flips) + 2), 4.2))
    facs = [("flip_seed", "Seed (noise floor)"), ("flip_wording", "Wording"), ("flip_order", "Option order")]
    for i, (k, lab) in enumerate(facs):
        y = flips[k].to_numpy() * 100
        if np.all(np.isnan(y)):
            continue
        err = np.vstack([y - flips[f"{k}_lo"] * 100, flips[f"{k}_hi"] * 100 - y])
        ax.bar(x + (i - 1) * w, y, w, yerr=err, capsize=3, label=lab)
    ax.set_xticks(x, names); ax.set_ylabel("Correctness flip rate (%)")
    ax.set_title(f"Fig. 2  Flip rate by factor{title_note}", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(outdir / "fig2_flip_rate_by_factor.png", dpi=200); plt.close(fig)

    if len(card) >= 2:  # Fig 3: bump chart of ranks across metrics
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for m in ranks.index:
            ax.plot(range(len(METRICS)), ranks.loc[m, METRICS], marker="o", label=m)
        ax.set_xticks(range(len(METRICS)), ["Greedy", "Mean", "Reliable", "Worst wording"])
        ax.invert_yaxis(); ax.set_yticks(range(1, len(ranks) + 1)); ax.set_ylabel("Rank (1 = best)")
        ax.set_title(f"Fig. 3  Model rank by metric{title_note}", fontsize=10)
        ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout(); fig.savefig(outdir / "fig3_rank_bump.png", dpi=200); plt.close(fig)

    if not budget.empty:  # Fig 4: agreement vs test budget
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.scatter(budget["runs_per_item"], budget["share_all_pairs_agree"] * 100)
        for _, r in budget.iterrows():
            ax.annotate(f"{r.seeds}s×{r.wordings}w×{r.orders}o",
                        (r.runs_per_item, r.share_all_pairs_agree * 100), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.axhline(90, ls="--", lw=1, color="grey")
        ax.set_xlabel("Runs per item in the design"); ax.set_ylabel("Resamples where all pairs agree (%)")
        ax.set_title(f"Fig. 4  Agreement with held-out runs{title_note}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(outdir / "fig4_budget_curve.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------ main

def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")["analysis"]
    roles = {m["name"]: m.get("role") for m in load_yaml("configs/models.yaml")["models"]}
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=cfg["n_bootstrap"])
    ap.add_argument("--ranking-only", action="store_true", help="drop smoke-test / size-ladder models")
    ap.add_argument("--title-note", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    level = cfg["ci_level"]

    df = pd.read_csv(a.scored)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    models = sorted(df["model"].unique())
    if a.ranking_only:
        models = [m for m in models if roles.get(m) == "ranking"]
    # keep only items every model has fully run, so all models are compared on the same items
    counts = df[df["model"].isin(models)].groupby(["model", "item_id"]).size().unstack()
    full = counts.max().max()
    items = sorted(counts.columns[(counts == full).all(axis=0)])
    print(f"{len(models)} models, {len(items)} complete items, {full} runs per item")

    tables = {m: item_tables(df, m, items) for m in models}
    boots = bootstrap(tables, len(items), a.n_boot, rng)
    outdir = RESULTS / "analysis" / a.scored.stem
    outdir.mkdir(parents=True, exist_ok=True)

    # reliability card
    rows = []
    for m in models:
        pt = point_metrics(tables[m])
        row = {"model": m}
        for k, v in pt.items():
            row[k] = v
            row[f"{k}_lo"], row[f"{k}_hi"] = ci(boots[m][k], level)
        row["mean_minus_reliable"] = pt["mean_acc"] - pt["reliable_acc"]
        row["mean_minus_reliable_lo"], row["mean_minus_reliable_hi"] = ci(boots[m]["mean_acc"] - boots[m]["reliable_acc"], level)
        rows.append(row)
    card = pd.DataFrame(rows)
    card.to_csv(outdir / "reliability_card_full.csv", index=False)

    def fmt(r, k):
        if np.isnan(r[k]):
            return "n/a"
        return f"{r[k] * 100:.1f} [{r[f'{k}_lo'] * 100:.1f}, {r[f'{k}_hi'] * 100:.1f}]"
    show = ["greedy_acc", "mean_acc", "reliable_acc", "worst_wording_acc",
            "flip_seed", "flip_wording", "flip_order", "parse_fail_rate"]
    md = ["| Model | " + " | ".join(show) + " |", "|---|" + "---:|" * len(show)]
    for _, r in card.iterrows():
        md.append(f"| {r['model']} | " + " | ".join(fmt(r, k) for k in show) + " |")
    note = (f"\n\nAll values in %, with {int(level * 100)}% bootstrap CIs over {len(items)} items "
            f"({a.n_boot} resamples). Flip rates: share of run pairs that differ only in that factor whose "
            "correctness differs (wording/order: greedy cells; seed: sampled runs in the same cell).")
    (outdir / "reliability_card.md").write_text("\n".join(md) + note + "\n")

    # pairwise ranking stability
    prow, ranks = [], pd.DataFrame(index=models)
    for k in METRICS:
        ranks[k] = card.set_index("model")[k].rank(ascending=False, method="min")
    for a_, b_ in itertools.combinations(models, 2):
        r = {"model_a": a_, "model_b": b_}
        for k in METRICS:
            d = boots[a_][k] - boots[b_][k]
            pt = point_metrics(tables[a_])[k] - point_metrics(tables[b_])[k]
            lo, hi = ci(d, level)
            r[f"{k}_diff"] = pt; r[f"{k}_lo"] = lo; r[f"{k}_hi"] = hi
            r[f"{k}_sig"] = bool(lo > 0 or hi < 0)
            r[f"{k}_swap_share"] = float(np.mean(np.sign(d) != np.sign(pt))) if pt != 0 else np.nan
        r["discordant_greedy_vs_reliable"] = bool(np.sign(r["greedy_acc_diff"]) * np.sign(r["reliable_acc_diff"]) < 0)
        r["H3_sig_discordant"] = bool(r["discordant_greedy_vs_reliable"] and r["greedy_acc_sig"] and r["reliable_acc_sig"])
        prow.append(r)
    pairwise = pd.DataFrame(prow)
    pairwise.to_csv(outdir / "pairwise.csv", index=False)

    budget = budget_curve(df, models, items, min(a.n_boot, 300), rng)
    if not budget.empty:
        budget.to_csv(outdir / "budget_curve.csv", index=False)

    flips = card[["model"] + [c for c in card.columns if c.startswith("flip_") or c.startswith("change_")]]
    chance = float((1 / df.drop_duplicates("item_id").set_index("item_id").loc[items, "n_options"]).mean())
    figures(card, flips, ranks, budget, outdir, a.title_note, chance)

    summary = {
        "n_items": len(items), "models": models, "runs_per_item": int(full),
        "H1_mean_minus_reliable_pts": {r["model"]: round(r["mean_minus_reliable"] * 100, 1) for _, r in card.iterrows()},
        "H1_holds_for_all": bool((card["mean_minus_reliable"] >= 0.15).all()),
        "n_pairs": len(pairwise),
        "discordant_pairs_greedy_vs_reliable": int(pairwise["discordant_greedy_vs_reliable"].sum()) if len(pairwise) else 0,
        "H3_significant_discordant_pairs": int(pairwise["H3_sig_discordant"].sum()) if len(pairwise) else 0,
    }
    if len(models) >= 3:
        from scipy.stats import kendalltau  # summary only: tau moves in big steps with 5-6 models
        summary["kendall_tau_greedy_vs_reliable"] = float(kendalltau(ranks["greedy_acc"], ranks["reliable_acc"]).statistic)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print((outdir / "reliability_card.md").read_text())
    print(json.dumps(summary, indent=2))
    print(f"outputs in {outdir}")


if __name__ == "__main__":
    main()
