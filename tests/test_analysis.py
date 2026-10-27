"""Metric definitions checked on synthetic models whose behaviour we control exactly."""
import itertools

import numpy as np
import pandas as pd

from src.analyse import budget_curve, item_tables, point_metrics

ITEMS = [f"i{k}" for k in range(20)]
W, O, S = ["w0", "w1", "w2"], ["o0", "o1"], [0, 1, 2, 3]


def make(model, rule):
    """rule(item_index, wording, order, seed_or_None) -> correct (0/1)."""
    rows = []
    for k, it in enumerate(ITEMS):
        for w, o in itertools.product(W, O):
            for s in S + [None]:
                c = int(rule(k, w, o, s))
                rows.append(dict(model=model, item_id=it, wording_id=w, order_id=o, seed=s,
                                 decoding="greedy" if s is None else "sampled", correct=c,
                                 parse_fail=False, chosen_text="gold" if c else "wrong"))
    return pd.DataFrame(rows)


def test_perfect_model():
    df = make("perfect", lambda k, w, o, s: 1)
    m = point_metrics(item_tables(df, "perfect", ITEMS))
    assert m["greedy_acc"] == m["mean_acc"] == m["reliable_acc"] == 1
    assert m["flip_seed"] == m["flip_wording"] == m["flip_order"] == 0


def test_order_only_model():
    df = make("ord", lambda k, w, o, s: o == "o0")      # right only in original order
    m = point_metrics(item_tables(df, "ord", ITEMS))
    assert m["greedy_acc"] == 1 and m["mean_acc"] == 0.5 and m["reliable_acc"] == 0
    assert m["flip_order"] == 1 and m["flip_wording"] == 0 and m["flip_seed"] == 0


def test_wording_and_worst_wording():
    df = make("wd", lambda k, w, o, s: not (w == "w2" and k < 10))  # w2 breaks half the items
    m = point_metrics(item_tables(df, "wd", ITEMS))
    assert m["worst_wording_acc"] == 0.5
    assert np.isclose(m["flip_wording"], 0.5 * (2 / 3))           # 2 of 3 wording pairs flip on half the items
    assert m["flip_order"] == 0


def test_seed_noise():
    df = make("noisy", lambda k, w, o, s: True if s is None else s % 2 == 0)
    m = point_metrics(item_tables(df, "noisy", ITEMS))
    assert m["reliable_acc"] == 0 and m["mean_acc"] == 0.5
    assert np.isclose(m["flip_seed"], 4 / 6)                       # 4 of 6 seed pairs differ


def test_budget_curve_clear_gap_agrees():
    rng = np.random.default_rng(0)
    df = pd.concat([make("strong", lambda k, w, o, s: k < 18), make("weak", lambda k, w, o, s: k < 4)])
    b = budget_curve(df, ["strong", "weak"], ITEMS, n_boot=50, rng=rng)
    assert (b["share_all_pairs_agree"] == 1).all()
