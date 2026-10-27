import pytest

from src.make_variants import apply_order, auto_check, reshuffle
from src.parse_outputs import parse_letter
from src.prompts import mmlu_user_message


@pytest.mark.parametrize("raw,expected", [
    ("C", "C"), (" C\n", "C"), ("C.", "C"), ("(C)", "C"), ("C)", "C"),
    ("C. Paris", "C"), ("Answer: C", "C"), ("The answer is C.", "C"),
    ("answer: (B)", "B"), ("**D**", "D"), ("c", None), ("", None),
    ("I think the", None), ("C or D", None), ("A, B", None), ("K", None),
    ("The correct answer is E", "E"),
])
def test_parse(raw, expected):
    assert parse_letter(raw, 10)[0] == expected


def test_parse_out_of_range():
    assert parse_letter("E", 4) == (None, "letter_out_of_range")


@pytest.mark.parametrize("seed", range(50))
def test_reshuffle_moves_gold_and_keeps_text(seed):
    opts = [f"opt{i}" for i in range(10)]
    gold = seed % 10
    perm = reshuffle(10, gold, seed)
    assert sorted(perm) == list(range(10))
    new_opts, new_gold = apply_order(opts, gold, perm)
    assert new_gold != gold
    assert new_opts[new_gold] == opts[gold]


def test_reshuffle_deterministic():
    assert reshuffle(10, 3, 123) == reshuffle(10, 3, 123)


def test_prompt_letters():
    p = mmlu_user_message("What?", ["x", "y", "z"])
    assert "A. x" in p and "C. z" in p and "D." not in p


def test_auto_check_numbers():
    assert "numbers_changed" in auto_check("What is 2+3?", "What is 2 plus 4?", ["5"])
    assert auto_check("Compute the value of 2+3 here.", "Here, compute the value of 2+3.", ["5"]) == []
