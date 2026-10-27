"""BFCL checker: BFCL's own answers must score 100%, and each typical mistake must be caught."""
import copy
import json
from pathlib import Path

import pytest

from src.bfcl_checker import check, parse_call
from src.common import ROOT, read_jsonl

ITEMS_PATH = ROOT / "data" / "sampled" / "bfcl_multiple_n200.jsonl"
needs_data = pytest.mark.skipif(not ITEMS_PATH.exists(), reason="run python -m src.sample_bfcl first")


def concretize(v):
    """Turn a possible-answer value into one concrete value (first allowed option, recursively)."""
    if isinstance(v, dict):
        return {k: concretize(next(x for x in vals if x != "")) for k, vals in v.items()
                if any(x != "" for x in vals)}
    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
        return [concretize(x) for x in v]
    return v


def oracle_call(item):
    name, args = next(iter(item["ground_truth"][0].items()))
    call = {k: concretize(next(x for x in vals if x != "")) for k, vals in args.items()
            if any(x != "" for x in vals)}
    return name, call


@needs_data
def test_oracle_passes_all_200():
    items = read_jsonl(ITEMS_PATH)
    assert len(items) == 200
    fails = []
    for it in items:
        name, call = oracle_call(it)
        r = check(json.dumps({"name": name, "arguments": call}), it["functions"], it["ground_truth"])
        if not r["correct"]:
            fails.append((it["item_id"], r["error_type"]))
    assert fails == []


@pytest.fixture
def item():
    return next(it for it in read_jsonl(ITEMS_PATH) if it["bfcl_id"] == "multiple_0")


@needs_data
def test_mistakes_are_caught(item):
    fns, gt = item["functions"], item["ground_truth"]
    ok = {"side1": 5, "side2": 4, "side3": 3}
    assert check(json.dumps({"name": "triangle_properties.get", "arguments": ok}), fns, gt)["correct"] == 1
    cases = {
        "wrong_func_name": {"name": "circle_properties.get", "arguments": {"radius": 3}},
        "missing_required": {"name": "triangle_properties.get", "arguments": {"side1": 5, "side2": 4}},
        "value_error": {"name": "triangle_properties.get", "arguments": {**ok, "side3": 6}},
        "unexpected_param": {"name": "triangle_properties.get", "arguments": {**ok, "colour": "red"}},
        "type_error": {"name": "triangle_properties.get", "arguments": {**ok, "side1": "5"}},
    }
    for expected, call in cases.items():
        r = check(json.dumps(call), fns, gt)
        assert r["correct"] == 0 and expected in r["error_type"], (expected, r)


@needs_data
def test_optional_and_string_normalisation():
    it = next(i for i in read_jsonl(ITEMS_PATH) if i["bfcl_id"] == "multiple_2")  # country_info.capital
    fns, gt = it["functions"], it["ground_truth"]
    assert check('{"name": "country_info.capital", "arguments": {"country": "brazil"}}', fns, gt)["correct"]
    assert check('{"name": "country_info.capital", "arguments": {"country": " Brazil."}}', fns, gt)["correct"]
    assert not check('{"name": "country_info.capital", "arguments": {"country": "Brasil"}}', fns, gt)["correct"]


@pytest.mark.parametrize("raw,rule", [
    ('{"name": "f", "arguments": {"a": 1}}', "name_arguments"),
    ('```json\n{"name": "f", "arguments": {"a": 1}}\n```', "name_arguments"),
    ('Sure! {"name": "f", "parameters": {"a": "}"}} hope this helps', "name_arguments"),
    ('[{"name": "f", "arguments": {"a": 1}}]', "name_arguments"),
    ('{"f": {"a": 1}}', "bfcl_shape"),
    ('{"name": "f", "arguments": "{\\"a\\": 1}"}', "name_arguments"),
    ('I would call f with a=1', "no_json"),
    ('{"name": "f", "arguments": {"a": 1', "truncated_json"),   # cut off at max_new_tokens
])
def test_parse_call(raw, rule):
    assert parse_call(raw)[1] == rule


def test_int_accepted_for_float():
    fns = [{"name": "g", "parameters": {"type": "dict", "properties": {"x": {"type": "float"}}, "required": ["x"]}}]
    gt = [{"g": {"x": [2.0]}}]
    assert check('{"name": "g", "arguments": {"x": 2}}', fns, gt)["correct"] == 1


def test_rewording_value_check():
    from src.make_variants import auto_check_bfcl
    gt = [{"hotel.book": {"city": ["New York", "NYC"], "nights": [3], "date": ["2024-05-01"], "pool": ["", True]}}]
    orig = "Book a hotel in New York for 3 nights starting 2024-05-01."
    assert auto_check_bfcl(orig, "Starting 2024-05-01, I need a New York hotel for 3 nights.", gt) == []
    bad = auto_check_bfcl(orig, "Starting May 1st 2024, I need a hotel in NYC for three nights.", gt)
    assert "arg_value_missing:nights" in bad and "arg_value_missing:date" in bad
    assert "arg_value_missing:city" in bad          # "New York" was in the original, so it must stay
