"""BFCL "multiple" checker: function name + argument values against the possible-answer file.

This is a faithful port of the Python path of BFCL's AST checker
(gorilla, berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py,
commit 6ea57973, Apache-2.0): `multiple_function_checker` -> `simple_function_checker`
with `type_checker`, `string_checker`, `list_checker`, `dict_checker`, `list_dict_checker`.
Differences, all stated in the paper:
  * Models answer with JSON, not Python call syntax, so the parser is ours (`parse_call`).
  * No model-specific name conversion: every model sees and must return the names as written.

Rules (same as BFCL): exactly one call; the name must match the expected function; every
required parameter present; no parameter outside the schema / answer key; every value in the
allowed list (strings compared after removing spaces and ,./-_*^ and lower-casing; int accepted
for float); a parameter may be omitted only if "" is in its allowed list.
"""
from __future__ import annotations

import json
import re

PYTHON_TYPE_MAPPING = {"string": str, "integer": int, "float": float, "boolean": bool,
                       "array": list, "tuple": list, "dict": dict, "any": str}
PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple"]


# ------------------------------------------------------------------ parsing

def _first_json_object(s: str) -> str | None:
    """Return the first balanced {...} span, ignoring braces inside JSON strings."""
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc: esc = False
                elif c == "\\": esc = True
                elif c == '"': in_str = False
            elif c == '"': in_str = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        start = s.find("{", start + 1)
    return None


def parse_call(raw: str) -> tuple[list[dict] | None, str]:
    """Parse model text into BFCL's output form [{name: {arg: value}}].

    Accepted: one JSON object {"name": ..., "arguments": {...}} (key "parameters" also accepted),
    optionally inside ```json fences, optionally wrapped in a one-element list, and
    {"<function name>": {...args}} (BFCL's own shape). Anything else is a parse failure.
    """
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s).strip()
    obj = None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        span = _first_json_object(s)
        if span is None:  # an opening brace that never closes = cut off at max_new_tokens
            return None, "truncated_json" if "{" in s else "no_json"
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            return None, "invalid_json"
    if isinstance(obj, list):
        if len(obj) != 1 or not isinstance(obj[0], dict):
            return ([{"__multiple__": {}}] * max(len(obj), 2), "wrong_count") if len(obj) > 1 else (None, "bad_list")
        obj = obj[0]
    if not isinstance(obj, dict):
        return None, "not_object"
    if "name" in obj and isinstance(obj["name"], str):
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):            # some models double-encode the arguments
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None, "args_not_json"
        if not isinstance(args, dict):
            return None, "args_not_object"
        return [{obj["name"]: args}], "name_arguments"
    if len(obj) == 1 and isinstance(next(iter(obj.values())), dict):
        return [obj], "bfcl_shape"
    return None, "no_function_name"


# ------------------------------------------------------------------ checker (port)

def get_possible_answer_type(possible_answer: list):
    for answer in possible_answer:
        if answer != "":
            return type(answer)
    return None


def type_checker(param, value, possible_answer, expected_type_description,
                 expected_type_converted, nested_type_converted):
    result = {"valid": True, "error": [], "is_variable": False, "error_type": "type_error:simple"}
    is_variable = False
    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type is not None and possible_answer_type != expected_type_converted:
        is_variable = True

    if type(value) == expected_type_converted:
        if nested_type_converted is None:
            result["is_variable"] = is_variable
            return result
        for possible_answer_item in possible_answer:
            flag = True
            if type(possible_answer_item) == list:
                for value_item in value:
                    r = type_checker(param, value_item, possible_answer_item,
                                     str(nested_type_converted), nested_type_converted, None)
                    if not r["valid"]:
                        flag = False
                        break
            if flag:
                return {"valid": True, "error": [], "is_variable": is_variable}
        result["valid"] = False
        result["error"] = [f"Nested type checking failed for parameter {param!r}."]
        result["error_type"] = "type_error:nested"

    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type is not None and type(value) == possible_answer_type:
        result["is_variable"] = True
        return result

    result["valid"] = False
    result["error"].append(f"Incorrect type for parameter {param!r}. Expected {expected_type_description}, "
                           f"got {type(value).__name__}.")
    result["error_type"] = "type_error:simple"
    return result


def standardize_string(input_string: str) -> str:
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", input_string).lower().replace("'", '"')


def string_checker(param, model_output, possible_answer):
    std = [standardize_string(p) for p in possible_answer if type(p) == str]
    if standardize_string(model_output) not in std:
        return {"valid": False, "error": [f"Invalid value for {param!r}: {model_output!r}."],
                "error_type": "value_error:string"}
    return {"valid": True, "error": []}


def list_checker(param, model_output, possible_answer):
    out = [standardize_string(v) if type(v) == str else v for v in list(model_output)]
    std = [[standardize_string(x) if type(x) == str else x for x in p] for p in possible_answer]
    if out not in std:
        return {"valid": False, "error": [f"Invalid value for {param!r}: {model_output!r}."],
                "error_type": "value_error:list/tuple"}
    return {"valid": True, "error": []}


def dict_checker(param, model_output, possible_answers):
    result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
    for possible_answer in possible_answers:
        if possible_answer == "":
            continue
        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        flag = True
        for key, value in model_output.items():
            if key not in possible_answer:
                result.update(error=[f"Unexpected dict key {key!r}."], error_type="value_error:dict_key")
                flag = False
                break
            sv = standardize_string(value) if type(value) == str else value
            spa = [standardize_string(x) if type(x) == str else x for x in possible_answer[key]]
            if sv not in spa:
                result.update(error=[f"Invalid dict value for {key!r}."], error_type="value_error:dict_value")
                flag = False
                break
        for key, value in possible_answer.items():
            if key not in model_output and "" not in value:
                result.update(error=[f"Missing dict key {key!r}."], error_type="value_error:dict_key")
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}
    return result


def list_dict_checker(param, model_output, possible_answers):
    result = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}
    for answer in possible_answers:
        flag = True
        if len(model_output) != len(answer):
            result = {"valid": False, "error": ["Wrong number of dictionaries."],
                      "error_type": "value_error:list_dict_count"}
            continue
        for i in range(len(model_output)):
            result = dict_checker(param, model_output[i], [answer[i]])
            if not result["valid"]:
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}
    return result


def simple_function_checker(func_description: dict, model_output: dict, possible_answer: dict) -> dict:
    possible_answer = list(possible_answer.values())[0]
    func_name = func_description["name"]
    param_details = func_description["parameters"]["properties"]
    required_params = func_description["parameters"].get("required", [])
    result = {"valid": True, "error": [], "error_type": "simple_function_checker:unclear"}

    if func_name not in model_output:
        return {"valid": False, "error": [f"Function {func_name!r} not called."],
                "error_type": "simple_function_checker:wrong_func_name"}
    model_params = model_output[func_name]

    for param in required_params:
        if param not in model_params:
            return {"valid": False, "error": [f"Missing required parameter {param!r}."],
                    "error_type": "simple_function_checker:missing_required"}

    for param, value in model_params.items():
        if param not in param_details or param not in possible_answer:
            return {"valid": False, "error": [f"Unexpected parameter {param!r}."],
                    "error_type": "simple_function_checker:unexpected_param"}
        etd = param_details[param]["type"]
        if etd not in PYTHON_TYPE_MAPPING:
            return {"valid": False, "error": [f"Unsupported schema type {etd!r}."],
                    "error_type": "schema:unsupported_type"}
        etc = PYTHON_TYPE_MAPPING[etd]
        ntc = None
        if etd in PYTHON_NESTED_TYPE_CHECK_LIST:
            nested = param_details[param].get("items", {}).get("type")
            ntc = PYTHON_TYPE_MAPPING.get(nested) if nested else None
        if etd == "tuple" and type(value) == tuple:
            value = list(value)
        if etd == "float" and type(value) == int:
            value = float(value)

        tc = type_checker(param, value, possible_answer[param], etd, etc, ntc)
        if not tc["valid"]:
            return tc
        if not tc["is_variable"]:
            if etc == dict:
                r = dict_checker(param, value, possible_answer[param])
                if not r["valid"]:
                    return r
                continue
            elif etc == list and ntc == dict:
                r = list_dict_checker(param, value, possible_answer[param])
                if not r["valid"]:
                    return r
                continue
            elif etc == str:
                r = string_checker(param, value, possible_answer[param])
                if not r["valid"]:
                    return r
                continue
            elif etc == list:
                r = list_checker(param, value, possible_answer[param])
                if not r["valid"]:
                    return r
                continue
        if value not in possible_answer[param]:
            return {"valid": False, "error": [f"Invalid value for {param!r}: {value!r}."],
                    "error_type": "value_error:others"}

    for param in possible_answer:
        if param not in model_params and "" not in possible_answer[param]:
            return {"valid": False, "error": [f"Parameter {param!r} missing and not optional."],
                    "error_type": "simple_function_checker:missing_optional"}
    return result


def multiple_function_checker(func_descriptions: list, model_output: list, possible_answers: list) -> dict:
    if len(model_output) != len(possible_answers):
        return {"valid": False, "error": ["Wrong number of functions."],
                "error_type": "multiple_function_checker:wrong_count"}
    expected = list(possible_answers[0].keys())[0]
    desc = next(f for f in func_descriptions if f["name"] == expected)
    return simple_function_checker(desc, model_output[0], possible_answers[0])


def check(raw: str, functions: list, ground_truth: list) -> dict:
    """Parse + check one model output. Returns parse status, correctness and error type."""
    call, rule = parse_call(raw)
    if call is None:
        return {"parse_fail": True, "parse_rule": rule, "correct": 0, "error_type": "parse_failure",
                "called": None}
    if rule == "wrong_count":
        return {"parse_fail": False, "parse_rule": rule, "correct": 0,
                "error_type": "multiple_function_checker:wrong_count", "called": None}
    r = multiple_function_checker(functions, call, ground_truth)
    name = next(iter(call[0]))
    return {"parse_fail": False, "parse_rule": rule, "correct": int(r["valid"]),
            "error_type": "ok" if r["valid"] else r["error_type"],
            # canonical form of the call, used for answer-change rates across conditions
            "called": json.dumps({name: call[0][name]}, sort_keys=True, ensure_ascii=False)}
