"""One fixed prompt template per task. Do not tune it per model: that would confound the study."""
from __future__ import annotations

import json

from .common import LETTERS

MMLU_INSTRUCTION = (
    "The following is a multiple-choice question. Answer with the letter of the correct "
    "option only, with no explanation."
)


def mmlu_user_message(question: str, options: list[str]) -> str:
    lines = [MMLU_INSTRUCTION, "", f"Question: {question.strip()}", "", "Options:"]
    lines += [f"{LETTERS[i]}. {o.strip()}" for i, o in enumerate(options)]
    lines += ["", "Answer:"]
    return "\n".join(lines)


def to_chat(user_message: str) -> list[dict]:
    """No system prompt: models differ in whether they support one, so leave it out for all."""
    return [{"role": "user", "content": user_message}]


BFCL_INSTRUCTION = (
    "You can call exactly one of the functions below to fulfil the user's request. Choose the "
    "right function and fill in its arguments with values from the request. Respond with only a "
    'JSON object of the form {"name": "<function name>", "arguments": {"<argument>": <value>}} '
    "and nothing else."
)


def bfcl_user_message(request: str, functions: list[dict]) -> str:
    """Functions are listed one compact JSON schema per line, in the order given (the order factor)."""
    lines = [BFCL_INSTRUCTION, "", "Functions:"]
    lines += [json.dumps(f, ensure_ascii=False) for f in functions]
    lines += ["", f"Request: {request.strip()}", "", "JSON:"]
    return "\n".join(lines)
