"""One fixed prompt template per task. Do not tune it per model: that would confound the study."""
from __future__ import annotations

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
