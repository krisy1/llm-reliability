"""Answer parser for multiple-choice outputs.

Policy (state this in the paper): accept a small, fixed set of formats that unambiguously
name ONE option letter; everything else is a parse failure and is scored as wrong.

Accepted (case-insensitive for the wrapper words, letter must be upper-case or alone):
    "C"   "C."   "C)"   "(C)"   "C. Paris"   "Answer: C"   "The answer is C"   "**C**"
Rejected:
    ""   "c"   "I think..." (I is a word, not option I)   "C or D"   "A, B"   "K" (not offered)
"Letter + space + text" is deliberately NOT accepted: "I think" / "A good..." would be misread.
"""
from __future__ import annotations

import re

from .common import LETTERS

_PATTERNS = [
    ("bare", re.compile(r"^\(?([A-J])\)?[.):]?$")),                                  # C  (C)  C.  C)
    ("letter_then_text", re.compile(r"^\(?([A-J])[.):]\s+\S")),                       # C. Paris
    ("answer_prefix", re.compile(r"^(?:the\s+)?(?:correct\s+)?(?:answer|option)\s*(?:is)?\s*[:\-]?\s*\(?((?-i:[A-J]))\)?(?:[.):]|\s|$)",
                                 re.IGNORECASE)),
]


def parse_letter(raw: str, n_options: int) -> tuple[str | None, str]:
    """Return (letter or None, rule_name)."""
    s = raw.strip().replace("**", "").strip()
    s = s.split("\n")[0].strip()
    if not s:
        return None, "empty"
    for name, pat in _PATTERNS:
        m = pat.match(s)
        if m:
            letter = m.group(1)
            if LETTERS.index(letter) >= n_options:
                return None, "letter_out_of_range"
            return letter, name
    return None, "no_match"
