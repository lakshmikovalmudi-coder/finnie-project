"""Compliance node: every answer leaves with an education-not-advice disclaimer."""

import re

from src.state import FinnieState

DISCLAIMER = (
    "_Finnie provides financial education, not personalized financial, investment, "
    "or tax advice. Consider a licensed professional before making decisions._"
)

# Advice-like phrasing the specialists should have avoided; softened as a backstop.
ADVICE_PATTERNS = [
    (re.compile(r"\byou should (buy|sell|invest in)\b", re.I), r"some investors choose to \1"),
    (
        re.compile(r"\bI recommend (buying|selling|investing in)\b", re.I),
        r"one option people consider is \1",
    ),
    (re.compile(r"\bguaranteed returns?\b", re.I), "expected (not guaranteed) returns"),
]


def soften(text: str) -> str:
    for pattern, replacement in ADVICE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


async def run(state: FinnieState) -> dict:
    answer = soften(state.get("answer", "")).rstrip()
    if DISCLAIMER not in answer:
        answer = f"{answer}\n\n---\n{DISCLAIMER}"
    return {"answer": answer}
