"""Goal Planning agent: extracts goal parameters, computes savings deterministically, explains."""

import json

from pydantic import BaseModel, Field

from src.agents.base import explain
from src.llm import get_llm
from src.state import FinnieState
from src.tools.goal_math import monthly_savings_needed, projection_series

SYSTEM_PROMPT = """\
You explain a savings plan. The numbers were computed by a calculator — quote
them exactly, never recompute. Mention the assumed return rate and that real
returns vary year to year.
"""

EXTRACT_PROMPT = """\
Extract savings-goal parameters from the user's message. Use null for anything
not stated. annual_return is a decimal (7% -> 0.07).
"""


class GoalParams(BaseModel):
    target: float | None = Field(None, description="Target amount in dollars")
    years: float | None = Field(None, description="Years until the goal")
    annual_return: float | None = Field(None, description="Expected annual return, decimal")
    current_savings: float | None = Field(None, description="Amount already saved")


async def run(state: FinnieState) -> dict:
    extractor = get_llm(temperature=0).with_structured_output(GoalParams)
    params: GoalParams = await extractor.ainvoke(
        [("system", EXTRACT_PROMPT), ("human", state["query"])]
    )
    if not params.target or not params.years:
        return {
            "answer": "To plan a goal I need a **target amount** and **how many years** "
            'you have — e.g. *"How much should I save monthly to reach $50,000 in 5 years?"* '
            "You can also use the **Goals** tab calculator.",
            "data": {},
        }
    plan = monthly_savings_needed(
        target=params.target,
        years=params.years,
        annual_return=params.annual_return if params.annual_return is not None else 0.06,
        current_savings=params.current_savings or 0.0,
    )
    answer = await explain(SYSTEM_PROMPT, state["query"], json.dumps(plan.to_dict(), indent=2))
    series = projection_series(
        plan.monthly_contribution, plan.years, plan.annual_return, plan.current_savings
    )
    return {
        "answer": answer,
        "sources": ["Goal calculator"],
        "data": {"goal": plan.to_dict(), "projection": series},
    }
