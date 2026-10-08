"""Deterministic savings math. The LLM explains these numbers; it never computes them."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GoalPlan:
    target: float
    years: float
    annual_return: float
    current_savings: float
    monthly_contribution: float
    total_contributed: float
    projected_growth: float

    def to_dict(self) -> dict:
        return asdict(self)


def future_value(present: float, monthly: float, annual_return: float, years: float) -> float:
    """Value after `years` of monthly contributions with monthly compounding."""
    n = round(years * 12)
    r = annual_return / 12
    if r == 0:
        return present + monthly * n
    growth = (1 + r) ** n
    return present * growth + monthly * (growth - 1) / r


def monthly_savings_needed(
    target: float, years: float, annual_return: float = 0.06, current_savings: float = 0.0
) -> GoalPlan:
    """Monthly contribution required to reach `target` in `years`."""
    if years <= 0:
        raise ValueError("years must be positive")
    n = round(years * 12)
    r = annual_return / 12
    growth = (1 + r) ** n if r else 1.0
    remaining = target - current_savings * growth
    if remaining <= 0:
        monthly = 0.0
    elif r == 0:
        monthly = remaining / n
    else:
        monthly = remaining * r / (growth - 1)
    total_contributed = current_savings + monthly * n
    return GoalPlan(
        target=target,
        years=years,
        annual_return=annual_return,
        current_savings=current_savings,
        monthly_contribution=round(monthly, 2),
        total_contributed=round(total_contributed, 2),
        projected_growth=round(max(target - total_contributed, 0.0), 2),
    )


def projection_series(
    monthly: float, years: float, annual_return: float, current_savings: float = 0.0
) -> list[dict[str, float]]:
    """Year-by-year balance for charting in the Goals tab."""
    return [
        {"year": y, "balance": round(future_value(current_savings, monthly, annual_return, y), 2)}
        for y in range(int(years) + 1)
    ]
