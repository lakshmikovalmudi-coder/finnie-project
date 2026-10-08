import pytest

from src.tools.goal_math import future_value, monthly_savings_needed, projection_series


def test_plan_reaches_target():
    plan = monthly_savings_needed(target=50_000, years=5, annual_return=0.06)
    assert future_value(0, plan.monthly_contribution, 0.06, 5) == pytest.approx(50_000, rel=1e-3)


def test_zero_return_is_simple_division():
    plan = monthly_savings_needed(target=12_000, years=1, annual_return=0.0)
    assert plan.monthly_contribution == 1_000


def test_already_funded_goal_needs_nothing():
    plan = monthly_savings_needed(target=1_000, years=10, annual_return=0.05, current_savings=5_000)
    assert plan.monthly_contribution == 0


def test_invalid_years():
    with pytest.raises(ValueError):
        monthly_savings_needed(target=1_000, years=0)


def test_projection_has_one_point_per_year():
    assert len(projection_series(100, 10, 0.06)) == 11
