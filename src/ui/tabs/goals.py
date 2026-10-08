"""Goals tab: deterministic savings calculator with a projection chart."""

import pandas as pd
import streamlit as st

from src.tools.goal_math import monthly_savings_needed, projection_series


def render() -> None:
    st.subheader("Goal Planner")
    c1, c2 = st.columns(2)
    target = c1.number_input("Target amount ($)", min_value=0.0, value=50_000.0, step=1_000.0)
    years = c2.number_input("Years to goal", min_value=1, max_value=60, value=10)
    current = c1.number_input("Already saved ($)", min_value=0.0, value=5_000.0, step=500.0)
    rate = c2.slider("Assumed annual return (%)", 0.0, 12.0, 6.0, 0.5) / 100

    plan = monthly_savings_needed(target, years, rate, current)
    m1, m2, m3 = st.columns(3)
    m1.metric("Save per month", f"${plan.monthly_contribution:,.2f}")
    m2.metric("Total you contribute", f"${plan.total_contributed:,.2f}")
    m3.metric("Growth from returns", f"${plan.projected_growth:,.2f}")

    series = projection_series(plan.monthly_contribution, years, rate, current)
    st.line_chart(pd.DataFrame(series).set_index("year")["balance"])
    st.caption("Assumes a constant return compounded monthly. Real returns vary year to year.")
