"""Portfolio tab: upload holdings and see diversification and risk metrics."""

import pandas as pd
import streamlit as st

from src.agents.portfolio import analyze_portfolio
from src.aio import run_sync
from src.config import DATA_DIR


def render() -> None:
    st.subheader("Portfolio Analysis")
    st.caption("CSV columns: `symbol`, `shares`, optional `asset_class` (equity, bond, cash…).")

    uploaded = st.file_uploader("Upload holdings CSV", type="csv")
    use_sample = st.button("Use sample portfolio")
    if uploaded is not None:
        st.session_state["portfolio_rows"] = pd.read_csv(uploaded).to_dict("records")
    elif use_sample:
        st.session_state["portfolio_rows"] = pd.read_csv(DATA_DIR / "sample_portfolio.csv").to_dict(
            "records"
        )

    rows = st.session_state.get("portfolio_rows")
    if not rows:
        st.info(
            "Upload a CSV or load the sample to begin. "
            "The Chat tab can then answer questions about it."
        )
        return

    try:
        analysis = run_sync(analyze_portfolio(rows))
    except ValueError as exc:
        st.error(str(exc))
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total value", f"${analysis['total_value']:,.2f}")
    c2.metric("Holdings", len(analysis["holdings"]))
    c3.metric(
        "Effective holdings",
        analysis["effective_holdings"],
        help="1 / HHI — how many equal-weight positions this behaves like",
    )

    df = pd.DataFrame(analysis["holdings"])
    st.bar_chart(df.set_index("symbol")["weight"])
    st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("**By asset class**")
    st.bar_chart(pd.Series(analysis["by_asset_class"], name="weight"))

    for flag in analysis["flags"]:
        st.warning(flag)
    if analysis["unpriced"]:
        st.info(f"No price data for: {', '.join(analysis['unpriced'])}")
