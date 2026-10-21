"""Portfolio tab: upload holdings and see diversification and risk metrics."""

import altair as alt
import pandas as pd
import streamlit as st

from src.agents.portfolio import analyze_portfolio
from src.aio import run_sync
from src.config import DATA_DIR


def _donut(weights: pd.Series, label: str) -> alt.Chart:
    """Share-of-portfolio donut; weights are fractions that sum to 1."""
    data = weights.rename("weight").rename_axis(label).reset_index()
    return (
        alt.Chart(data)
        .mark_arc(innerRadius=60)
        .encode(
            theta="weight:Q",
            color=alt.Color(f"{label}:N", legend=alt.Legend(title=None)),
            tooltip=[f"{label}:N", alt.Tooltip("weight:Q", format=".1%")],
        )
        .properties(height=260)
    )


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
    if not analysis["holdings"]:
        st.error(f"No price data for any holding: {', '.join(analysis['unpriced'])}")
        return
    risk = analysis["risk"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total value", f"${analysis['total_value']:,.2f}")
    c2.metric("Holdings", len(analysis["holdings"]))
    c3.metric(
        "Effective holdings",
        analysis["effective_holdings"],
        help="1 / HHI — how many equal-weight positions this behaves like",
    )
    if risk:
        c4.metric(
            "Volatility (annual)",
            f"{risk['volatility']:.1%}",
            help=f"How much the value typically swings in a year. Risk level: {risk['risk_level']}",
        )
        c5.metric(
            "Max drawdown",
            f"{risk['max_drawdown']:.1%}",
            help="Worst peak-to-trough drop over the period shown below",
        )

    df = pd.DataFrame(analysis["holdings"])
    left, right = st.columns(2)
    left.markdown("**By holding**")
    left.altair_chart(_donut(df.set_index("symbol")["weight"], "symbol"), width="stretch")
    right.markdown("**By asset class**")
    right.altair_chart(
        _donut(pd.Series(analysis["by_asset_class"]), "asset_class"), width="stretch"
    )

    if risk:
        df["volatility"] = df["symbol"].map(risk["holding_volatility"])
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "price": st.column_config.NumberColumn(format="$%.2f"),
            "value": st.column_config.NumberColumn(format="$%.2f"),
            "weight": st.column_config.NumberColumn(format="percent"),
            "volatility": st.column_config.NumberColumn("volatility (annual)", format="percent"),
        },
    )

    if risk:
        st.markdown(
            f"**Growth of $100** · {risk['start_date']} to {risk['end_date']} · "
            f"return {risk['period_return']:+.1%}"
        )
        growth = pd.DataFrame(risk["growth_of_100"])
        growth["date"] = pd.to_datetime(growth["date"])
        st.line_chart(growth.set_index("date")["value"])
        if risk["sample_data"]:
            st.caption("🧪 Some price history is sample data, not live.")

    for flag in analysis["flags"]:
        st.warning(flag)
    if analysis["considerations"]:
        st.markdown("**Things to learn about**")
        for note in analysis["considerations"]:
            st.info(note)
    if analysis["unpriced"]:
        st.info(f"No price data for: {', '.join(analysis['unpriced'])}")
    if risk and risk["missing_history"]:
        st.info(f"No price history for: {', '.join(risk['missing_history'])}")
