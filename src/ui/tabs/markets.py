"""Markets tab: cached quotes and price history for a watchlist."""

import pandas as pd
import streamlit as st

from src.tools.market_data import get_histories, get_quotes
from src.tools.portfolio_math import risk_metrics

DEFAULT_WATCHLIST = "SPY, QQQ, AAPL, MSFT"


def _closes(histories: dict[str, dict]) -> pd.DataFrame:
    """Wide frame of daily closes: one column per symbol, indexed by date."""
    frame = pd.DataFrame(
        {s: pd.Series({b["date"]: b["close"] for b in h["bars"]}) for s, h in histories.items()}
    )
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def render() -> None:
    st.subheader("Markets")
    st.caption(
        "Quotes are cached for 30 minutes and daily history for 12 hours, to stay within "
        "Alpha Vantage's free tier (25 requests/day)."
    )

    raw = st.text_input("Tickers (comma-separated, max 5)", DEFAULT_WATCHLIST)
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()][:5]
    if not symbols:
        return

    quotes = get_quotes(symbols)
    missing = [s for s in symbols if s not in quotes]
    cols = st.columns(max(len(quotes), 1))
    for col, quote in zip(cols, quotes.values(), strict=False):
        col.metric(
            label=f"{quote['symbol']}{' (sample)' if quote['source'] == 'mock' else ''}",
            value=f"${quote['price']:,.2f}",
            delta=f"{quote['change']:+.2f} ({quote['change_percent']:+.2f}%)",
        )
    if missing:
        st.info(f"No data for: {', '.join(missing)}")

    histories = get_histories(list(quotes))
    if histories:
        closes = _closes(histories)
        st.markdown("**Performance comparison** · each line starts at 100")
        # Rebase from each symbol's first close so different prices share one scale.
        st.line_chart(closes / closes.apply(lambda c: c.dropna().iloc[0]) * 100)

        symbol = st.selectbox("Price history for", list(histories))
        bars = histories[symbol]["bars"]
        stats = risk_metrics({symbol: 1.0}, {symbol: bars})
        if stats:
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Return since {stats['start_date']}", f"{stats['period_return']:+.1%}")
            m2.metric(
                "Volatility (annual)",
                f"{stats['volatility']:.1%}",
                help="Standard deviation of daily returns, scaled to a year",
            )
            m3.metric(
                "Max drawdown", f"{stats['max_drawdown']:.1%}", help="Worst peak-to-trough drop"
            )
        st.line_chart(closes[symbol].dropna().rename(f"{symbol} close ($)"))

    if any(q["source"] == "mock" for q in quotes.values()) or any(
        h["source"] == "mock" for h in histories.values()
    ):
        st.caption("🧪 Sample data shown — add ALPHA_VANTAGE_API_KEY to .env for live data.")
