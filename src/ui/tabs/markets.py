"""Markets tab: cached quotes for a watchlist."""

import streamlit as st

from src.tools.market_data import get_quotes

DEFAULT_WATCHLIST = "SPY, QQQ, AAPL, MSFT"


def render() -> None:
    st.subheader("Markets")
    st.caption(
        "Quotes are cached for 30 minutes to stay within Alpha Vantage's free tier "
        "(25 requests/day)."
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
    if any(q["source"] == "mock" for q in quotes.values()):
        st.caption("🧪 Sample data shown — add ALPHA_VANTAGE_API_KEY to .env for live quotes.")
