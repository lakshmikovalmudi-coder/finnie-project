import pandas as pd
import pytest

from src.tools.portfolio_math import (
    CONSIDER_CONCENTRATION,
    CONSIDER_FEW_HOLDINGS,
    CONSIDER_ONE_ASSET_CLASS,
    analyze,
    load_holdings,
    risk_metrics,
)


def holdings(*rows: dict) -> pd.DataFrame:
    return load_holdings(pd.DataFrame(list(rows)))


def bars(*closes: float) -> list[dict]:
    return [{"date": f"2026-01-{day:02d}", "close": c} for day, c in enumerate(closes, 1)]


EVEN_FIVE = [
    {"symbol": s, "shares": 1, "asset_class": c}
    for s, c in [("A", "equity"), ("B", "equity"), ("C", "equity"), ("D", "bond"), ("E", "cash")]
]


# --- load_holdings: input validation ---------------------------------------------


def test_columns_and_symbols_are_normalized():
    df = holdings({"Symbol": " aapl ", "Shares": "3", "Asset_Class": "Bond"})
    assert df.to_dict("records") == [{"symbol": "AAPL", "shares": 3, "asset_class": "bond"}]


def test_missing_asset_class_defaults_to_equity():
    df = holdings({"symbol": "VTI", "shares": 1}, {"symbol": "BND", "shares": 1})
    assert df["asset_class"].tolist() == ["equity", "equity"]


def test_duplicate_tickers_are_merged():
    df = holdings({"symbol": "VTI", "shares": 2}, {"symbol": "vti", "shares": 3})
    assert df.to_dict("records") == [{"symbol": "VTI", "shares": 5, "asset_class": "equity"}]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"symbol": "AAPL"}], "missing columns"),
        ([{"symbol": "AAPL", "shares": -1}], "negative"),
        ([{"symbol": "AAPL", "shares": None}], "share count"),
        ([{"symbol": None, "shares": 1}], "symbol"),
    ],
)
def test_bad_rows_are_rejected(rows, message):
    with pytest.raises(ValueError, match=message):
        holdings(*rows)


def test_empty_csv_is_rejected():
    with pytest.raises(ValueError, match="no rows"):
        load_holdings(pd.DataFrame(columns=["symbol", "shares"]))


def test_non_numeric_shares_are_rejected():
    with pytest.raises(ValueError):
        holdings({"symbol": "AAPL", "shares": "ten"})


# --- analyze: weights, flags, considerations -----------------------------------------


def test_concentration_flag_and_weights():
    result = analyze(
        holdings({"Symbol": "aapl", "Shares": 90}, {"Symbol": "bnd", "Shares": 10}),
        {"AAPL": 10.0, "BND": 10.0},
    )
    assert result["total_value"] == 1_000
    assert any("AAPL" in f for f in result["flags"])
    assert CONSIDER_CONCENTRATION in result["considerations"]


def test_hhi_and_effective_holdings_for_equal_weights():
    result = analyze(holdings(*EVEN_FIVE), dict.fromkeys("ABCDE", 10.0))
    assert result["hhi"] == pytest.approx(0.2)
    assert result["effective_holdings"] == 5
    assert result["flags"] == [] and result["considerations"] == []


def test_single_holding_is_fully_concentrated():
    result = analyze(holdings({"symbol": "TSLA", "shares": 4}), {"TSLA": 250.0})
    assert result["hhi"] == 1.0
    assert result["effective_holdings"] == 1.0
    assert "Only 1 priced holding —" in " ".join(result["flags"])
    assert result["considerations"] == [
        CONSIDER_CONCENTRATION,
        CONSIDER_FEW_HOLDINGS,
        CONSIDER_ONE_ASSET_CLASS,
    ]


def test_asset_class_breakdown_sums_to_one():
    result = analyze(holdings(*EVEN_FIVE), dict.fromkeys("ABCDE", 10.0))
    assert result["by_asset_class"] == {"equity": 0.6, "bond": 0.2, "cash": 0.2}


def test_unpriced_holdings_are_reported():
    df = holdings({"symbol": "AAPL", "shares": 1}, {"symbol": "ZZZZ", "shares": 1})
    assert analyze(df, {"AAPL": 100.0})["unpriced"] == ["ZZZZ"]


def test_no_priced_holdings_does_not_crash():
    result = analyze(holdings({"symbol": "ZZZZ", "shares": 1}), {})
    assert result["total_value"] == 0
    assert result["holdings"] == []
    assert result["effective_holdings"] == 0
    assert result["unpriced"] == ["ZZZZ"]


def test_zero_share_portfolio_does_not_divide_by_zero():
    result = analyze(holdings({"symbol": "AAPL", "shares": 0}), {"AAPL": 100.0})
    assert result["total_value"] == 0
    assert result["effective_holdings"] == 0


# --- risk_metrics ----------------------------------------------------------------------


def test_flat_prices_have_no_volatility_or_drawdown():
    risk = risk_metrics({"A": 1.0}, {"A": bars(10, 10, 10, 10)})
    assert risk["volatility"] == 0
    assert risk["max_drawdown"] == 0
    assert risk["period_return"] == 0
    assert risk["risk_level"] == "low"


def test_drawdown_and_return_follow_the_price_path():
    risk = risk_metrics({"A": 1.0}, {"A": bars(100, 120, 90, 110)})
    assert risk["max_drawdown"] == pytest.approx(-0.25)  # 120 → 90
    assert risk["period_return"] == pytest.approx(0.10)
    assert risk["growth_of_100"][0]["value"] == 100
    assert risk["growth_of_100"][-1]["value"] == pytest.approx(110)


def test_portfolio_return_is_weighted():
    risk = risk_metrics(
        {"UP": 0.75, "FLAT": 0.25}, {"UP": bars(100, 100, 120), "FLAT": bars(50, 50, 50)}
    )
    assert risk["period_return"] == pytest.approx(0.15)


def test_holding_without_history_is_reweighted_and_listed():
    risk = risk_metrics({"A": 0.5, "B": 0.5}, {"A": bars(100, 100, 110)})
    assert risk["missing_history"] == ["B"]
    assert risk["period_return"] == pytest.approx(0.10)  # A alone, at full weight


def test_only_dates_shared_by_all_holdings_are_used():
    a = bars(100, 101, 102, 103)
    b = bars(10, 11, 12, 13)[1:]  # starts a day later
    risk = risk_metrics({"A": 0.5, "B": 0.5}, {"A": a, "B": b})
    assert risk["start_date"] == "2026-01-02"
    assert risk["trading_days"] == 2


def test_too_little_history_returns_empty():
    assert risk_metrics({"A": 1.0}, {"A": bars(100, 101)}) == {}
    assert risk_metrics({"A": 1.0}, {}) == {}


def test_high_volatility_is_labelled_high():
    risk = risk_metrics({"A": 1.0}, {"A": bars(100, 110, 95, 115, 90, 120)})
    assert risk["risk_level"] == "high"
