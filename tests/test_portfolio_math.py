import pandas as pd

from src.tools.portfolio_math import analyze, load_holdings


def test_concentration_flag_and_weights():
    df = load_holdings(
        pd.DataFrame([{"Symbol": "aapl", "Shares": 90}, {"Symbol": "bnd", "Shares": 10}])
    )
    result = analyze(df, {"AAPL": 10.0, "BND": 10.0})
    assert result["total_value"] == 1_000
    assert any("AAPL" in f for f in result["flags"])


def test_unpriced_holdings_are_reported():
    df = load_holdings(
        pd.DataFrame([{"symbol": "AAPL", "shares": 1}, {"symbol": "ZZZZ", "shares": 1}])
    )
    assert analyze(df, {"AAPL": 100.0})["unpriced"] == ["ZZZZ"]
