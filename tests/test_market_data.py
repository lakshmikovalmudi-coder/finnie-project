"""Quotes and price history: live → mock fallback order, without network access."""

import dataclasses
import json

import pytest
import requests

from src.config import settings
from src.tools import market_data

MOCK_QUOTES = json.loads((settings.mock_dir / "quotes.json").read_text())


@pytest.fixture
def live_settings(monkeypatch):
    """Pretend a key is configured so the live path is attempted first."""
    monkeypatch.setattr(
        market_data,
        "settings",
        dataclasses.replace(settings, use_mock_data=False, alpha_vantage_api_key="test-key"),
    )


@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setattr(market_data, "settings", dataclasses.replace(settings, use_mock_data=True))


def test_mock_history_is_deterministic_and_ends_at_mock_price(mock_settings):
    first = market_data.get_history("vti")
    assert first == market_data.get_history("VTI")
    assert first["source"] == "mock"
    assert len(first["bars"]) == market_data.MOCK_HISTORY_DAYS
    assert first["bars"][-1]["close"] == pytest.approx(MOCK_QUOTES["VTI"]["price"])
    dates = [b["date"] for b in first["bars"]]
    assert dates == sorted(dates)


def test_unknown_symbol_has_no_history(mock_settings):
    assert market_data.get_history("ZZZZ") is None
    assert market_data.get_histories(["VTI", "ZZZZ"]).keys() == {"VTI"}


def test_live_history_is_used_when_available(live_settings, monkeypatch):
    live = [{"date": "2026-10-08", "close": 1.0}, {"date": "2026-10-09", "close": 2.0}]
    monkeypatch.setattr(market_data, "_fetch_history", lambda symbol: live)
    assert market_data.get_history("SPY") == {"symbol": "SPY", "source": "live", "bars": live}


@pytest.mark.parametrize(
    "failure",
    [
        lambda symbol: None,  # quota hit: Alpha Vantage answers 200 with a "Note"
        lambda symbol: (_ for _ in ()).throw(requests.Timeout("slow")),
    ],
)
def test_history_falls_back_to_mock_when_live_fails(live_settings, monkeypatch, failure):
    monkeypatch.setattr(market_data, "_fetch_history", failure)
    assert market_data.get_history("SPY")["source"] == "mock"


def test_quote_falls_back_to_mock_on_http_error(live_settings, monkeypatch):
    def boom(symbol):
        raise requests.HTTPError("503")

    monkeypatch.setattr(market_data, "_fetch_quote", boom)
    quote = market_data.get_quote("aapl")
    assert quote["source"] == "mock"
    assert quote["price"] == MOCK_QUOTES["AAPL"]["price"]
