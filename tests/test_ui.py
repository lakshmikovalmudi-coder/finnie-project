"""Streamlit smoke tests: every tab renders on mock data without raising."""

import dataclasses

import pytest
from streamlit.testing.v1 import AppTest

from src.config import ROOT_DIR
from src.tools import market_data, news_search

APP = str(ROOT_DIR / "src" / "ui" / "app.py")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for module in (market_data, news_search):
        monkeypatch.setattr(
            module, "settings", dataclasses.replace(module.settings, use_mock_data=True)
        )


def run_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception, app.exception
    return app


def test_all_tabs_render():
    app = run_app()
    assert [t.label for t in app.tabs] == [
        "💬 Chat",
        "📊 Portfolio",
        "📈 Markets",
        "🎯 Goals",
        "📚 Knowledge",
    ]


def test_markets_tab_shows_quotes_and_history_stats():
    app = run_app()
    labels = [m.label for m in app.metric]
    assert "SPY (sample)" in labels
    assert "Volatility (annual)" in labels
    assert any(label.startswith("Return since") for label in labels)


def test_sample_portfolio_shows_risk_metrics():
    app = run_app()
    next(b for b in app.button if b.label == "Use sample portfolio").click().run()
    assert not app.exception, app.exception
    labels = [m.label for m in app.metric]
    assert {"Total value", "Effective holdings", "Max drawdown"} <= set(labels)
    assert any("of the portfolio" in w.value for w in app.warning)  # concentration flag
    assert app.session_state["portfolio_rows"]
