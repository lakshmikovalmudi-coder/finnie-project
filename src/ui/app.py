"""Streamlit entry point. Run from the project root: streamlit run src/ui/app.py"""

import logging
import sys
from pathlib import Path

# Streamlit puts this file's folder on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src.agents.compliance import DISCLAIMER  # noqa: E402
from src.config import settings  # noqa: E402
from src.ui.tabs import chat, goals, knowledge, markets, portfolio  # noqa: E402

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Finnie — AI Finance Tutor", page_icon="💸", layout="wide")


def sidebar() -> None:
    with st.sidebar:
        st.title("💸 Finnie")
        st.caption("Your AI finance tutor for getting started with investing.")
        st.subheader("Status")
        st.write("LLM:", "✅" if settings.openai_api_key else "❌ set OPENAI_API_KEY")
        st.write(
            "Market data:",
            "🧪 mock"
            if settings.use_mock_data or not settings.alpha_vantage_api_key
            else "✅ live (cached 30 min)",
        )
        news_key = (
            settings.tavily_api_key
            if settings.news_provider == "tavily"
            else settings.serpapi_api_key
        )
        st.write(
            "News:",
            "🧪 mock" if settings.use_mock_data or not news_key else f"✅ {settings.news_provider}",
        )
        st.divider()
        st.caption(DISCLAIMER)


def main() -> None:
    sidebar()
    tabs = st.tabs(["💬 Chat", "📊 Portfolio", "📈 Markets", "🎯 Goals", "📚 Knowledge"])
    with tabs[0]:
        chat.render()
    with tabs[1]:
        portfolio.render()
    with tabs[2]:
        markets.render()
    with tabs[3]:
        goals.render()
    with tabs[4]:
        knowledge.render()


main()
