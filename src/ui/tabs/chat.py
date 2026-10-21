"""Chat tab: one conversation routed through the LangGraph multi-agent graph."""

from uuid import uuid4

import streamlit as st

from src.aio import run_sync
from src.graph import run_query

ROUTE_LABELS = {
    "finance_qa": "📘 Finance Q&A",
    "portfolio": "📊 Portfolio",
    "market": "📈 Market",
    "goals": "🎯 Goals",
    "news": "📰 News",
    "tax": "🧾 Tax Education",
}

EXAMPLES = [
    "What is compound interest?",
    "ETFs vs mutual funds — what's the difference?",
    "How is AAPL doing today?",
    "How much should I save monthly to reach $50,000 in 5 years?",
    "Explain 401(k) vs Roth IRA",
    "What's in the financial news today?",
]


def _show(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg.get("route"):
            st.caption(f"Answered by {ROUTE_LABELS.get(msg['route'], msg['route'])}")
        if msg.get("understood_as"):
            st.caption(f"Follow-up understood as: *{msg['understood_as']}*")
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")


def render() -> None:
    history: list[dict] = st.session_state.setdefault("chat_history", [])
    # The graph remembers the conversation per thread_id; the UI list is only for display.
    thread_id: str = st.session_state.setdefault("thread_id", uuid4().hex)

    if history and st.button("🧹 New conversation"):
        st.session_state["chat_history"] = []
        st.session_state["thread_id"] = uuid4().hex
        st.rerun()

    if not history:
        st.markdown("**Try asking:**")
        cols = st.columns(3)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 3].button(example, key=f"ex_{i}", width="stretch"):
                st.session_state["pending_query"] = example
                st.rerun()

    for msg in history:
        _show(msg)

    query = st.chat_input("Ask Finnie about investing, saving, markets, or taxes…")
    query = query or st.session_state.pop("pending_query", None)
    if not query:
        return

    history.append({"role": "user", "content": query})
    _show(history[-1])
    with st.spinner("Thinking…"):
        result = run_sync(run_query(query, st.session_state.get("portfolio_rows"), thread_id))
    rewritten = result.get("query", query)
    history.append(
        {
            "role": "assistant",
            "content": result.get("answer", ""),
            "route": result.get("route"),
            "sources": result.get("sources", []),
            "understood_as": rewritten if rewritten.strip() != query.strip() else None,
        }
    )
    _show(history[-1])
