"""Knowledge tab: browse and search the curated articles that ground RAG answers."""

import streamlit as st

from src.config import settings


def render() -> None:
    st.subheader("Knowledge Base")
    articles = sorted(
        p for p in settings.knowledge_dir.glob("*.md") if p.name.lower() != "readme.md"
    )
    st.caption(f"{len(articles)} articles (target: 50–100)")

    query = st.text_input("Search the knowledge base")
    if query:
        from src.rag.retriever import search  # lazy: loads the embedding model

        with st.spinner("Searching…"):
            for hit in search(query):
                with st.expander(f"{hit['title']}  ·  score {hit['score']:.2f}"):
                    st.markdown(hit["text"])
        return

    for path in articles:
        text = path.read_text()
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        with st.expander(title):
            st.markdown(text)
