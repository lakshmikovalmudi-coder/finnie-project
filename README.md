# Finnie — AI Finance Assistant

A multi-agent conversational tutor that helps beginners learn to invest.
A LangGraph router sends each question to one of six specialist agents, and
answers are grounded in a curated knowledge base (RAG) and live market data.

> Finnie provides **financial education, not advice.**

## Architecture

```
User → Router → { Finance Q&A | Portfolio | Market | Goals | News | Tax } → Compliance → Answer
```

See [`agent.md`](agent.md) for roles, state, and coding standards.

| Agent | Grounded in |
|-------|-------------|
| Finance Q&A | RAG over `data/knowledge_base/` |
| Portfolio Analysis | Uploaded CSV + market prices + pandas |
| Market Analysis | Alpha Vantage (30-min cache, mock fallback) |
| Goal Planning | Deterministic savings math |
| News Synthesizer | Tavily or SerpApi (cached, mock fallback) |
| Tax Education | RAG over `data/knowledge_base/` |

**Stack:** LangGraph · LangChain · OpenAI · FAISS · sentence-transformers (all-MiniLM-L6-v2) · Streamlit

## Setup

Requires **Python 3.12**.

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your API keys
python -m src.rag.ingest      # build the FAISS index
streamlit run src/ui/app.py
```

No Alpha Vantage / news key yet? Set `USE_MOCK_DATA=true` — the app runs on sample data.

## Project layout

```
src/
  config.py  llm.py  state.py  graph.py
  agents/   router + 6 specialists + compliance
  tools/    market_data, news_search, portfolio_math, goal_math, cache
  rag/      ingest (build index), retriever
  ui/       app.py + tabs/ (chat, portfolio, markets, goals, knowledge)
data/
  knowledge_base/   finance articles (target 50–100)
  mock/             fallback quotes & news
  sample_portfolio.csv
tests/
```

## Tests

```bash
pytest
ruff check .
```
