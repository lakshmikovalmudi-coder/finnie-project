# agent.md — Finnie Multi-Agent Architecture

Finnie is a conversational finance **tutor** for beginner investors. It explains
concepts, analyzes holdings, and runs the numbers — it **educates, it does not
advise**. Every response that touches money must make that distinction clear.

Source of truth for scope: `Capstone Notes.pdf` → *Project 1 — Finnie*.

---

## 1. System Overview

```
                        ┌──────────────────────────┐
  Streamlit UI  ──────▶ │   LangGraph StateGraph    │
  (Chat / Portfolio /   │                           │
   Markets / Goals /    │   ┌──────────────────┐    │
   Knowledge tabs)      │   │  Router (classify)│    │
                        │   └────────┬─────────┘    │
                        │            │ route        │
     ┌──────────┬───────┼──────┬─────┴────┬─────────┼──────────┐
     ▼          ▼       ▼      ▼          ▼         ▼          │
 finance_qa  portfolio market  goals     news      tax         │
  (RAG)     (pandas)  (Alpha  (math)   (Tavily/   (RAG)        │
                       Vantage)         SerpApi)               │
     └──────────┴───────┴──────┴──────────┴─────────┘          │
                        │                                      │
                        ▼                                      │
                  ┌───────────┐                                │
                  │ compliance│  ← appends disclaimer, blocks  │
                  └─────┬─────┘    personalized advice         │
                        ▼                                      │
                       END                                     │
                        └──────────────────────────────────────┘
```

Flow for every user turn:

1. **Router** classifies the query into exactly one route (structured LLM output,
   with a keyword fallback if the LLM call fails).
2. The graph dispatches to one **specialist agent** node.
3. The specialist calls its **tools** (RAG, market data, news, math) and drafts
   an answer grounded in tool output.
4. The **compliance** node appends the education-not-advice disclaimer and
   records sources.
5. The final state is returned to the UI.

Graph wiring lives in `src/graph.py`. Shared state is `FinnieState` in
`src/state.py`.

---

## 2. Agent Roles

| Node | File | Responsibility | Tools | Never does |
|------|------|----------------|-------|-----------|
| **Router** | `src/agents/router.py` | Classify query → one of `finance_qa`, `portfolio`, `market`, `goals`, `news`, `tax`. Returns route + confidence. | LLM (structured output) | Answer the question itself |
| **Finance Q&A** | `src/agents/finance_qa.py` | Explain concepts (compound interest, ETFs vs mutual funds, diversification). Default route when unsure. | RAG retriever | Recommend specific securities |
| **Portfolio Analysis** | `src/agents/portfolio.py` | Analyze uploaded holdings: allocation, concentration, diversification and risk flags. | `portfolio_math`, `market_data` | Tell the user to buy/sell |
| **Market Analysis** | `src/agents/market.py` | Pull live quotes and explain what the numbers mean (P/E, % change, volume). | `market_data` (cached Alpha Vantage) | Predict prices |
| **Goal Planning** | `src/agents/goals.py` | Compute monthly savings needed to hit a target; show assumptions. | `goal_math` (deterministic) | Let the LLM do arithmetic |
| **News Synthesizer** | `src/agents/news.py` | Summarize current financial news **with source links**. | `news_search` (Tavily / SerpApi, cached) | Summarize without citing sources |
| **Tax Education** | `src/agents/tax.py` | Explain 401(k) vs IRA vs Roth, contribution concepts. | RAG retriever | Give tax advice for the user's situation |
| **Compliance** | `src/agents/compliance.py` | Append disclaimer, attach sources, soften advice-like phrasing. | — | Be skipped |

> The four core roles called out for this workspace are **Router, Portfolio,
> Market, Goals**. The notes specify six specialists, so Finance Q&A, News and
> Tax are scaffolded too — they're cheap to stub and are part of the rubric.

### Routing rules
- Low router confidence (< `ROUTER_MIN_CONFIDENCE`) → `finance_qa`.
- Queries mentioning tickers/prices → `market`; "my holdings / my portfolio" →
  `portfolio`; "save / retire / target / by age" → `goals`; "news / today /
  headlines" → `news`; "401k / IRA / Roth / tax" → `tax`.
- Requests for personal advice ("should I buy X?") still route normally, but the
  specialist reframes into education and compliance adds the disclaimer.

---

## 3. Shared State Contract

`FinnieState` (TypedDict, `src/state.py`):

| Key | Type | Set by |
|-----|------|--------|
| `messages` | `list[BaseMessage]` (add_messages reducer) | UI / agents |
| `query` | `str` | UI |
| `route` | `Route` literal | Router |
| `route_confidence` | `float` | Router |
| `portfolio` | `list[dict] \| None` | UI (Portfolio tab upload) |
| `answer` | `str` | Specialist |
| `sources` | `list[str]` | Specialist / RAG / News |
| `data` | `dict` | Specialist (structured payload for UI charts) |
| `error` | `str \| None` | Any node |

Agents **return partial dicts** (LangGraph merges them). They never mutate
state in place.

---

## 4. Coding Standards

**Language & runtime**
- Python **3.12** (the `venv/` is 3.12; avoid 3.13).
- **Async throughout**: agent nodes are `async def`; blocking I/O (`requests`)
  is wrapped with `asyncio.to_thread`. The UI calls `src/aio.py::run_sync(...)` at
  the edge — never `asyncio.run`, which breaks the cached OpenAI client after one call.

**Structure**
- One agent per file in `src/agents/`, each exposing `async def run(state) -> dict`.
- Tools in `src/tools/` are **pure-ish functions with no LLM calls**; they are
  unit-testable without network or API keys.
- Prompts live as module-level constants (`SYSTEM_PROMPT`) at the top of each
  agent file.
- Config is read **only** via `src/config.py` (`settings`). No `os.getenv`
  scattered through the code.

**LLM usage**
- Get models from `src/llm.py::get_llm()` — never instantiate `ChatOpenAI` directly.
- Use structured output (Pydantic) for anything the code branches on (routing).
- **Numbers come from Python, not the LLM.** Goal math, portfolio weights, and
  returns are computed deterministically; the LLM only explains them.
- Ground answers in tool/RAG output. If retrieval returns nothing, say so.

**External APIs**
- Every external call goes through `src/tools/cache.py` (`@ttl_cache`).
- Every external tool has a **mock fallback** (`data/mock/`) used when the key
  is missing, the quota is hit, or `USE_MOCK_DATA=true`. The demo must work
  offline.
- Never log or display API keys. `.env` is git-ignored and must not be
  submitted.

**Compliance**
- Every user-facing answer passes through the compliance node.
- Banned phrasings in agent output: "you should buy", "I recommend investing in",
  "guaranteed return". Reframe as "here's how to think about it".

**Style & quality**
- Type hints on all public functions; Pydantic v2 for structured data.
- `ruff check . && ruff format .` before committing.
- `pytest` for tools (math, cache, fallbacks) and router keyword fallback.
- Docstrings: one line stating *what* and *why*, not *how*.
- Logging via `logging.getLogger(__name__)`; no `print` in `src/`.

---

## 5. Adding a New Agent (checklist)

1. Add the route name to `Route` in `src/state.py`.
2. Create `src/agents/<name>.py` with `SYSTEM_PROMPT` and `async def run(state)`.
3. Register it in `AGENT_NODES` in `src/graph.py`.
4. Add a keyword rule in `src/agents/router.py::KEYWORD_ROUTES`.
5. Add a test in `tests/`.
