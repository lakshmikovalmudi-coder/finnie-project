# Knowledge Base

Target: **50–100 short articles** (course requirement) that ground RAG answers.

- One topic per file, Markdown, 200–600 words.
- First line is `# Title`; add a `Source:` line linking where facts were verified
  (Investopedia, Bogleheads wiki, SEC Investor.gov).
- Suggested folders/prefixes: `basics-`, `investing-`, `retirement-`, `tax-`, `risk-`.
- Rebuild the index after adding articles: `python -m src.rag.ingest`

This README is skipped during ingestion.
