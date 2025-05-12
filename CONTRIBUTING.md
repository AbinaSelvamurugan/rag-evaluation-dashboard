# Contributing

Protect existing Project 6 code; prefer adding a strict adapter instead of forking the RAG implementation. Record software versions, document source-grounded relevance judgments, and keep holdout questions reserved. Never add fabricated metrics, external secrets, or real private document text to tests. Include unit tests and clear error states for new metrics and bridge fields. Run `python -m pytest -q` and, if installed, `ruff check src tests app` before submitting changes. Do not merge integration claims without passing tests.
