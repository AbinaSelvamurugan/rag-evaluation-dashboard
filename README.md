# RAG Evaluation Dashboard

Evaluation and optimization of retrieval-augmented generation systems.

This is a Streamlit application for evaluating an existing RAG pipeline. It captures retrieval and generation traces through a bridge, scores them with DeepEval judge metrics when a judge provider is configured, computes deterministic retrieval metrics from dataset annotations, stores versioned experiments in SQLite, and compares isolated indexes.

## Quick start

Recommended Python 3.13 (also supports >=3.11,<3.14). Run commands **from the repository root**:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -e '.[all]'
cp .env.example .env                      # Windows: copy .env.example .env
# Set environment variables in your shell or load .env yourself (not auto-loaded).
python -m pytest -q
ragbench validate-dataset data/smoke_fixture.jsonl
ragbench import-dataset data/smoke_fixture.jsonl
streamlit run app/streamlit_app.py
```

Streamlit opens at `http://localhost:8501`. When no runs exist yet, the Overview shows guidance rather than metrics. For an offline smoke test, copy the version printed by `import-dataset`, replace `REPLACE_WITH_IMPORTED_VERSION` in `configs/fixture.json`, then run:

```bash
ragbench run --config configs/fixture.json             # preview only
ragbench run --config configs/fixture.json --execute
ragbench history
ragbench matrix --config configs/fixture.json --sizes 64,128 --overlaps 0,0.1 --retrievers keyword --execute
```

## Project 6 integration

1. Inspect Project 6's ingestion, document extraction, chunking units, embedding model, vector store, retrievers, generation endpoint, source IDs, citations, usage, and latency, and check whether its index can be rebuilt with isolation.
2. Add the [evaluation bridge API](docs/integration.md) inside or alongside Project 6, delegating to its own ingestion, indexing, retrieval, and generation services.
3. Configure `PROJECT6_EVAL_URL` and, if needed, `PROJECT6_EVAL_TOKEN` as environment variables. This application does not load `.env` itself, although DeepEval can auto-load `.env` at import unless `DEEPEVAL_DISABLE_DOTENV=1`.
4. Import a source-validated dataset with the expected answers and relevant original document IDs. For chunk sweeps, document IDs should stay stable; chunk IDs change when the chunker changes.
5. Copy `configs/project6.json`, set the dataset version and settings advertised by your bridge, and export `OPENAI_API_KEY` or other judge provider credentials for your selected DeepEval model.
6. Run `ragbench run --config your-config.json --execute`, inspect the traces, then run matrix experiments. Use the `holdout` split once after configuration selection.

The bridge rejects mismatched index fingerprints, unsupported retrievers, unadvertised chunk units or strategies, unexpected passage ranks, and mismatched effective generation or retrieval configuration. See [integration](docs/integration.md).

## Commands

```bash
ragbench validate-dataset path/to/questions.jsonl
ragbench import-dataset path/to/questions.csv
ragbench run --config configs/project6.json
ragbench run --config configs/project6.json --execute
ragbench matrix --config configs/project6.json --sizes 256,512,768 --overlaps 0,0.1,0.2 --retrievers dense,keyword
ragbench matrix --config configs/project6.json --sizes 256,512,768 --overlaps 0,0.1 --retrievers dense,keyword --execute
ragbench compare BASELINE_RUN_ID CANDIDATE_RUN_ID
ragbench export RUN_ID --format markdown --output results/report-RUN_ID.md
ragbench export RUN_ID --format json --output results/run-RUN_ID.json
ragbench resume RUN_ID
ragbench history
python -m pytest -q
ruff check src tests app
```

Preview mode performs no API calls. Every matrix configuration creates a separate run, and each reindex request must return an index isolated by fingerprint. The CLI and the Streamlit UI both run matrix configurations sequentially, with bounded per-run question concurrency. The UI also enforces a 5,000-request matrix safety limit. Size experiment matrices and budgets conservatively. Prices are not estimated; judge metric evaluation can involve multiple API requests.

## Views

| Page | Purpose |
|---|---|
| Overview | completed runs, score means with sample counts, clear missing states |
| Evaluation Dataset | import, version, check missing labels, annotate via new version, export |
| Run Evaluation | configurable matrix, request preview, launch, SQLite-based live progress, cancel |
| Chunking Experiments | size, overlap, index count, latency and quality comparisons |
| Retriever Experiments | retrieval and generation differences across recorded strategies |
| Detailed Results | question, reference, passages/ranks/scores, answer, judge reasons/errors, timing, usage |
| Experiment History | immutable runs and paired comparisons with uncertainty estimates |
| Reports & Export | CSV, JSON trace bundle, Markdown reports and definitions |

## Method and data model

`Question` fields: `question_id`, `question`, `expected_answer`, `relevant_document_ids`, `relevant_chunk_ids`, `reference_context`, `category`, `difficulty`, `sufficient_evidence`, `expected_abstention`, `annotation_status`, `split`, `dataset_version`. `dataset_version` is derived from a SHA-256 hash of the canonical question records, and annotation edits never overwrite it. Both JSONL and CSV are bounded to 5 MiB and 10,000 records.

DeepEval metrics: `FaithfulnessMetric` (answer vs retrieved contexts), `AnswerRelevancyMetric` (question vs answer), `ContextualPrecisionMetric` (ranked contexts using the expected answer), optional `ContextualRecallMetric` and `ContextualRelevancyMetric`. The `context_precision` name denotes DeepEval Contextual Precision, not deterministic `precision_at_k`. Deterministic metrics use the annotated relevant document and chunk IDs; incomplete labeling yields `unavailable` rather than zero. Means exclude metric errors and report valid counts. See [metrics](docs/metrics.md) and [methodology](docs/evaluation_methodology.md).

Storage: `results/ragbench.sqlite3` (created at first use), tables `datasets`, `runs`, `question_results`, `metric_cache`. SQLite stores full question traces as JSON and runs as immutable snapshots, one record per `(run_id, question_id)`. The `RAGBENCH_HOME` setting redirects the store. Dataset versions do not mutate. Reports are generated from stored traces.

## Security and data handling

Retrieved texts, questions, answers, identifiers, and reasons are stored in plaintext in the local SQLite database. DeepEval judges send their inputs to the configured external provider, so inference is not local or private. To expose the dashboard remotely, configure `RAGBENCH_ADMIN_TOKEN`, HTTPS, network-level authentication or reverse-proxy restrictions, and restricted file permissions. Without an admin token the app is intended for localhost-only use, since Streamlit does not provide a complete authentication layer. The Docker Compose deployment binds to `127.0.0.1` by default. Retention and deletion are manual; manage backups in protected environments. See [security](docs/security.md).

## Documentation

- [Architecture](docs/architecture.md)
- [Project 6 integration contract](docs/integration.md)
- [Metric definitions](docs/metrics.md)
- [Experimental design](docs/experiments.md)
- [Evaluation methodology](docs/evaluation_methodology.md)
- [Security and privacy](docs/security.md)
- [Deployment](docs/deployment.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Research report outline](docs/research_report_outline.md)

## Scope limits

Live evaluation requires model and API credentials, source-grounded corpus annotations, and the Project 6 bridge described in the integration guide. Feature gates fail explicitly when their prerequisites are missing rather than returning placeholder results. HTTP retry handling applies to Project 6 requests; hard cancellation of an in-progress external judge call is not supported. A cancelled run drains in-flight requests, saves completed records, and stops submitting new work.
