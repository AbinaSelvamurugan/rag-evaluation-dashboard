# Integrating with the actual Project 6 services

**Status: versioned protocol defined; the endpoints are implemented by the Project 6 backend.** The following is an explicit versioned interface **to be added to the existing Project 6 backend**, delegating to its own ingestion, indexing, retrieval, and generation services.

## Discover the existing system before connecting

Inspect (and record values under your project's documentation) the ingestion/extraction pipeline, token-vs-character chunker, chunk boundaries, embedding model/version, vector DB/index namespace, cosine/dot-product score conventions, retriever/reranker settings, prompt-template versions, generation provider/model, final citations, response usage, errors, and current API permissions. Identify how retrieval can be run with precise `top_k` and how you can capture the passages *actually sent to the generator*.

## Required HTTP endpoints (ragbench-v1)

Use HTTPS outside localhost. Configure `PROJECT6_EVAL_URL`; optional `PROJECT6_EVAL_TOKEN` is sent in an `Authorization: Bearer ...` header. Access to the index endpoint must be restricted to administrators in the **Project 6 backend** too; merely hiding it in the dashboard is inadequate.

### `GET /eval/capabilities`

```json
{
  "protocol": "ragbench-v1",
  "retrievers": ["dense", "keyword"],
  "chunk_units": ["characters"],
  "chunk_strategies": ["recursive"],
  "supports_reindex": true,
  "rerankers": []
}
```

List only verified working methods. If Project 6 cannot isolate corpus indexes and rebuild them, report `supports_reindex:false` and note chunk sweeps will be blocked. This example is a **schema illustration**, not a discovered capability list.

### `POST /eval/indexes`

Request body: full `ExperimentConfig` JSON, including `chunk_strategy`, `chunk_size`, `chunk_overlap`, `chunk_unit`, `embedding_model`, and experiment metadata.

Response:

```json
{
  "index_id": "opaque-project6-collection-id",
  "fingerprint": "sha256-or-project6-reproducible-index-fingerprint",
  "effective_config": {
    "chunk_strategy": "recursive",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "chunk_unit": "characters",
    "embedding_model": "actual-embedding-model"
  },
  "chunk_count": 1234,
  "index_bytes": null,
  "indexing_ms": 4200
}
```

The bridge should use Project 6's **real ingestion and chunker**, rebuild separate collections for changed chunk settings and corpus version, and return an immutable index identifier. The fingerprint should commit to corpus checksum, extractor/chunker versions, chunk settings, embedding model, and index collection. The caller verifies the effective configuration. `null` is required when measurements are unavailable.

### `POST /eval/trace`

Request:

```json
{
  "question": "question from versioned dataset",
  "index_id": "opaque-project6-collection-id",
  "config": {"...": "complete ExperimentConfig as above"}
}
```

Response schema (values shown are illustrative **not measured**):

```json
{
  "question": "question from versioned dataset",
  "answer": "answer returned by Project 6",
  "passages": [{
    "document_id": "stable-source-document-id",
    "chunk_id": "actual-chunk-id",
    "rank": 1,
    "text": "exact text provided to the generator",
    "score": null,
    "source": "document.pdf#page=2",
    "metadata": {"page": 2}
  }],
  "effective_config": {
    "retriever": "dense",
    "top_k": 5,
    "generation_model": "actual-model-name",
    "prompt_version": "real-prompt-version",
    "reranker": null,
    "similarity_threshold": null,
    "context_budget": 4096
  },
  "index_id": "opaque-project6-collection-id",
  "index_fingerprint": "exact-fingerprint-returned-by-indexes",
  "retrieval_ms": null,
  "generation_ms": null,
  "total_ms": null,
  "input_tokens": null,
  "output_tokens": null,
  "citations": [],
  "trace_metadata": {"provider_request_id": "redacted-or-safe-id"}
}
```

**Requirements:** Preserve complete ranked passages and stable source/chunk IDs as actually retrieved; ranks must start at 1 without gaps; emitted context must match generation input. Do not reconstruct passages using another retrieval path. Report measured latencies from monotonic clocks and model token usage only when the provider exposes actual values. Avoid sending chain-of-thought or private provider headers. Return the actual generation response, never a template or mocked completion. Echo/verify real effective model/prompt, top-k and retriever fields. Failed calls should return non-2xx status and safe diagnostic messages. Reject unsupported configurations rather than silently using the Project 6 default.

### Integration sequence

1. Implement endpoints in the existing backend by invoking actual ingestion, indexing, retriever and answer-generation services.
2. Add a backend-only authenticated evaluator account with the necessary index privileges.
3. Manually verify a single trace's answer, passage order and source IDs against a normal Project 6 call.
4. Run an integration test with an answerable question and known relevant source IDs; verify index fingerprint and request configuration.
5. Reindex two distinct chunk settings and confirm differing isolated collection IDs/fingerprints when boundaries differ.
6. Re-run tests with an intentionally invalid method and fingerprint; the bridge must fail explicitly.
7. Configure environment, import a verified corpus-grounded dataset, run a small real evaluation, then controlled matrix experiments.

If Project 6 is only a Python package, create these bridge routes around the real service functions. Do not copy or replace the functioning pipeline. For customized auth or model providers, adapt `Project6HTTPAdapter` and add integration tests.

**Why the boundary matters:** These Project 6-specific services supply the retrieval traces, corpus reindexes, and generated answers that evaluation depends on. The fixture adapter exercises the dashboard but does not exercise the real RAG pipeline.
