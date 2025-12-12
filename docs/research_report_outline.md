# Final technical research report outline

1. **Abstract and research questions:** where do retrieval and generation quality fail in Project 6?
2. **System description:** ingestion, corpus, chunking, embeddings, DB, retrieval, reranking, prompt and generator versions.
3. **Dataset and ground truth:** sampling, corpus citations, review process, question categories, held-out set.
4. **Metrics and evaluators:** faithfulness, answer relevance, contextual precision, context recall, deterministic rank metrics, judge limits.
5. **Experiment design:** controlled chunk-size/overlap matrix; retriever/top-k/reranker variants; index isolation and run identity.
6. **Execution and evidence:** real per-question traces, timestamps, failed attempts, provider usage and missing-data accounting.
7. **Results:** separate tables/figures for retrieval and generation scores, distributions, category breakdowns and latency/usage.
8. **Statistical analysis:** matched-question deltas, confidence intervals when justified, judge variability, multiple comparisons.
9. **Failure taxonomy and case studies:** omitted evidence, poor ranking, unsupported claims, wrong intent, abstention and metric failures.
10. **Limitations and threats to validity:** judge bias, annotation incompleteness, corpus shift, provider and prompt variability.
11. **Recommendations:** quality/latency/cost trade-offs and conditional deployment choices; never pick a winner without measured evidence.
12. **Reproducibility appendix:** code revision, dataset checksum, configuration snapshots, indexes, provider versions and export paths.

> Populate sections 6–9 only after a real Project 6 run with source-validated annotations; do not insert hypothetical scores.
