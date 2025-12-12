# Controlled experiment design

## Chunk size and overlap

Select sizes in a documented **shared unit** (`characters` or `tokens`, as Project 6 advertises). Matrix expansion is Cartesian product of chunk sizes × overlap *fractions* × retrievers. Each configuration creates a new run and calls `POST /eval/indexes` with its exact chunking parameters. The backend must return isolated, immutable `index_id`, `fingerprint`, matching effective parameters, and optionally chunk count/index bytes/indexing time. The caller rejects mismatched effective index settings and mismatched run trace/index fingerprints. `max_questions` and matrix size (≤100 configs) bound experiments. The Streamlit UI also refuses matrices greater than 5,000 total RAG requests and runs matrix configurations sequentially. Use the **same corpus version**, question split, embedding model, retriever, top-k, prompt, generation model and judge when isolating the effect of chunking.

## Retriever and top-k comparisons

Compare *advertised* Project 6 retrievers only, against the same corpus/index settings. The caller rejects unadvertised retrievers/rerankers. Report ranking metrics alongside answer faithfulness and relevance, latency, context size and actual token usage if exposed. `top_k` can be changed via full config snapshot; it is not silently adjusted. For MMR/reranking, ensure the Project 6 backend reports the final ordering fed to generation and the reranker parameters in trace metadata.

## Interpreting results

Start with a development split; use a separately held-out versioned split once to estimate transfer after tuning. Paired comparisons match identical question IDs, dataset versions and split. For ≥5 valid matched pairs, 2,000 fixed-seed bootstrap resamples supply an exploratory 95% confidence interval for mean **candidate minus baseline** delta. No interval or statistical superiority claim is made for fewer than five pairs. Across multiple configurations and metrics, account for multiple testing and evaluator variation before declaring a winner. Review failure traces, context omission, hallucinated answers, category breakdowns, and quality/latency/usage trade-offs. Unknown prices are **not** reported.

## Reproducibility

Save the dataset checksum, configuration JSON, index fingerprint, package versions, timestamps and full passages. Use distinct runs for parameter changes; never reuse an incompatible index. Cache keys include exact question/answer/passages and evaluator configuration/version. An interrupted job can be resumed if index identity is stable; previously saved results are not duplicated. Beware resumed fixtures have no independent generator and are **not** real quality experiments.
