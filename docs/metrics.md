# Metrics and input contracts

All quality scores below are in **[0,1]**, larger is better, and aggregate as the mean of *valid* question-level values only. Reports expose `n`, sample standard deviation (where computed), missing/errors, and individual results. No missing score is converted to zero and there is no arbitrary composite score.

| Metric | Class | Inputs | Definition | Limitations |
|---|---|---|---|---|
| Faithfulness | LLM judge | question, real generated answer, real retrieved context | DeepEval `FaithfulnessMetric`: judged supported claim fraction | faithfulness does not verify real-world truth or completeness of context |
| Answer relevance | LLM judge | question, generated answer | DeepEval `AnswerRelevancyMetric`: responsiveness of statements | relevant but inaccurate answers can score high |
| Context precision | LLM judge / reference-based | question, generated answer, expected answer, ranked passages | DeepEval `ContextualPrecisionMetric`: average precision of judged relevant retrieved nodes | depends on reference answer and judge; **not** `precision_at_k` |
| Context recall | LLM judge / reference-based | question, answer, expected answer, passages | DeepEval `ContextualRecallMetric`: coverage of reference facts | depends on reference completeness |
| Context relevancy | LLM judge | question, answer, passages | DeepEval `ContextualRelevancyMetric`: retrieved text's question relevance | not a ranking-sensitive context precision substitute |
| Hit Rate@k | retrieval ground truth | verified IDs, ranked retrieved passages | 1 if any relevant ID appears among returned top-k passages, else 0 | incomplete annotations bias results |
| Precision@k | retrieval ground truth | verified IDs, retrieved passages | relevant retrieved passage count / requested k | duplicates of relevant documents can count at multiple ranks; missing top-k positions count as nonrelevant |
| Recall@k | retrieval ground truth | verified IDs, retrieved passages | count of unique relevant IDs found / all annotated relevant IDs | doc or chunk granularity must be consistent |
| MRR | retrieval ground truth | verified IDs, ranks | reciprocal rank of first relevant passage | ignores later relevant passages |
| nDCG@k | retrieval ground truth | verified IDs, ranks | binary gain discounted cumulative gain / ideal DCG | graded relevance not implemented |

If annotated chunk IDs are present, chunk IDs take precedence over document IDs. Because chunk IDs change across chunk configurations, **prefer stable relevant document IDs for chunk-size sweeps**. Deterministic retrieval scores are `unavailable` when relevance labels are missing, identifiers are absent, or the evidence is insufficient. Missing expected answer makes Contextual Precision/Recall unavailable. Missing contexts makes grounding-oriented metrics unavailable. Other exceptions are recorded as metric errors.

**Pass/fail** applies only to LLM-judge scores with configured `judge_threshold`. Deterministic retrieval scores are informational, not automatically pass/fail. Accuracy against reference answers, citation correctness/completeness, unsupported-claim rate, cost estimates and graded relevance were not implemented because their required trusted inputs or provider accounting were not available; do not mistake absent columns for zeros.

LLM-as-a-judge is nondeterministic across model releases/prompts/provider behavior even when the generation seed is fixed; the cache is exact-trace based and may suppress repeated judge sampling. Disable cache and rerun questions to investigate evaluator variability. Do not infer a confidence interval from cached duplicate values.
