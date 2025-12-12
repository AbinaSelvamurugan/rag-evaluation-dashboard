# Evaluation methodology and annotation protocol

## Problem statement

A RAG answer can be topical but wrong, context-grounded but incomplete, or factually correct only because the generator guessed without evidence. Measure retrieval, grounded generation and responsiveness **separately**, not through an undocumented weighted score.

## Dataset construction

Start from the actual Project 6 source corpus. For each question, read the primary source, record source-document IDs and optional chunk IDs, and write an expected answer **before** evaluating the model. Have a second reviewer check the reference and evidence coverage. Mark `annotation_status=verified` only after validation. Add categories for factual, exact-ID, multipart, multi-document, troubleshooting, ambiguous, insufficient-evidence and (if supported) conversational follow-up questions. Mark `sufficient_evidence=false` for unanswerable queries; expected abstention is recorded separately. The real corpus must determine the expected answer; **never** use the system's generated answer as its own ground truth.

Do not use chunk IDs from a baseline index as the only relevance labels in a new chunked index. Stable source-document IDs are preferable for cross-configuration comparisons. Use consistent semantics for denominator sets. Questions with nonexhaustive annotations must remain draft/unreviewed and their retrieval metrics unavailable.

Separate `development` and reserved `holdout` questions; do not inspect or tune on holdout repeatedly. Dataset versions are SHA-256-derived and any annotation change creates a new version; configurations record the exact version.

## Failure taxonomy

- **Retrieval omission:** relevant annotated source absent from top-k (verified labels required).
- **Poor ranking:** relevant chunks exist but appear late; low MRR/nDCG or DeepEval contextual precision.
- **Generation grounding failure:** pertinent evidence retrieved but unsupported answer claims; low faithfulness.
- **Answer intent failure:** context may be correct but response doesn't address the question; low answer relevance.
- **Inadequate corpus evidence:** expected abstention; separate assessment, not a forced numerical zero.
- **Measurement failure:** unavailable input, judge/provider exception, or trace failure. Never call this a quality-zero outcome.

## Limitations and bias

Judges are model-dependent, sensitive to prompt/instruction and retrieval phrasing, may misclassify claims, and can change with model revisions. Rerun a subset with cache disabled to quantify variability; do not misrepresent cached repeats as independent measurements. Question labels may be incomplete, chunk count and relevance denominators may shift, providers may vary latencies, and token/cost accounting may be unavailable. Context recall isn't a guarantee that underlying documents are true. DeepEval Contextual Precision is based on judged relevance to expected answer, not manually verified passage IDs.
