"""Metric glossary used by the UI and reports; do not combine as composite score."""
METRICS = {
    'faithfulness': {'kind': 'llm-judge', 'description': 'Fraction of answer claims judged supported by retrieved context.', 'needs': 'question, answer, retrieved text', 'limits': 'Depends on judge and context quality.'},
    'answer_relevance': {'kind': 'llm-judge', 'description': 'Relevance of response statements to the user question.', 'needs': 'question, answer', 'limits': 'Not an accuracy or grounding measure.'},
    'context_precision': {'kind': 'llm-judge', 'description': 'DeepEval Contextual Precision: ranks judged relevant retrieved nodes ahead of irrelevant ones.', 'needs': 'question, answer, reference answer, ranked contexts', 'limits': 'Reference-dependent judge, not deterministic precision@k.'},
    'context_recall': {'kind': 'llm-judge', 'description': 'Coverage of reference answer facts by retrieved passages.', 'needs': 'question, answer, reference answer, contexts', 'limits': 'Reference answer completeness affects estimate.'},
    'context_relevancy': {'kind': 'llm-judge', 'description': 'Overall question-relevance of retrieved text.', 'needs': 'question, answer, contexts', 'limits': 'Not the same as ranked context precision.'},
    'hit_rate': {'kind': 'retrieval-ground-truth', 'description': '1 if any annotated relevant item appears within top k, otherwise 0.', 'needs': 'relevant document/chunk IDs, ranked passages', 'limits': 'Requires exhaustive reliable relevance annotations.'},
    'precision_at_k': {'kind': 'retrieval-ground-truth', 'description': 'Relevant retrieved passages divided by requested k (missing positions count as nonrelevant).', 'needs': 'relevant IDs, ranked passages', 'limits': 'Underretrieval is penalized with the requested-k denominator.'},
    'recall_at_k': {'kind': 'retrieval-ground-truth', 'description': 'Relevant annotated IDs retrieved divided by total annotated relevant IDs.', 'needs': 'relevant IDs, ranked passages', 'limits': 'Chunk IDs may change with reindexing; use stable doc IDs across chunk sweeps.'},
    'mrr': {'kind': 'retrieval-ground-truth', 'description': 'Reciprocal rank of first relevant retrieved passage.', 'needs': 'relevant IDs, ranked passages', 'limits': 'Captures only first relevant passage.'},
    'ndcg': {'kind': 'retrieval-ground-truth', 'description': 'Binary-gain DCG normalized by best possible ordering at k.', 'needs': 'relevant IDs, ranked passages', 'limits': 'Binary relevance unless annotation is extended.'},
}
# All scores are normalized into [0,1], higher is better, arithmetic mean over
# successful question-level values; missing/error values are excluded, counts shown.
