"""Actual retrieved identifiers versus verified human-labeled ground truth."""
from math import log2
from rag_eval.config import MetricResult, Question, Trace

RETRIEVAL_NAMES = ('hit_rate', 'precision_at_k', 'recall_at_k', 'mrr', 'ndcg')


def calculate(question: Question, trace: Trace, top_k: int | None = None) -> dict[str, MetricResult]:
    k = top_k if top_k is not None else len(trace.passages)
    if k < 0:
        raise ValueError('top_k must be nonnegative')
    ids = (set(question.relevant_chunk_ids) if question.relevant_chunk_ids else set(question.relevant_document_ids))
    if question.annotation_status != 'verified' or not ids or question.sufficient_evidence is not True:
        return {name: MetricResult(status='unavailable', reason='Requires verified, answerable relevance annotations') for name in RETRIEVAL_NAMES}
    chunks = bool(question.relevant_chunk_ids)
    found = [p.chunk_id if chunks else p.document_id for p in trace.passages[:k]]
    rel = [int(x in ids) for x in found]
    unique_matches = len(set(found) & ids)
    first = next((i + 1 for i, v in enumerate(rel) if v), None)
    dcg = sum(v / log2(i + 2) for i, v in enumerate(rel))
    ideal = sum(1 / log2(i + 2) for i in range(min(len(ids), k)))
    values = {
        'hit_rate': float(any(rel)),
        'precision_at_k': sum(rel) / k if k else 0.0,
        'recall_at_k': unique_matches / len(ids),
        'mrr': 1.0 / first if first else 0.0,
        'ndcg': dcg / ideal if ideal else 0.0,
    }
    return {name: MetricResult(status='ok', score=round(min(value, 1.0), 8)) for name, value in values.items()}
