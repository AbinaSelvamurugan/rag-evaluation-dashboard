"""Validated immutable experimental parameters and fingerprinting."""
from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


def digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    name: str = Field(min_length=1, max_length=100)
    dataset_version: str = Field(min_length=8, max_length=100)
    adapter: Literal['fixture', 'project6_http'] = 'project6_http'
    chunk_strategy: str = Field(default='recursive', min_length=1, max_length=60)
    chunk_size: int = Field(default=512, ge=32, le=8192)
    chunk_overlap: int = Field(default=64, ge=0, le=4096)
    chunk_unit: Literal['characters', 'tokens'] = 'characters'
    embedding_model: str = Field(default='project6-default', min_length=1)
    retriever: str = Field(default='dense', min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    similarity_threshold: float | None = Field(default=None, ge=0, le=1)
    reranker: str | None = None
    generation_model: str = 'project6-default'
    prompt_version: str = 'project6-default'
    context_budget: int = Field(default=4096, ge=128, le=131072)
    judge_model: str | None = None
    judge_threshold: float = Field(default=0.7, ge=0, le=1)
    metrics: tuple[str, ...] = ('faithfulness', 'answer_relevance', 'context_precision', 'hit_rate', 'precision_at_k', 'recall_at_k', 'mrr', 'ndcg')
    seed: int = 42
    concurrency: int = Field(default=1, ge=1, le=8)
    max_questions: int = Field(default=25, ge=1, le=10000)
    timeout_seconds: float = Field(default=60, ge=1, le=600)
    max_retries: int = Field(default=2, ge=0, le=5)
    use_cache: bool = True
    split: Literal['development', 'holdout', 'all'] = 'development'

    @model_validator(mode='after')
    def valid(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('chunk_overlap must be strictly less than chunk_size, in the same unit')
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError('duplicate metrics')
        if not self.metrics:
            raise ValueError('choose at least one metric')
        from rag_eval.metrics.registry import METRICS
        invalid = set(self.metrics) - set(METRICS)
        if invalid:
            raise ValueError(f'unsupported metrics: {sorted(invalid)}')
        if self.adapter == 'fixture' and any(METRICS[x]['kind'] == 'llm-judge' for x in self.metrics):
            # Offline fixture is intentionally not a quality benchmark.
            raise ValueError('fixture mode permits deterministic retrieval metrics only; disable LLM judges')
        if self.adapter == 'project6_http' and any(METRICS[x]['kind'] == 'llm-judge' for x in self.metrics) and not self.judge_model:
            raise ValueError('judge_model is required for LLM-as-judge metrics')
        return self

    @property
    def fingerprint(self) -> str:
        return digest(self.model_dump())


class Passage(BaseModel):
    model_config = ConfigDict(extra='forbid')
    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=500000)
    score: float | None = None
    source: str | None = None
    metadata: dict = Field(default_factory=dict)


class Trace(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str
    answer: str = Field(max_length=500000)
    passages: list[Passage]
    effective_config: dict
    index_id: str
    index_fingerprint: str
    retrieval_ms: float | None = Field(default=None, ge=0)
    generation_ms: float | None = Field(default=None, ge=0)
    total_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    citations: list[str] = Field(default_factory=list)
    trace_metadata: dict = Field(default_factory=dict)

    @model_validator(mode='after')
    def ranks_valid(self):
        if [p.rank for p in self.passages] != list(range(1, len(self.passages) + 1)):
            raise ValueError('retrieved passages require contiguous ranks starting at 1')
        return self


class IndexInfo(BaseModel):
    index_id: str = Field(min_length=1)
    fingerprint: str = Field(min_length=8)
    effective_config: dict
    chunk_count: int | None = Field(default=None, ge=0)
    index_bytes: int | None = Field(default=None, ge=0)
    indexing_ms: float | None = Field(default=None, ge=0)


class Question(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question_id: str = Field(min_length=1, max_length=150)
    question: str = Field(min_length=3, max_length=12000)
    expected_answer: str | None = None
    relevant_document_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    reference_context: list[str] = Field(default_factory=list)
    category: str = 'unspecified'
    difficulty: Literal['easy', 'medium', 'hard'] = 'medium'
    sufficient_evidence: bool | None = None
    expected_abstention: bool = False
    annotation_status: Literal['unreviewed', 'draft', 'verified'] = 'unreviewed'
    split: Literal['development', 'holdout'] = 'development'
    dataset_version: str = 'pending'

    @model_validator(mode='after')
    def annotations(self):
        for item in self.relevant_chunk_ids + self.relevant_document_ids:
            if not isinstance(item, str) or not item.strip() or len(item) > 1024:
                raise ValueError('invalid relevance identifier')
        if self.annotation_status == 'verified' and self.sufficient_evidence is True and not (self.expected_answer and (self.relevant_chunk_ids or self.relevant_document_ids)):
            raise ValueError('verified answerable questions require answer and relevant IDs')
        return self


class MetricResult(BaseModel):
    score: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None
    passed: bool | None = None
    status: Literal['ok', 'unavailable', 'error']
    error: str | None = None
    cached: bool = False


class QuestionResult(BaseModel):
    question_id: str
    status: Literal['completed', 'failed']
    trace: Trace | None = None
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    error: str | None = None
    started_at: str
    finished_at: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
