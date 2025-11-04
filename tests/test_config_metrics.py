from math import isclose
import pytest
from pydantic import ValidationError
from rag_eval.config import ExperimentConfig, Passage, Trace, Question
from rag_eval.metrics.retrieval import calculate
from rag_eval.evaluation.statistics import summaries, paired_delta
from rag_eval.config import MetricResult, QuestionResult


def trace(passages=None):
    ps = passages or [('d2', 'c2'), ('d1', 'c1'), ('d3', 'c3')]
    return Trace(question='What is X?', answer='answer', index_id='foo', index_fingerprint='abcdefgh',
                 effective_config={}, passages=[Passage(document_id=d, chunk_id=c, rank=i+1, text=f'Passage {i}') for i, (d,c) in enumerate(ps)])


def test_validation_guards(fixture_config):
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({**fixture_config.model_dump(), 'chunk_size':64, 'chunk_overlap':64})
    with pytest.raises(ValidationError, match='fixture mode'):
        ExperimentConfig.model_validate({**fixture_config.model_dump(), 'metrics':('faithfulness',)})
    with pytest.raises(ValidationError, match='unsupported metrics'):
        ExperimentConfig.model_validate({**fixture_config.model_dump(), 'metrics':('fake',)})
    with pytest.raises(ValidationError, match='judge_model'):
        ExperimentConfig.model_validate({**fixture_config.model_dump(), 'adapter':'project6_http', 'metrics':('faithfulness',), 'judge_model':None})


def test_rank_schema():
    with pytest.raises(ValidationError):
        Trace(question='What is X?', answer='answer', index_id='foo', index_fingerprint='abcdefgh',
              effective_config={}, passages=[Passage(document_id='d', chunk_id='c', rank=3, text='Hi')])


def test_retrieval_metrics_true_ranking():
    q = Question(question_id='x', question='What is X?', expected_answer='X', annotation_status='verified',
                 sufficient_evidence=True, relevant_document_ids=['d1','d3'])
    m = calculate(q, trace())
    assert m['hit_rate'].score == 1
    assert isclose(m['precision_at_k'].score, 2/3, abs_tol=1e-7)
    assert calculate(q, trace(), top_k=5)['precision_at_k'].score == 0.4
    assert m['recall_at_k'].score == 1
    assert m['mrr'].score == 0.5
    assert 0 < m['ndcg'].score < 1


def test_missing_annotations_are_not_zero():
    q = Question(question_id='x', question='What is X?')
    assert all(x.score is None and x.status == 'unavailable' for x in calculate(q, trace()).values())


def test_empty_retrieval_for_verified_question_is_legitimate_zero():
    q = Question(question_id='x', question='What is X?', expected_answer='X', annotation_status='verified',
                 sufficient_evidence=True, relevant_document_ids=['d1'])
    t = trace()
    t.passages = []
    assert all(x.score == 0 for x in calculate(q, t).values())


def test_summaries_missing_and_paired_ci():
    def record(n, v):
        return QuestionResult(question_id=f'q{n}', status='completed', started_at='0', finished_at='1',
                  metrics={'m':MetricResult(score=v, status='ok')} if v is not None else
                  {'m':MetricResult(status='error', error='provider down')})
    a = [record(i, 0.2) for i in range(7)]
    b = [record(i, 0.5) for i in range(7)]
    b[0] = record(0, None)
    assert summaries(b)[0]['n'] == 6
    assert summaries(b)[0]['missing_or_failed'] == 1
    d = paired_delta(a, b, 'm')
    assert d['n_paired'] == 6
    assert abs(d['delta'] - 0.3) < 1e-10
    assert d['ci_low'] is not None
    assert paired_delta(a[:3], b[:3], 'm')['ci_low'] is None
