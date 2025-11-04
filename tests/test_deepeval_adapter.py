import sys
import types
from rag_eval.config import ExperimentConfig, Question, Passage, Trace
from rag_eval.metrics.judges import evaluate_judge


def inputs():
    config = ExperimentConfig(name='real-mode-simulated-judge-test', dataset_version='aaaaaaaa',
        adapter='project6_http', judge_model='gpt-4o-mini', metrics=('faithfulness','answer_relevance','context_precision'))
    q = Question(question_id='id', question='What does the context say?', expected_answer='The text says A.')
    t = Trace(question=q.question, answer='The text says A.', passages=[Passage(document_id='d', chunk_id='c', rank=1,text='The text says A.')],
              effective_config={}, index_id='id', index_fingerprint='aaaaaaaa')
    return config, q, t


def test_context_precision_requires_reference():
    c,q,t = inputs()
    q.expected_answer = None
    assert evaluate_judge('context_precision', q,t,c).status == 'unavailable'
    assert evaluate_judge('context_recall', q,t,c).status == 'unavailable'


def test_empty_context_unavailable():
    c,q,t=inputs()
    t.passages=[]
    assert evaluate_judge('faithfulness',q,t,c).score is None


def test_actual_deepeval_contract_with_controlled_test_double(monkeypatch):
    """Checks mapping into the real DeepEval class names; does NOT judge real text."""
    c,q,t=inputs()
    seen=[]
    class LLMTestCase:
        def __init__(self, **kwargs):
            seen.append(kwargs)
    class Base:
        def __init__(self, **kwargs):
            self.kwargs=kwargs
            self.reason='CONTROLLED TEST ONLY'
            self.score=.5
        def measure(self, case):
            assert case
        def is_successful(self):
            return False
    fake=types.ModuleType('deepeval')
    cases=types.ModuleType('deepeval.test_case')
    cases.LLMTestCase=LLMTestCase
    metrics=types.ModuleType('deepeval.metrics')
    for cls in ('FaithfulnessMetric', 'AnswerRelevancyMetric','ContextualPrecisionMetric','ContextualRecallMetric','ContextualRelevancyMetric'):
        setattr(metrics, cls, Base)
    monkeypatch.setitem(sys.modules, 'deepeval', fake)
    monkeypatch.setitem(sys.modules, 'deepeval.test_case',cases)
    monkeypatch.setitem(sys.modules, 'deepeval.metrics',metrics)
    import rag_eval.metrics.judges as judges
    monkeypatch.setattr(judges, 'version', lambda _: '4.2.8')
    for m in c.metrics:
        result=evaluate_judge(m,q,t,c)
        assert result.score == .5
        assert result.reason == 'CONTROLLED TEST ONLY'
        assert result.passed is False
    assert all(x['actual_output']==t.answer and x['input']==q.question and x['retrieval_context']==['The text says A.'] for x in seen)
    assert seen[-1]['expected_output']==q.expected_answer
