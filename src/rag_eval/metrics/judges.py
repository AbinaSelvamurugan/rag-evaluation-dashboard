"""DeepEval 4.2.x native metric measurement. No simulated judge outputs."""
from importlib.metadata import version
from rag_eval.config import ExperimentConfig, MetricResult, Question, Trace

JUDGE_NAMES = ('faithfulness', 'answer_relevance', 'context_precision', 'context_recall', 'context_relevancy')


def evaluate_judge(name: str, question: Question, trace: Trace, config: ExperimentConfig) -> MetricResult:
    if name not in JUDGE_NAMES:
        raise ValueError(f'unknown LLM judge: {name}')
    if not config.judge_model:
        return MetricResult(status='unavailable', reason='judge_model unset')
    if name in ('context_precision', 'context_recall') and not question.expected_answer:
        return MetricResult(status='unavailable', reason='expected_answer required by DeepEval')
    if name != 'answer_relevance' and not trace.passages:
        return MetricResult(status='unavailable', reason='retrieval_context is empty')
    try:
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import (FaithfulnessMetric, AnswerRelevancyMetric,
            ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric)
        # version is stored in run config as well, not just cached computed scores.
        version('deepeval')
        cls = {'faithfulness': FaithfulnessMetric, 'answer_relevance': AnswerRelevancyMetric,
               'context_precision': ContextualPrecisionMetric, 'context_recall': ContextualRecallMetric,
               'context_relevancy': ContextualRelevancyMetric}[name]
        case = LLMTestCase(input=question.question, actual_output=trace.answer,
                           expected_output=question.expected_answer,
                           retrieval_context=[p.text for p in trace.passages])
        metric = cls(model=config.judge_model, threshold=config.judge_threshold,
                     include_reason=True, async_mode=False)
        metric.measure(case)
        if metric.score is None:
            return MetricResult(status='unavailable', reason=getattr(metric, 'reason', None))
        return MetricResult(status='ok', score=float(metric.score),
                            reason=getattr(metric, 'reason', None),
                            passed=bool(metric.is_successful()))
    except Exception as exc:
        return MetricResult(status='error', error=f'{type(exc).__name__}: {str(exc)[:350]}')
