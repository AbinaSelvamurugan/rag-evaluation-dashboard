"""Idempotent, bounded, cancellable run coordinator with persistent traces."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from rag_eval.adapters.fixture import make_adapter
from rag_eval.config import ExperimentConfig, IndexInfo, MetricResult, Question, QuestionResult, Trace, digest, utc_now
from rag_eval.metrics.retrieval import calculate, RETRIEVAL_NAMES
from rag_eval.metrics.judges import evaluate_judge
from rag_eval.metrics.registry import METRICS
from rag_eval.storage.repository import Repository


def environment_snapshot() -> dict:
    packages = {}
    for pkg in ('ragbench-studio', 'deepeval', 'streamlit', 'pydantic', 'pandas', 'numpy', 'plotly', 'httpx'):
        try:
            packages[pkg] = version(pkg)
        except PackageNotFoundError:
            packages[pkg] = None
    return {'python': platform.python_version(), 'packages': packages, 'evaluation_framework': 'deepeval'}


def select_questions(repo: Repository, config: ExperimentConfig) -> list[Question]:
    records = repo.dataset(config.dataset_version)
    selected = [x for x in records if config.split == 'all' or x.split == config.split]
    if not selected:
        raise ValueError(f'no questions in split {config.split}')
    return selected[:config.max_questions]


def plan(repo: Repository, config: ExperimentConfig) -> dict:
    qs = select_questions(repo, config)
    judge_names = [m for m in config.metrics if METRICS[m]['kind'] == 'llm-judge']
    return {'questions': len(qs), 'rag_requests': len(qs), 'judge_metric_evaluations': len(qs) * len(judge_names),
            'index_rebuilds': 1, 'llm_metrics': judge_names,
            'warning': 'One LLM metric evaluation can make multiple provider calls; costs unknown without provider pricing/usage.'}


def start_run(repo: Repository, config: ExperimentConfig) -> str:
    return repo.create_run(config, plan(repo, config)['questions'], environment_snapshot())


def _eval_one(q: Question, config: ExperimentConfig, adapter, index: IndexInfo, repo: Repository) -> QuestionResult:
    started = utc_now()
    try:
        trace = adapter.run_question(q.question, config, index)
        if trace.question != q.question:
            raise ValueError('adapter returned trace for a different question')
        if trace.index_fingerprint != index.fingerprint:
            raise ValueError('adapter returned a mismatched index fingerprint')
        scores = {}
        retrieval_scores = calculate(q, trace, config.top_k)
        for name in config.metrics:
            if name in RETRIEVAL_NAMES:
                scores[name] = retrieval_scores[name]
                continue
            key = digest({'question': q.model_dump(), 'trace': trace.model_dump(), 'metric': name,
                          'model': config.judge_model, 'threshold': config.judge_threshold,
                          'framework_version': environment_snapshot()['packages']['deepeval']})
            cached = repo.cache_get(key) if config.use_cache else None
            if cached is not None:
                scores[name] = MetricResult.model_validate(cached).model_copy(update={'cached': True})
            else:
                result = evaluate_judge(name, q, trace, config)
                scores[name] = result
                if result.status == 'ok' and config.use_cache:
                    repo.cache_put(key, result.model_dump())
        return QuestionResult(question_id=q.question_id, status='completed', trace=trace,
                              metrics=scores, started_at=started, finished_at=utc_now())
    except Exception as exc:
        return QuestionResult(question_id=q.question_id, status='failed',
                              error=f'{type(exc).__name__}: {str(exc)[:500]}',
                              started_at=started, finished_at=utc_now())


def execute_run(repo: Repository, run_id: str, adapter=None) -> dict:
    """Run synchronously; UI may invoke in a daemon worker. Resume skips saved records."""
    row = repo.status(run_id)
    if row['status'] == 'completed':
        return row
    config = ExperimentConfig.model_validate(row['config'])
    questions = select_questions(repo, config)
    repo.update_run(run_id, status='running')
    try:
        if repo.status(run_id)['cancelled']:
            repo.update_run(run_id, status='cancelled')
            return repo.status(run_id)
        adapter = adapter or make_adapter(config.adapter)
        index = adapter.prepare_index(config)
        for previous in repo.runs():
            prior_index = previous['index_data']
            if previous['id'] == run_id or not prior_index:
                continue
            old = previous['config']
            fields = ('chunk_size', 'chunk_overlap', 'chunk_strategy', 'chunk_unit', 'embedding_model')
            if any(old.get(k) != getattr(config, k) for k in fields):
                if (prior_index['index_id'] == index.index_id or prior_index['fingerprint'] == index.fingerprint):
                    raise ValueError('Project 6 reused index identity across different chunk/embedding configurations')
        if row['index_data'] and row['index_data']['fingerprint'] != index.fingerprint:
            raise ValueError('resume index fingerprint changed; cannot safely reuse results')
        repo.update_run(run_id, status='running', index_data=index.model_dump())
        existing = {x.question_id for x in repo.results(run_id)}
        pending = iter(x for x in questions if x.question_id not in existing)
        with ThreadPoolExecutor(max_workers=config.concurrency, thread_name_prefix='ragbench') as executor:
            inflight = {}
            for _ in range(config.concurrency):
                q = next(pending, None)
                if q is not None and not repo.status(run_id)['cancelled']:
                    inflight[executor.submit(_eval_one, q, config, adapter, index, repo)] = q.question_id
            while inflight:
                completed, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in completed:
                    inflight.pop(future)
                    # Every completed request is persisted before updating total progress.
                    repo.save_result(run_id, future.result())
                    if not repo.status(run_id)['cancelled']:
                        q = next(pending, None)
                        if q is not None:
                            inflight[executor.submit(_eval_one, q, config, adapter, index, repo)] = q.question_id
        status = repo.status(run_id)
        if status['cancelled']:
            repo.update_run(run_id, status='cancelled')
        elif status['done'] != status['total']:
            repo.update_run(run_id, status='incomplete', error='question results missing')
        elif status['failed']:
            repo.update_run(run_id, status='incomplete', error=f'{status["failed"]} question(s) failed; inspect traces')
        else:
            missing = [(r.question_id, name, m.status) for r in repo.results(run_id)
                       for name, m in r.metrics.items() if m.status != 'ok']
            if missing:
                repo.update_run(run_id, status='incomplete',
                    error=f'{len(missing)} unavailable/failed metric evaluation(s); first: {missing[0]}')
            else:
                repo.update_run(run_id, status='completed')
    except Exception as exc:
        repo.update_run(run_id, status='failed', error=f'{type(exc).__name__}: {str(exc)[:500]}')
    return repo.status(run_id)
