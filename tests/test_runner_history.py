import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from rag_eval.adapters.fixture import FixtureAdapter
from rag_eval.config import ExperimentConfig, QuestionResult, MetricResult, utc_now
from rag_eval.experiments.runner import start_run, execute_run, plan
from rag_eval.experiments.planner import matrix
from rag_eval.reporting.report import markdown_report, json_export


def test_end_to_end_offline_trace_and_history(repo,fixture_config):
    info=plan(repo,fixture_config)
    assert info['questions']==3
    assert info['judge_metric_evaluations']==0
    run_id=start_run(repo,fixture_config)
    run=execute_run(repo,run_id)
    assert run['status']=='completed'
    assert run['done']==3
    assert run['index_data']['fingerprint']
    results=repo.results(run_id)
    assert len(results)==3
    assert all(r.trace is not None and 'FIXTURE' in r.trace.answer for r in results)
    assert all(r.metrics['hit_rate'].status=='ok' for r in results)
    assert execute_run(repo,run_id)['done']==3
    assert len(repo.results(run_id))==3
    text=markdown_report(repo,run_id)
    assert 'TEST FIXTURE' in text and 'Rank 1' in text
    dump=json.loads(json_export(repo,run_id))
    assert len(dump['question_results'])==3
    assert repo.status(run_id)['environment']['packages']['pydantic']


def test_cancel_queued_run(repo,fixture_config):
    id=start_run(repo,fixture_config)
    repo.cancel(id)
    run=execute_run(repo,id)
    assert run['status']=='cancelled'
    assert run['done']==0


def test_run_failure_persisted_not_fake_completion(repo,fixture_config):
    id=start_run(repo,fixture_config.model_copy(update={'adapter':'project6_http'}))
    status=execute_run(repo,id)
    assert status['status']=='failed'
    assert status['done']==0
    assert status['error']


def test_resume_skips_existing_records(repo,fixture_config):
    id=start_run(repo,fixture_config)
    started=utc_now()
    repo.save_result(id,QuestionResult(question_id='fixture-q1',status='completed',
      started_at=started,finished_at=started, metrics={'hit_rate':MetricResult(score=0.9,status='ok')}))
    r=execute_run(repo,id)
    assert r['status']=='completed'
    assert len(repo.results(id))==3
    assert next(x for x in repo.results(id) if x.question_id=='fixture-q1').metrics['hit_rate'].score==.9


def test_matrix_config_variation(repo,fixture_config):
    cfgs=matrix(fixture_config,[64,128],[0,0.1],['keyword'])
    assert len(cfgs)==4
    assert len({x.fingerprint for x in cfgs})==4
    assert all(x.chunk_overlap < x.chunk_size for x in cfgs)
    with pytest.raises(ValueError):
        matrix(fixture_config,list(range(32,140)),[0],['keyword'])


def test_failing_adapter_question_results(repo,fixture_config):
    class Broken(FixtureAdapter):
        def run_question(self, question, config, index):
            raise TimeoutError('controlled timeout')
    rid=start_run(repo,fixture_config)
    row=execute_run(repo,rid,adapter=Broken())
    assert row['status']=='incomplete'
    assert row['failed']==3
    assert all('TimeoutError' in r.error and r.trace is None for r in repo.results(rid))


def test_same_database_opened_twice_consistently(repo, fixture_config):
    from rag_eval.storage.repository import Repository
    rid=start_run(repo,fixture_config)
    second=Repository(repo.root)
    assert second.status(rid)['config']['dataset_version']==fixture_config.dataset_version


def test_metric_errors_do_not_mark_run_completed(repo, fixture_config, monkeypatch):
    from rag_eval.config import MetricResult
    import rag_eval.experiments.runner as runner
    config = fixture_config.model_copy(update={'adapter': 'project6_http', 'judge_model': 'gpt-test',
                                                'metrics': ('faithfulness',)})
    monkeypatch.setattr(runner, 'evaluate_judge', lambda *args: MetricResult(status='error', error='TEST provider unavailable'))
    rid = start_run(repo, config)
    result = execute_run(repo, rid, adapter=FixtureAdapter())
    assert result['status'] == 'incomplete'
    assert 'unavailable/failed' in result['error']
    assert all(r.metrics['faithfulness'].score is None for r in repo.results(rid))


def test_refuses_same_index_across_different_chunk_sizes(repo, fixture_config):
    class ReusesIndex(FixtureAdapter):
        def prepare_index(self, config):
            idx = super().prepare_index(config)
            return idx.model_copy(update={'index_id': 'same-id', 'fingerprint': 'same-fingerprint'})
        def run_question(self, question, config, index):
            return super().run_question(question, config,
                super().prepare_index(config)).model_copy(update={'index_id': index.index_id,
                                                                  'index_fingerprint': index.fingerprint})
    a = fixture_config.model_copy(update={'chunk_size':64,'chunk_overlap':0})
    b = fixture_config.model_copy(update={'chunk_size':128,'chunk_overlap':0})
    id_a = start_run(repo,a)
    assert execute_run(repo,id_a,adapter=ReusesIndex())['status']=='completed'
    id_b = start_run(repo,b)
    row=execute_run(repo,id_b,adapter=ReusesIndex())
    assert row['status']=='failed'
    assert 'reused index identity' in row['error']
