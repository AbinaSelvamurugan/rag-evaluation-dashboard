import pytest
from rag_eval.adapters.fixture import FixtureAdapter
from rag_eval.adapters.project6_adapter import Project6HTTPAdapter, IntegrationError
from rag_eval.config import IndexInfo, Trace


def test_fixture_rebuild_and_rank(fixture_config):
    a=FixtureAdapter()
    index=a.prepare_index(fixture_config)
    t=a.run_question('What is context precision?', fixture_config,index)
    assert t.passages[0].rank==1
    assert '[FIXTURE RESPONSE' in t.answer
    assert all(p.source=='TEST FIXTURE' for p in t.passages)
    other=fixture_config.model_copy(update={'chunk_size':64, 'chunk_overlap':0})
    assert a.prepare_index(other).fingerprint != index.fingerprint


def test_reject_unsafe_endpoint():
    with pytest.raises(IntegrationError):
        Project6HTTPAdapter('http://example.org')
    with pytest.raises(IntegrationError):
        Project6HTTPAdapter('')


def test_capability_and_trace_contract(monkeypatch, fixture_config):
    config=fixture_config.model_copy(update={'adapter':'project6_http', 'retriever':'keyword'})
    a=Project6HTTPAdapter('http://127.0.0.1:8050')
    calls=[]
    def fake_call(method,path,payload=None,timeout=30,retries=2):
        calls.append(path)
        if path=='/eval/capabilities':
            return {'protocol':'ragbench-v1', 'retrievers':['keyword'], 'chunk_units':['characters'],
                    'chunk_strategies':['fixed'], 'supports_reindex':True}
        if path=='/eval/indexes':
            return {'index_id':'idx', 'fingerprint':'abcdefgh', 'effective_config':{
                k:getattr(config,k) for k in ('chunk_strategy','chunk_size','chunk_overlap','chunk_unit','embedding_model')}}
        if path=='/eval/trace':
            return {'question':payload['question'], 'answer':'REAL TRACE CONTRACT TEST (NOT PROJECT 6)',
                   'passages':[{'document_id':'d', 'chunk_id':'c','rank':1,'text':'actual returned passage'}],
                   'effective_config':{k:getattr(config,k) for k in ('retriever','top_k','generation_model','prompt_version','reranker','similarity_threshold','context_budget')},
                   'index_id':'idx','index_fingerprint':'abcdefgh'}
    monkeypatch.setattr(a,'_call',fake_call)
    index=a.prepare_index(config)
    result=a.run_question('question',config,index)
    assert [p.chunk_id for p in result.passages]==['c']
    assert calls==['/eval/capabilities','/eval/indexes','/eval/trace']


def test_reject_wrong_model(monkeypatch, fixture_config):
    a=Project6HTTPAdapter('http://localhost:8000')
    c=fixture_config
    index=IndexInfo(index_id='x',fingerprint='abcdefgh',effective_config={})
    monkeypatch.setattr(a,'_call',lambda *_a,**_k: {'question':'q', 'answer':'a', 'passages':[],
        'effective_config':{'retriever':'WRONG','top_k':c.top_k,'generation_model':c.generation_model,'prompt_version':c.prompt_version,'reranker':c.reranker,'similarity_threshold':c.similarity_threshold,'context_budget':c.context_budget},
        'index_id':'x','index_fingerprint':'abcdefgh'})
    with pytest.raises(IntegrationError,match='mismatch'):
        a.run_question('q',c,index)


def test_http_retry_for_rate_limit(monkeypatch):
    import httpx
    import rag_eval.adapters.project6_adapter as module
    calls=[]
    def server(request):
        calls.append(request)
        if len(calls)==1:
            return httpx.Response(429, json={'error':'backoff'})
        return httpx.Response(200,json={'protocol':'ragbench-v1','retrievers':[],
                                        'chunk_strategies':[], 'chunk_units':[], 'supports_reindex':False})
    original=httpx.Client
    transport=httpx.MockTransport(server)
    monkeypatch.setattr(module.httpx,'Client',lambda **kw: original(transport=transport,**kw))
    monkeypatch.setattr(module.time,'sleep',lambda _: None)
    a=Project6HTTPAdapter('http://localhost:8000')
    assert a.capabilities()['protocol']=='ragbench-v1'
    assert len(calls)==2


def test_http_permission_failure_not_retried(monkeypatch):
    import httpx
    import rag_eval.adapters.project6_adapter as module
    calls=[]
    def server(request):
        calls.append(request)
        return httpx.Response(401, json={'error':'denied'})
    original=httpx.Client
    monkeypatch.setattr(module.httpx,'Client',lambda **kw: original(transport=httpx.MockTransport(server), **kw))
    with pytest.raises(IntegrationError,match='request failed'):
        Project6HTTPAdapter('http://localhost:8050').capabilities()
    assert len(calls)==1


def test_http_request_timeout(monkeypatch):
    import httpx
    import rag_eval.adapters.project6_adapter as module
    def server(request):
        raise httpx.ReadTimeout('controlled network timeout')
    original=httpx.Client
    monkeypatch.setattr(module.httpx,'Client',lambda **kw: original(transport=httpx.MockTransport(server),**kw))
    monkeypatch.setattr(module.time,'sleep',lambda _: None)
    with pytest.raises(IntegrationError):
        Project6HTTPAdapter('http://localhost:8050').capabilities()


def test_invalid_source_metadata_rejected():
    from pydantic import ValidationError
    from rag_eval.config import Passage
    with pytest.raises(ValidationError):
        Passage(document_id='d', chunk_id='c', rank=1, text='x', metadata=['not','object'])
