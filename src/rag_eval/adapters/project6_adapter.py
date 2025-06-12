"""Strict HTTP integration boundary. Does not guess Project 6 API semantics."""
import os
from urllib.parse import urlsplit
import time
import httpx
from rag_eval.adapters.base import RAGAdapter
from rag_eval.config import ExperimentConfig, IndexInfo, Trace, digest


class IntegrationError(RuntimeError):
    pass


class Project6HTTPAdapter(RAGAdapter):
    verified = False  # becomes true only on successful handshake

    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base_url = (base_url or os.getenv('PROJECT6_EVAL_URL') or '').rstrip('/')
        self.token = token if token is not None else os.getenv('PROJECT6_EVAL_TOKEN', '')
        parsed = urlsplit(self.base_url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise IntegrationError('Project 6 URL must not include credentials, query, or fragment')
        if not self.base_url or not self.base_url.startswith(('https://', 'http://')):
            raise IntegrationError('PROJECT6_EVAL_URL must be set to a configured Project 6 evaluation bridge')
        if not self.base_url.startswith('https://') and not (self.base_url.startswith('http://127.0.0.1:') or self.base_url.startswith('http://localhost:')):
            raise IntegrationError('plain HTTP permitted only for localhost; use TLS elsewhere')
        self._caps = None

    def _call(self, method: str, path: str, payload: dict | None = None, timeout: float = 30, retries: int = 2) -> dict:
        headers = {'Authorization': f'Bearer {self.token}'} if self.token else {}
        for attempt in range(retries + 1):
            try:
                with httpx.Client(timeout=timeout, follow_redirects=False, headers=headers) as client:
                    response = client.request(method, self.base_url + path, json=payload)
                    response.raise_for_status()
                    if len(response.content) > 8 * 1024 * 1024:
                        raise IntegrationError('Project 6 returned more than 8 MiB')
                    value = response.json()
                    if not isinstance(value, dict):
                        raise IntegrationError(f'{path} response must be JSON object')
                    return value
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                transient = isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)) or (isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (429, 502, 503, 504))
                if not transient or attempt == retries:
                    raise IntegrationError(f'Project6 {path} request failed: {type(exc).__name__}') from exc
                time.sleep(min(2 ** attempt, 8))
        raise AssertionError('unreachable')

    def capabilities(self) -> dict:
        if self._caps is None:
            caps = self._call('GET', '/eval/capabilities')
            if caps.get('protocol') != 'ragbench-v1':
                raise IntegrationError('Project 6 bridge requires protocol=ragbench-v1')
            for key in ('retrievers', 'chunk_units', 'chunk_strategies', 'supports_reindex'):
                if key not in caps:
                    raise IntegrationError(f'bridge missing capability: {key}')
            self._caps = caps
            self.verified = True  # protocol handshake succeeded; does not assess quality of the underlying implementation
        return self._caps

    def prepare_index(self, config: ExperimentConfig) -> IndexInfo:
        caps = self.capabilities()
        if config.retriever not in caps['retrievers']:
            raise IntegrationError(f'retriever {config.retriever!r} not advertised by Project 6')
        if config.chunk_unit not in caps['chunk_units'] or config.chunk_strategy not in caps['chunk_strategies']:
            raise IntegrationError('chunk strategy/unit not advertised by Project 6')
        if config.reranker and config.reranker not in caps.get('rerankers', []):
            raise IntegrationError('reranker not supported by Project 6')
        if not caps['supports_reindex']:
            raise IntegrationError('bridge cannot create isolated indexes; chunking experiment blocked')
        response = self._call('POST', '/eval/indexes', config.model_dump(), config.timeout_seconds, config.max_retries)
        index = IndexInfo.model_validate(response)
        expected = {k: getattr(config, k) for k in ('chunk_strategy', 'chunk_size', 'chunk_overlap', 'chunk_unit', 'embedding_model')}
        if any(index.effective_config.get(k) != v for k, v in expected.items()):
            raise IntegrationError('Project 6 index effective config does not match requested settings')
        return index

    def run_question(self, question: str, config: ExperimentConfig, index: IndexInfo) -> Trace:
        start = time.perf_counter()
        response = self._call('POST', '/eval/trace', {'question': question, 'config': config.model_dump(), 'index_id': index.index_id}, config.timeout_seconds, config.max_retries)
        roundtrip_ms = (time.perf_counter() - start) * 1000
        trace = Trace.model_validate(response)
        trace.trace_metadata['client_roundtrip_ms'] = roundtrip_ms  # network + backend; not generation-only latency
        if trace.question != question or trace.index_id != index.index_id or trace.index_fingerprint != index.fingerprint:
            raise IntegrationError('bridge returned wrong question/index; refusing contaminated trace')
        for k in ('retriever', 'top_k', 'generation_model', 'prompt_version',
                  'reranker', 'similarity_threshold', 'context_budget'):
            if trace.effective_config.get(k) != getattr(config, k):
                raise IntegrationError(f'bridge effective configuration mismatch: {k}')
        if len(trace.passages) > config.top_k:
            raise IntegrationError('bridge returned more than top_k passages')
        return trace
