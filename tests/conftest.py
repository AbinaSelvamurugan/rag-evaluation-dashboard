import pytest
from rag_eval.datasets.loader import parse_bytes
from rag_eval.storage.repository import Repository
from pathlib import Path


@pytest.fixture
def repo(tmp_path):
    return Repository(tmp_path)


@pytest.fixture
def fixtures():
    path = Path(__file__).resolve().parents[1] / 'data' / 'smoke_fixture.jsonl'
    return parse_bytes(path.read_bytes(), path.name)


@pytest.fixture
def fixture_config(repo, fixtures):
    from rag_eval.config import ExperimentConfig
    version = repo.put_dataset(fixtures)
    return ExperimentConfig(name='TEST FIXTURE', dataset_version=version, adapter='fixture',
            chunk_strategy='fixed', chunk_unit='characters', chunk_size=128, chunk_overlap=8,
            retriever='keyword', top_k=2, metrics=('hit_rate', 'precision_at_k', 'recall_at_k', 'mrr', 'ndcg'))
