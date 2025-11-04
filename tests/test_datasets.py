import json
import pytest
from rag_eval.datasets.loader import parse_bytes, to_csv, to_jsonl, diagnostics
from rag_eval.config import Question


def test_dataset_version_stable(fixtures, repo):
    assert parse_bytes(to_jsonl(fixtures), 'x.jsonl')[0].dataset_version == fixtures[0].dataset_version
    assert repo.put_dataset(fixtures) == repo.put_dataset(fixtures)
    assert repo.dataset(fixtures[0].dataset_version) == fixtures


def test_duplicate_id():
    one = {'question_id': 'x', 'question': 'What is X?'}
    with pytest.raises(ValueError, match='duplicate'):
        parse_bytes((json.dumps(one) + '\n' + json.dumps(one)).encode(), 'x.jsonl')


def test_missing_columns():
    with pytest.raises(ValueError, match='columns'):
        parse_bytes(b'question\nhello\n', 'x.csv')


def test_unknown_fields_and_invalid_ids():
    with pytest.raises(ValueError):
        parse_bytes(b'{"question_id":"x","question":"What is X?","injected":true}', 'x.jsonl')
    with pytest.raises(ValueError):
        Question(question_id='x', question='What is X?', relevant_document_ids=[' '])


def test_ground_truth_and_verified_requirements():
    with pytest.raises(ValueError, match='verified'):
        Question(question_id='x', question='What is X?', annotation_status='verified', sufficient_evidence=True)
    q = Question(question_id='x', question='What is X?')
    assert diagnostics([q])[0]['missing_reference']


def test_csv_roundtrip(fixtures):
    new = parse_bytes(to_csv(fixtures), 'x.csv')
    assert len(new) == len(fixtures)
    assert new[0].relevant_document_ids == ['fixture-doc-a']


def test_size_format_and_missing_version(fixtures, repo):
    with pytest.raises(ValueError, match='larger'):
        parse_bytes(b'x' * (5 * 1024 * 1024 + 1), 'x.jsonl')
    with pytest.raises(ValueError, match='jsonl'):
        parse_bytes(b'whatever', 'x.docx')
    with pytest.raises(KeyError):
        repo.dataset('does-not-exist')
    with pytest.raises(ValueError):
        repo.put_dataset([fixtures[0].model_copy(update={'dataset_version': 'pending'})])


def test_revision_creates_new_version(fixtures, repo):
    original = repo.put_dataset(fixtures)
    replacement = [fixtures[0].model_copy(update={'expected_answer': 'a revised answer'})] + fixtures[1:]
    revised = parse_bytes(to_jsonl(replacement), 'rev.jsonl')
    assert repo.put_dataset(revised) != original
    assert repo.dataset(original)[0].expected_answer != repo.dataset(revised[0].dataset_version)[0].expected_answer
