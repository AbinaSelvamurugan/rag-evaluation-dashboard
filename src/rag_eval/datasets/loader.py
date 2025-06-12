"""Size-bounded versioned CSV/JSONL evaluation sets."""
import csv
from io import StringIO
import json
from collections import Counter
from rag_eval.config import Question, digest

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_ROWS = 10000


def parse_bytes(content: bytes, filename: str) -> list[Question]:
    if not content or len(content) > MAX_FILE_BYTES:
        raise ValueError('dataset empty or larger than 5 MiB')
    text = content.decode('utf-8-sig')
    if filename.lower().endswith('.jsonl'):
        rows = [json.loads(x) for x in text.splitlines() if x.strip()]
    elif filename.lower().endswith('.csv'):
        reader = csv.DictReader(StringIO(text))
        required = {'question_id', 'question'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('missing required CSV columns: ' + ', '.join(sorted(required - set(reader.fieldnames or []))))
        rows = []
        for row in reader:
            for col in ('relevant_document_ids', 'relevant_chunk_ids', 'reference_context'):
                if col in row:
                    row[col] = json.loads(row[col] or '[]')
            for col in ('expected_abstention', 'sufficient_evidence'):
                if row.get(col) not in (None, ''):
                    if row[col].lower() not in ('true', 'false'):
                        raise ValueError(f'{col} must be true or false')
                    row[col] = row[col].lower() == 'true'
                elif col in row:
                    row[col] = None if col == 'sufficient_evidence' else False
            rows.append({k: v for k, v in row.items() if v not in (None, '')})
    else:
        raise ValueError('dataset must be .csv or .jsonl')
    if len(rows) > MAX_ROWS or not rows:
        raise ValueError('dataset must contain 1–10000 questions')
    questions = [Question.model_validate(row) for row in rows]
    ids = [q.question_id for q in questions]
    repeated = [x for x, n in Counter(ids).items() if n > 1]
    if repeated:
        raise ValueError(f'duplicate question identifiers: {repeated[:10]}')
    canonical = [{k: v for k, v in q.model_dump().items() if k != 'dataset_version'} for q in questions]
    version = digest(canonical)[:20]
    return [q.model_copy(update={'dataset_version': version}) for q in questions]


def to_jsonl(questions: list[Question]) -> bytes:
    return ('\n'.join(json.dumps(q.model_dump(), ensure_ascii=False) for q in questions) + '\n').encode()


def to_csv(questions: list[Question]) -> bytes:
    output = StringIO()
    fields = list(Question.model_fields)
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for q in questions:
        row = q.model_dump()
        for key, val in row.items():
            if isinstance(val, list):
                row[key] = json.dumps(val, ensure_ascii=False)
            # Guard exported CSV against spreadsheet formula injection.
            elif isinstance(val, str) and val.lstrip().startswith(('=', '+', '-', '@')):
                row[key] = "'" + val
        writer.writerow(row)
    return output.getvalue().encode()


def diagnostics(questions: list[Question]) -> list[dict]:
    return [{'question_id': q.question_id, 'missing_reference': not bool(q.expected_answer),
             'missing_relevance_ids': not bool(q.relevant_chunk_ids or q.relevant_document_ids),
             'pending_review': q.annotation_status != 'verified',
             'split': q.split} for q in questions]
