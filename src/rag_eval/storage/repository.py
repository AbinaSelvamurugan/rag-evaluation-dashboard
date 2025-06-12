"""Durable, immutable dataset versions and idempotent run records in SQLite."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock
from uuid import uuid4
from rag_eval.config import ExperimentConfig, Question, QuestionResult, utc_now
from rag_eval.datasets.loader import to_jsonl, parse_bytes


class Repository:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.environ.get('RAGBENCH_HOME', 'results')).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'ragbench.sqlite3'
        self._lock = RLock()
        self._init()

    def _conn(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA busy_timeout=30000')
        connection.execute('PRAGMA foreign_keys=ON')
        return connection

    def _init(self):
        with self._conn() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS datasets(
                    version TEXT PRIMARY KEY, created_at TEXT NOT NULL, data BLOB NOT NULL, question_count INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS runs(
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
                    status TEXT NOT NULL, config TEXT NOT NULL, dataset_version TEXT NOT NULL,
                    index_data TEXT, environment TEXT, total INTEGER NOT NULL, error TEXT,
                    cancelled INTEGER DEFAULT 0,
                    FOREIGN KEY(dataset_version) REFERENCES datasets(version));
                CREATE TABLE IF NOT EXISTS question_results(
                    run_id TEXT NOT NULL, question_id TEXT NOT NULL, result TEXT NOT NULL,
                    PRIMARY KEY(run_id,question_id), FOREIGN KEY(run_id) REFERENCES runs(id));
                CREATE TABLE IF NOT EXISTS metric_cache(
                    key TEXT PRIMARY KEY, result TEXT NOT NULL, created_at TEXT NOT NULL);
            ''')

    def put_dataset(self, questions: list[Question]) -> str:
        if not questions or len({q.dataset_version for q in questions}) != 1:
            raise ValueError('dataset must have exactly one content-derived version')
        version = questions[0].dataset_version
        if version == 'pending':
            raise ValueError('dataset must be canonicalized before saving')
        blob = to_jsonl(questions)
        with self._lock, self._conn() as db:
            db.execute('INSERT OR IGNORE INTO datasets VALUES(?,?,?,?)', (version, utc_now(), blob, len(questions)))
            existing = db.execute('SELECT data FROM datasets WHERE version=?', (version,)).fetchone()['data']
            if existing != blob:
                raise ValueError('dataset version collision or attempted mutation')
        return version

    def datasets(self) -> list[dict]:
        with self._conn() as db:
            return [dict(x) for x in db.execute('SELECT version, created_at, question_count FROM datasets ORDER BY created_at DESC')]

    def dataset(self, version: str) -> list[Question]:
        with self._conn() as db:
            found = db.execute('SELECT data FROM datasets WHERE version=?', (version,)).fetchone()
        if not found:
            raise KeyError(f'dataset not found: {version}')
        questions = parse_bytes(bytes(found['data']), 'dataset.jsonl')
        if questions[0].dataset_version != version:
            raise ValueError('stored dataset checksum mismatch')
        return questions

    def create_run(self, config: ExperimentConfig, total: int, environment: dict) -> str:
        if config.dataset_version not in {x['version'] for x in self.datasets()}:
            raise KeyError('dataset version not found')
        run_id = str(uuid4())
        with self._lock, self._conn() as db:
            db.execute('INSERT INTO runs(id,created_at,status,config,dataset_version,total,environment) VALUES(?,?,?,?,?,?,?)',
                       (run_id, utc_now(), 'queued', config.model_dump_json(), config.dataset_version, total, json.dumps(environment)))
        return run_id

    def status(self, run_id: str) -> dict:
        with self._conn() as db:
            row = db.execute('''SELECT r.*, (SELECT COUNT(*) FROM question_results q WHERE q.run_id=r.id) done,
                               (SELECT COUNT(*) FROM question_results q WHERE q.run_id=r.id AND json_extract(q.result,'$.status')='failed') failed
                               FROM runs r WHERE id=?''', (run_id,)).fetchone()
        if not row:
            raise KeyError(f'run not found: {run_id}')
        value = dict(row)
        for key in ('config', 'index_data', 'environment'):
            value[key] = json.loads(value[key]) if value[key] else None
        return value

    def runs(self) -> list[dict]:
        with self._conn() as db:
            ids = [x['id'] for x in db.execute('SELECT id FROM runs ORDER BY created_at DESC')]
        return [self.status(x) for x in ids]

    def update_run(self, run_id: str, *, status: str, error: str | None = None, index_data: dict | None = None):
        if status not in ('queued', 'running', 'completed', 'failed', 'cancelled', 'incomplete'):
            raise ValueError('unknown run status')
        with self._lock, self._conn() as db:
            db.execute('''UPDATE runs SET status=?, error=?, index_data=COALESCE(?,index_data),
                          started_at=CASE WHEN ?='running' THEN COALESCE(started_at,?) ELSE started_at END,
                          finished_at=CASE WHEN ? IN ('completed','failed','cancelled','incomplete') THEN ? ELSE finished_at END
                          WHERE id=?''', (status, error, json.dumps(index_data) if index_data else None,
                                          status, utc_now(), status, utc_now(), run_id))

    def cancel(self, run_id: str):
        with self._lock, self._conn() as db:
            db.execute('UPDATE runs SET cancelled=1 WHERE id=?', (run_id,))

    def save_result(self, run_id: str, result: QuestionResult):
        with self._lock, self._conn() as db:
            db.execute('INSERT OR IGNORE INTO question_results VALUES (?,?,?)',
                       (run_id, result.question_id, result.model_dump_json()))

    def results(self, run_id: str) -> list[QuestionResult]:
        with self._conn() as db:
            rows = db.execute('SELECT result FROM question_results WHERE run_id=? ORDER BY question_id', (run_id,)).fetchall()
        return [QuestionResult.model_validate_json(x['result']) for x in rows]

    def cache_get(self, key: str):
        with self._conn() as db:
            row = db.execute('SELECT result FROM metric_cache WHERE key=?', (key,)).fetchone()
        return json.loads(row['result']) if row else None

    def cache_put(self, key: str, result: dict):
        with self._lock, self._conn() as db:
            db.execute('INSERT OR REPLACE INTO metric_cache VALUES (?,?,?)', (key, json.dumps(result), utc_now()))
