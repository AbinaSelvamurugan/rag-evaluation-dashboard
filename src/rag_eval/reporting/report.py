"""Reports and honest machine-readable exports of persisted artifacts."""
import json
from rag_eval.evaluation.statistics import summaries, paired_delta
from rag_eval.metrics.registry import METRICS
from rag_eval.storage.repository import Repository


def json_export(repo: Repository, run_id: str) -> bytes:
    bundle = {'run': repo.status(run_id), 'question_results': [x.model_dump() for x in repo.results(run_id)]}
    return json.dumps(bundle, indent=2, ensure_ascii=False).encode()


def markdown_report(repo: Repository, run_id: str, baseline: str | None = None) -> str:
    run, results = repo.status(run_id), repo.results(run_id)
    questions = {q.question_id: q for q in repo.dataset(run['dataset_version'])}
    lines = ['# RAGBench Studio evaluation report', '', f'**Run:** `{run_id}`',
             f'**Status:** {run["status"]}', f'**Adapter:** {run["config"]["adapter"]}',
             f'**Dataset version:** `{run["dataset_version"]}`',
             f'**Started:** {run["started_at"]}', f'**Finished:** {run["finished_at"]}',
             f'**Processed:** {run["done"]}/{run["total"]}; question failures: {run["failed"]}', '',
             '## Configuration', '', '```json', json.dumps(run['config'], indent=2), '```', '',
             '## Aggregate measures', '', '| Metric | Mean | Valid n | Failed/unavailable |', '|---|---:|---:|---:|']
    for s in summaries(results):
        lines.append(f'| {s["metric"]} | {s["mean"]:.4f} | {s["n"]} | {s["missing_or_failed"]} |' if s['mean'] is not None else
                     f'| {s["metric"]} | unavailable | 0 | {s["missing_or_failed"]} |')
    lines += ['', '## Metric definitions and limits', '']
    for name in run['config']['metrics']:
        info = METRICS[name]
        lines += [f'- **{name} ({info["kind"]}):** {info["description"]} Needs: {info["needs"]}. Limitations: {info["limits"]}.']
    if baseline:
        br = repo.status(baseline)
        if br['dataset_version'] != run['dataset_version']:
            lines += ['', '## Paired comparison', 'Dataset versions differ; comparison withheld.']
        else:
            lines += ['', '## Paired differences versus baseline', '']
            for name in run['config']['metrics']:
                d = paired_delta(repo.results(baseline), results, name)
                lines += [f'- **{name}:** delta={d["delta"]}, n={d["n_paired"]}, CI=[{d["ci_low"]}, {d["ci_high"]}]. {d["note"]}']
    lines += ['', '## Question-level evidence', '']
    for result in results:
        q = questions.get(result.question_id)
        lines += [f'### {result.question_id}', f'- Question: {q.question if q else "?"}',
                  f'- Reference answer: {q.expected_answer if q else "unavailable"}',
                  f'- Status: {result.status}; error: {result.error or "none"}']
        if result.trace:
            lines += [f'- Answer: {result.trace.answer}',
                      f'- Retrieved passages: {len(result.trace.passages)}',
                      f'- Retrieval ms: {result.trace.retrieval_ms}, generation ms: {result.trace.generation_ms}, total ms: {result.trace.total_ms}',
                      f'- Usage input/output tokens: {result.trace.input_tokens}/{result.trace.output_tokens}']
            for p in result.trace.passages:
                lines += [f'  - Rank {p.rank} doc={p.document_id} chunk={p.chunk_id}: {p.text[:600]}']
        for name, metric in result.metrics.items():
            lines += [f'- {name}: {metric.score if metric.status == "ok" else metric.status}; {metric.reason or metric.error or ""}']
        lines += ['']
    lines += ['## Interpretation and limitations', '',
              '- Missing/failed metric evaluations have no numeric score and are excluded from means.',
              '- Scores depend on evaluator/provider and ground-truth quality. No automatic overall composite is calculated.',
              '- Results from fixture mode are software tests, not measurements of Project 6.',
              '- Comparisons with a tuned development set do not prove performance on a reserved holdout.',
              '- Recommendations require paired evaluation and quality/latency/cost trade-off review.']
    return '\n'.join(lines) + '\n'
