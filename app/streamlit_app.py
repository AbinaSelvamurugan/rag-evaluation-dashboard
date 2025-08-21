"""RAGBench Studio eight-view evaluation console (not a chatbot)."""
from __future__ import annotations
from collections import Counter
import hmac
import json
import os
import threading
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from pydantic import ValidationError
from rag_eval.adapters.fixture import make_adapter
from rag_eval.config import ExperimentConfig, Question, digest
from rag_eval.datasets.loader import parse_bytes, to_jsonl, to_csv, diagnostics
from rag_eval.evaluation.statistics import summaries, paired_delta
from rag_eval.experiments.planner import matrix, preview
from rag_eval.experiments.runner import start_run, execute_run, plan
from rag_eval.metrics.registry import METRICS
from rag_eval.reporting.report import json_export, markdown_report
from rag_eval.storage.repository import Repository

st.set_page_config(page_title='RAGBench Studio', page_icon='📊', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container {padding-top: 1.9rem; max-width: 1450px}
div[data-testid="stMetric"] {border: 1px solid rgba(130,130,130,.25); border-radius: 10px; padding: .75rem}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def repo() -> Repository:
    return Repository()

DB = repo()

@st.cache_resource
def execution_gate():
    # One matrix worker at a time in this Streamlit process; each run has bounded per-question concurrency.
    return threading.Semaphore(1)

def run_matrix_jobs(run_ids):
    with execution_gate():
        for rid in run_ids:
            execute_run(DB, rid)


def admin() -> bool:
    expected = os.getenv('RAGBENCH_ADMIN_TOKEN')
    if expected:
        entered = st.sidebar.text_input('Administrator token', type='password', help='Required for writes')
        return hmac.compare_digest(entered, expected)
    return True

CAN_WRITE = admin()
if not os.getenv('RAGBENCH_ADMIN_TOKEN'):
    st.sidebar.warning('Local-only mode: no administrative access token configured. Set RAGBENCH_ADMIN_TOKEN before remote deployment.')

PAGES = ['Overview', 'Evaluation Dataset', 'Run Evaluation', 'Chunking Experiments',
         'Retriever Experiments', 'Detailed Results', 'Experiment History', 'Reports & Export']
selection = st.sidebar.radio('Navigation', PAGES)
st.sidebar.caption('Offline fixture mode is deterministic matching only, not a quality benchmark. Project 6 results come from the live bridge.')
st.sidebar.caption(f'Database: {DB.path.name}')


def labeled_runs():
    return {f'{x["id"][:8]} | {x["status"]} | {x["config"]["name"]}': x for x in DB.runs()}


def pick_run(key: str, completed: bool = False):
    values = labeled_runs()
    if completed:
        values = {k: v for k, v in values.items() if v['status'] == 'completed'}
    if not values:
        st.info('No matching run has been saved. Import a dataset and launch an experiment first.')
        return None
    return values[st.selectbox('Saved run', list(values), key=key)]


def question_df(run_id: str) -> pd.DataFrame:
    rows = []
    questions = {q.question_id: q for q in DB.dataset(DB.status(run_id)['dataset_version'])}
    for item in DB.results(run_id):
        record = questions.get(item.question_id)
        base = {'question_id': item.question_id, 'category': record.category if record else None,
                'status': item.status, 'error': item.error,
                'retrieval_ms': item.trace.retrieval_ms if item.trace else None,
                'total_ms': item.trace.total_ms if item.trace else None,
                'input_tokens': item.trace.input_tokens if item.trace else None,
                'contexts': len(item.trace.passages) if item.trace else None,
                'context_characters': sum(len(x.text) for x in item.trace.passages) if item.trace else None}
        base.update({key: value.score if value.status == 'ok' else None for key, value in item.metrics.items()})
        rows.append(base)
    return pd.DataFrame(rows)


def run_table(runs):
    rows = []
    for r in runs:
        data = {'id': r['id'], 'name': r['config']['name'], 'status': r['status'],
                'adapter': r['config']['adapter'], 'dataset': r['dataset_version'],
                'chunk_size': r['config']['chunk_size'], 'overlap': r['config']['chunk_overlap'],
                'retriever': r['config']['retriever'], 'top_k': r['config']['top_k'],
                'done': r['done'], 'total': r['total'], 'failed': r['failed'], 'created_at': r['created_at'],
                'indexing_ms': (r.get('index_data') or {}).get('indexing_ms'),
                'chunk_count': (r.get('index_data') or {}).get('chunk_count')}
        for s in summaries(DB.results(r['id'])):
            data[s['metric']] = s['mean']
            data[s['metric'] + '_n'] = s['n']
        rows.append(data)
    return pd.DataFrame(rows)


def comparison_plot(data, x, y, color='retriever'):
    if data.empty or y not in data or data[y].notna().sum() == 0:
        st.info(f'No measured values available for {y}.')
        return
    st.plotly_chart(px.scatter(data, x=x, y=y, color=color, hover_data=['id', 'name', 'status'],
                                title=f'{y.replace("_", " ").title()} by {x.replace("_", " ")}'),
                    use_container_width=True)


def evidence(run):
    rows = DB.results(run['id'])
    if not rows:
        st.info('No question-level records saved yet.')
        return
    qs = {q.question_id: q for q in DB.dataset(run['dataset_version'])}
    df = question_df(run['id'])
    st.dataframe(df, use_container_width=True, hide_index=True)
    chosen = st.selectbox('Inspect question', [r.question_id for r in rows], key='trace_pick_' + run['id'])
    result = next(x for x in rows if x.question_id == chosen)
    q = qs[chosen]
    st.markdown(f'**Question:** {q.question}')
    st.markdown(f'**Reference:** {q.expected_answer or "Not annotated"}')
    st.caption(f'Annotation: {q.annotation_status} | Relevant documents: {q.relevant_document_ids} | Relevant chunks: {q.relevant_chunk_ids}')
    if result.error:
        st.error(result.error)
    if result.trace:
        st.markdown('**Actual answer from adapter**')
        st.text(result.trace.answer)
        st.json({'index_id': result.trace.index_id, 'index_fingerprint': result.trace.index_fingerprint,
                 'effective_config': result.trace.effective_config, 'latency_ms': {'retrieval': result.trace.retrieval_ms,
                 'generation': result.trace.generation_ms, 'total': result.trace.total_ms},
                 'token_usage': {'input': result.trace.input_tokens, 'output': result.trace.output_tokens},
                 'citations': result.trace.citations, 'metadata': result.trace.trace_metadata}, expanded=False)
        for p in result.trace.passages:
            with st.expander(f'Rank {p.rank} — {p.document_id} / {p.chunk_id} — score {p.score if p.score is not None else "unknown"}'):
                st.text(p.text)
                st.json({'source': p.source, 'metadata': p.metadata})
    for name, score in result.metrics.items():
        st.markdown(f'**{name}** · {score.score if score.status == "ok" else score.status} · {"PASS" if score.passed else "FAIL" if score.passed is False else "no threshold"}')
        if score.reason or score.error:
            st.caption(score.reason or score.error)
    with st.expander('Configuration snapshot'):
        st.json(run['config'])


def overview():
    st.title('RAGBench Studio')
    st.caption('Evaluation and optimization of retrieval-augmented generation · Project 6 integration boundary')
    runs = DB.runs()
    done = [r for r in runs if r['status'] == 'completed']
    latest = done[0] if done else None
    cols = st.columns(4)
    cols[0].metric('Completed experiments', len(done))
    cols[1].metric('Questions evaluated', sum(r['done'] for r in done))
    cols[2].metric('Failed runs', sum(r['status'] == 'failed' for r in runs))
    cols[3].metric('Incomplete / cancelled', sum(r['status'] in ('incomplete', 'cancelled') for r in runs))
    if not latest:
        st.info('No completed evaluations. Import or load a versioned dataset, configure an integration, and run an experiment. No scores are prepopulated.')
        return
    baseline = st.selectbox('Baseline configuration for comparisons', [x['id'] for x in done],
                             format_func=lambda run_id: next(x['config']['name'] + ' · ' + run_id[:8] for x in done if x['id']==run_id))
    with st.expander('Selected baseline configuration'):
        st.json(next(x['config'] for x in done if x['id']==baseline))
    st.subheader('Latest completed run')
    st.caption(f'{latest["id"]} · {latest["config"]["name"]} · {latest["created_at"]} · adapter: {latest["config"]["adapter"]}')
    if latest['config']['adapter'] == 'fixture':
        st.warning('TEST FIXTURE ONLY — these retrieval numbers are not measurements of Project 6.')
    vals = {s['metric']: s for s in summaries(DB.results(latest['id']))}
    columns = st.columns(4)
    for col, name in zip(columns, ['faithfulness', 'answer_relevance', 'context_precision', 'hit_rate']):
        value = vals.get(name, {})
        col.metric(name.replace('_', ' ').title(), f'{value["mean"]:.3f}' if value.get('mean') is not None else 'N/A',
                   help=f'Valid n={value.get("n", 0)}')
    table = run_table(done)
    if len(table):
        st.dataframe(table, use_container_width=True, hide_index=True)
        present = [x for x in METRICS if x in table and table[x].notna().any()]
        if present:
            metric = st.selectbox('Metric by experiment', present)
            st.plotly_chart(px.bar(table, x='name', y=metric, color='adapter', hover_data=[metric + '_n'],
                                   title=f'{metric} · valid-question counts in hover'), use_container_width=True)


def dataset_page():
    st.title('Evaluation Dataset')
    st.caption('Content-addressed immutable versions · only verified, source-annotated records qualify for retrieval-ground-truth metrics')
    uploaded = st.file_uploader('Import CSV or JSONL (≤5 MiB)', type=['csv','jsonl'])
    if uploaded and st.button('Validate and save dataset', disabled=not CAN_WRITE):
        try:
            questions = parse_bytes(uploaded.getvalue(), uploaded.name)
            version = DB.put_dataset(questions)
            st.success(f'Saved version {version}: {len(questions)} questions')
        except Exception as exc:
            st.error(str(exc))
    if st.button('Install packaged TEST FIXTURE dataset', disabled=not CAN_WRITE):
        try:
            data = (Path(__file__).resolve().parents[1] / 'data' / 'smoke_fixture.jsonl').read_bytes()
            records = parse_bytes(data, 'smoke_fixture.jsonl')
            st.success(f'Installed fixture dataset {DB.put_dataset(records)}')
        except Exception as exc:
            st.error(str(exc))
    datasets = DB.datasets()
    if not datasets:
        st.info('No dataset versions saved.')
        return
    selection = st.selectbox('Dataset version', [v['version'] for v in datasets])
    qs = DB.dataset(selection)
    diag = pd.DataFrame(diagnostics(qs))
    a,b,c = st.columns(3)
    a.metric('Questions', len(qs))
    b.metric('Verified annotations', sum(q.annotation_status == 'verified' for q in qs))
    c.metric('Missing reference answers', int(diag.missing_reference.sum()))
    st.dataframe(diag, use_container_width=True, hide_index=True)
    term = st.text_input('Search question text')
    categories = st.multiselect('Categories', sorted({q.category for q in qs}), default=sorted({q.category for q in qs}))
    shown = [q for q in qs if term.lower() in q.question.lower() and q.category in categories]
    st.dataframe(pd.DataFrame([q.model_dump() for q in shown]), use_container_width=True, hide_index=True)
    st.download_button('Export JSONL', to_jsonl(qs), file_name=f'dataset-{selection}.jsonl', mime='application/x-ndjson')
    st.download_button('Export CSV', to_csv(qs), file_name=f'dataset-{selection}.csv', mime='text/csv')
    st.subheader('Create an annotation revision')
    picked = st.selectbox('Question', [q.question_id for q in qs], key='dataset_question')
    q = next(q for q in qs if q.question_id == picked)
    with st.form('annotation'):
        answer = st.text_area('Reference answer', value=q.expected_answer or '', max_chars=25000)
        documents = st.text_input('Relevant document IDs (comma separated)', value=', '.join(q.relevant_document_ids))
        chunks = st.text_input('Relevant chunk IDs (comma separated)', value=', '.join(q.relevant_chunk_ids))
        status = st.selectbox('Review state', ['unreviewed', 'draft', 'verified'], index=['unreviewed', 'draft', 'verified'].index(q.annotation_status))
        evidence = st.selectbox('Sufficient evidence', ['unknown', 'yes', 'no'], index={None:0,True:1,False:2}[q.sufficient_evidence])
        submit = st.form_submit_button('Save as NEW dataset version', disabled=not CAN_WRITE)
    if submit:
        try:
            updated = q.model_copy(update={'expected_answer': answer or None, 'relevant_document_ids': [x.strip() for x in documents.split(',') if x.strip()],
                    'relevant_chunk_ids': [x.strip() for x in chunks.split(',') if x.strip()], 'annotation_status': status,
                    'sufficient_evidence': {'unknown':None,'yes':True,'no':False}[evidence]})
            items = [Question.model_validate((updated if x.question_id == picked else x).model_dump()) for x in qs]
            changed = parse_bytes(to_jsonl(items), 'revision.jsonl')
            st.success(f'Created dataset version {DB.put_dataset(changed)}')
        except Exception as exc:
            st.error(f'Annotation rejected: {exc}')


def runner_page():
    st.title('Run Evaluation')
    datasets = DB.datasets()
    if not datasets:
        st.info('Import a dataset before defining an experiment.')
        return
    version = st.selectbox('Versioned dataset', [v['version'] for v in datasets])
    mode = st.radio('Integration', ['project6_http', 'fixture'], horizontal=True)
    rerankers = []
    if mode == 'fixture':
        st.warning('OFFLINE FIXTURE: deterministic keyword ranking only, no LLM judge metrics, no Project 6 evaluation.')
        retrievers, strategies, units = ['keyword'], ['fixed'], ['characters']
    else:
        try:
            caps = make_adapter(mode).capabilities()
            retrievers, strategies, units = caps['retrievers'], caps['chunk_strategies'], caps['chunk_units']
            rerankers = caps.get('rerankers', [])
            st.success('Bridge handshake succeeded; this confirms the interface contract only, not the quality of the underlying RAG implementation.')
        except Exception as exc:
            st.error(f'Project 6 handshake unavailable: {exc}')
            retrievers, strategies, units = ['dense'], ['recursive'], ['characters']
    with st.form('configure'):
        name = st.text_input('Experiment name', 'Baseline')
        c1,c2,c3 = st.columns(3)
        strategy = c1.selectbox('Chunk strategy', strategies)
        unit = c2.selectbox('Chunk size unit', units)
        embedding = c3.text_input('Embedding model', 'project6-default')
        c4,c5,c6 = st.columns(3)
        sizes = c4.text_input('Chunk sizes (comma separated)', '512')
        overlaps = c5.text_input('Overlap fractions (comma separated)', '0.1')
        selected_retrievers = c6.multiselect('Retrievers', retrievers, default=retrievers[:1])
        c7,c8,c9 = st.columns(3)
        top_k = c7.number_input('Top-k', min_value=1, max_value=100, value=5)
        generation_model = c8.text_input('Actual generation model', 'project6-default')
        prompt_version = c9.text_input('Prompt version', 'project6-default')
        d1,d2,d3 = st.columns(3)
        context_budget = d1.number_input('Context budget (tokens)', 128, 131072, 4096)
        max_questions = d2.number_input('Maximum questions', 1, 10000, 20)
        concurrency = d3.number_input('Concurrency', 1, 8, 1)
        e1,e2,e3 = st.columns(3)
        timeout = e1.number_input('RAG HTTP timeout (s)', 1, 600, 60)
        retries = e2.number_input('HTTP retries', 0, 5, 2)
        split = e3.selectbox('Split', ['development', 'holdout', 'all'])
        judge_model = st.text_input('DeepEval evaluator model (required for judges, e.g. provider-compatible model)', 'gpt-4o-mini' if mode == 'project6_http' else '', disabled=mode == 'fixture')
        threshold = st.slider('Judge pass threshold', 0.0, 1.0, 0.7)
        reranker_selected = st.selectbox('Reranker (only supported Project 6 options)', ['None'] + rerankers)
        use_similarity = st.checkbox('Apply a similarity threshold (only if supported)')
        similarity = st.slider('Minimum similarity score', 0.0, 1.0, 0.5, disabled=not use_similarity)
        seed = st.number_input('Random seed', min_value=0, max_value=1000000, value=42)
        cache = st.checkbox('Cache exact-trace successful judge computations', value=True)
        default_metrics = list(METRICS) if mode == 'project6_http' else ['hit_rate','precision_at_k','recall_at_k','mrr','ndcg']
        metric_names = st.multiselect('Metrics', list(METRICS), default=default_metrics)
        st.caption('Judges transmit question, answer, and/or retrieved document text to the configured external provider. Cost and tokens are provider-dependent.')
        submit = st.form_submit_button('Preview experiment matrix')
    if submit:
        try:
            size_vals = [int(x.strip()) for x in sizes.split(',') if x.strip()]
            overlap_vals = [float(x.strip()) for x in overlaps.split(',') if x.strip()]
            if not size_vals or not overlap_vals or not selected_retrievers:
                raise ValueError('at least one chunk size, overlap, and retriever required')
            base = ExperimentConfig(name=name, dataset_version=version, adapter=mode, chunk_strategy=strategy,
                chunk_unit=unit, chunk_size=size_vals[0], chunk_overlap=0, embedding_model=embedding,
                retriever=selected_retrievers[0], top_k=top_k, generation_model=generation_model,
                prompt_version=prompt_version, context_budget=context_budget, judge_model=judge_model or None,
                judge_threshold=threshold, max_questions=max_questions, concurrency=concurrency,
                timeout_seconds=timeout, max_retries=retries, split=split, metrics=tuple(metric_names),
                reranker=None if reranker_selected=='None' else reranker_selected,
                similarity_threshold=similarity if use_similarity else None, seed=seed, use_cache=cache)
            configs = matrix(base, size_vals, overlap_vals, selected_retrievers)
            st.session_state['experiment_matrix'] = [x.model_dump() for x in configs]
        except Exception as exc:
            st.error(f'Invalid experiment: {exc}')
    if st.session_state.get('experiment_matrix'):
        configurations = [ExperimentConfig.model_validate(item) for item in st.session_state['experiment_matrix']]
        st.subheader('Execution preview')
        plans = preview(DB, configurations)
        st.dataframe(pd.DataFrame(plans), use_container_width=True, hide_index=True)
        st.caption(f'{len(configurations)} isolated index configurations; {sum(x["rag_requests"] for x in plans)} RAG requests; '
                   f'{sum(x["judge_metric_evaluations"] for x in plans)} judge metric evaluations (each may require multiple LLM calls).')
        total_rag = sum(x['rag_requests'] for x in plans)
        if total_rag > 5000:
            st.error('Matrix exceeds the 5,000-request UI safety ceiling. Reduce configurations or questions per configuration.')
        if st.button('Start saved run(s)', disabled=not CAN_WRITE or total_rag > 5000, type='primary'):
            ids = [start_run(DB, conf) for conf in configurations]
            threading.Thread(target=run_matrix_jobs, args=(ids,), name='ragbench-matrix', daemon=True).start()
            st.session_state['last_started'] = ids
            st.success('Queued runs: ' + ', '.join(x[:8] for x in ids) + '. Runs execute sequentially with a shared concurrency gate.')
    @st.fragment(run_every='2s')
    def monitor():
        recent = DB.runs()[:30]
        if recent:
            st.subheader('Execution status')
            for r in recent[:8]:
                st.write(f'{r["id"][:8]} · {r["config"]["name"]} · **{r["status"]}** · {r["done"]}/{r["total"]}')
                st.progress(min(r['done'] / r['total'], 1.0) if r['total'] else 0)
                if r['error']:
                    st.error(r['error'])
                if r['status'] in ('running', 'queued') and st.button('Cancel', key=f'cancel_{r["id"]}', disabled=not CAN_WRITE):
                    DB.cancel(r['id'])
    monitor()


def experiment_page(kind):
    st.title('Chunking Experiments' if kind == 'chunk' else 'Retriever Experiments')
    done = [x for x in DB.runs() if x['status'] == 'completed']
    if not done:
        st.info('No completed runs available to compare. Incomplete runs are not treated as completed benchmarks.')
        return
    # Compare controlled configurations only: every non-tested setting must match.
    ignored = {'name', 'chunk_size', 'chunk_overlap'} if kind == 'chunk' else {'name', 'retriever'}
    groups = {}
    for r in done:
        fixed = {k: v for k, v in r['config'].items() if k not in ignored}
        key = digest(fixed)[:12]
        groups.setdefault(key, []).append(r)
    ordered_groups = sorted(groups, key=lambda k: -len(groups[k]))
    chosen_group = st.selectbox('Controlled comparison group (all other settings identical)', ordered_groups,
        format_func=lambda key: f'{key} · {len(groups[key])} runs · dataset {groups[key][0]["dataset_version"][:12]}')
    done = groups[chosen_group]
    st.caption('Only the experimental variable (chunk size/overlap or retriever) can differ within this group.')
    data = run_table(done)
    adapters = st.multiselect('Adapter origin', sorted(set(data.adapter)), default=sorted(set(data.adapter)))
    data = data[data.adapter.isin(adapters)]
    if 'fixture' in adapters:
        st.warning('Fixture experiments represent software test behavior, NOT Project 6 quality.')
    if data.empty:
        return
    st.dataframe(data, use_container_width=True, hide_index=True)
    metrics = [m for m in METRICS if m in data.columns and data[m].notna().any()]
    if metrics:
        choice = st.selectbox('Compared metric', metrics)
        comparison_plot(data, 'chunk_size' if kind == 'chunk' else 'retriever', choice)
    if kind == 'chunk':
        if 'chunk_count' in data:
            comparison_plot(data, 'chunk_size', 'chunk_count')
        if 'indexing_ms' in data:
            comparison_plot(data, 'chunk_size', 'indexing_ms')
    else:
        if 'hit_rate' in data:
            comparison_plot(data, 'retriever', 'hit_rate')
    for value in ['retrieval_ms', 'total_ms', 'context_characters', 'input_tokens']:
        merged = []
        for r in done:
            if r['id'] not in set(data.id):
                continue
            df = question_df(r['id'])
            if not df.empty and value in df and df[value].notna().any():
                merged.append({'name': r['config']['name'], 'retriever': r['config']['retriever'],
                               'chunk_size': r['config']['chunk_size'], value: df[value].mean()})
        if merged:
            comparison_plot(pd.DataFrame(merged).assign(id=lambda x: x.name, status='completed'),
                            'chunk_size' if kind == 'chunk' else 'retriever', value)
    choices = [r for r in done if r['id'] in set(data.id)]
    selected = st.selectbox('Inspect underlying question traces', [r['id'] for r in choices])
    selected_run = next(r for r in choices if r['id'] == selected)
    evidence(selected_run)
    st.subheader('Per-question variation and categories')
    question_data = question_df(selected_run['id'])
    numeric_metrics = [x for x in METRICS if x in question_data and question_data[x].notna().any()]
    if numeric_metrics:
        measure = st.selectbox('Distribution metric', numeric_metrics, key='metric_distribution_' + kind)
        st.plotly_chart(px.histogram(question_data, x=measure, nbins=12, title='Measured score distribution'), use_container_width=True)
        st.plotly_chart(px.box(question_data, x='category', y=measure, points='all', title='Question-category breakdown'), use_container_width=True)
    if len(choices) >= 2:
        baseline_id = st.selectbox('Paired comparison baseline', [x['id'] for x in choices], key='baseline_' + kind)
        candidates = [x['id'] for x in choices if x['id'] != baseline_id]
        candidate_id = st.selectbox('Candidate', candidates, key='candidate_' + kind)
        pairs = [{'metric': name, **paired_delta(DB.results(baseline_id), DB.results(candidate_id), name)}
                 for name in sorted(set(DB.status(baseline_id)['config']['metrics']) & set(DB.status(candidate_id)['config']['metrics']))]
        st.dataframe(pd.DataFrame(pairs), use_container_width=True, hide_index=True)
        st.caption('Candidate minus baseline on matched valid questions; exploratory intervals if n ≥ 5.')


def results_page():
    st.title('Detailed Evaluation Results')
    r = pick_run('detail')
    if r:
        if r['config']['adapter'] == 'fixture':
            st.warning('TEST FIXTURE ONLY')
        evidence(r)


def history_page():
    st.title('Experiment History')
    runs = DB.runs()
    if not runs:
        st.info('No history yet.')
        return
    st.dataframe(run_table(runs), use_container_width=True, hide_index=True)
    labels = {f'{r["id"][:8]} · {r["config"]["name"]} · {r["status"]}': r for r in runs}
    selected = st.multiselect('Compare two runs', list(labels), max_selections=2)
    if len(selected) == 2:
        a, b = (labels[x] for x in selected)
        if a['dataset_version'] != b['dataset_version'] or a['config']['split'] != b['config']['split']:
            st.error('Dataset/split differs. Paired comparison blocked.')
        else:
            results = []
            for name in sorted(set(a['config']['metrics']) & set(b['config']['metrics'])):
                results.append({'metric': name, **paired_delta(DB.results(a['id']), DB.results(b['id']), name)})
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            st.caption('Differences use matched question IDs and exclude unavailable metric scores. Confidence interval only when ≥5 paired questions.')
    st.subheader('Top-k sensitivity (observed completed runs)')
    completed = [x for x in runs if x['status'] == 'completed']
    k_groups = {}
    for r in completed:
        frozen = {k: v for k, v in r['config'].items() if k not in ('name', 'top_k')}
        k_groups.setdefault(digest(frozen)[:12], []).append(r)
    eligible = {key: values for key, values in k_groups.items() if len({x['config']['top_k'] for x in values}) >= 2}
    if eligible:
        group = st.selectbox('Constant-settings top-k group', list(eligible))
        top_table = run_table(eligible[group])
        metrics = [m for m in METRICS if m in top_table and top_table[m].notna().any()]
        if metrics:
            metric = st.selectbox('Sensitivity metric', metrics)
            st.plotly_chart(px.line(top_table.sort_values('top_k'), x='top_k', y=metric, markers=True,
                                  hover_data=['id', 'name'], title='Top-k sensitivity under fixed other variables'), use_container_width=True)
    else:
        st.caption('Need at least two completed configurations differing only in top-k.')
    r = pick_run('history_one')
    if r:
        st.json({'configuration': r['config'], 'status': r['status'], 'index': r['index_data'], 'error': r['error']}, expanded=False)


def reports_page():
    st.title('Reports & Export')
    r = pick_run('report')
    if not r:
        return
    baseline = st.selectbox('Paired comparison baseline (optional)', ['None'] + [x['id'] for x in DB.runs() if x['id'] != r['id']])
    report = markdown_report(DB, r['id'], None if baseline == 'None' else baseline)
    st.download_button('Download Markdown report', report, file_name=f'ragbench-{r["id"][:8]}.md', mime='text/markdown')
    st.download_button('Download full trace JSON', json_export(DB, r['id']), file_name=f'ragbench-{r["id"][:8]}.json', mime='application/json')
    flat = question_df(r['id'])
    if not flat.empty:
        safe = flat.copy()
        for col in safe.select_dtypes(include=['object']).columns:
            safe[col] = safe[col].map(lambda x: ("'" + x) if isinstance(x, str) and x.lstrip().startswith(('=', '+', '-', '@')) else x)
        st.download_button('Download question score CSV', safe.to_csv(index=False).encode(), file_name=f'ragbench-{r["id"][:8]}.csv', mime='text/csv')
    st.markdown(report[:30000])


{
    'Overview': overview, 'Evaluation Dataset': dataset_page, 'Run Evaluation': runner_page,
    'Chunking Experiments': lambda: experiment_page('chunk'),
    'Retriever Experiments': lambda: experiment_page('retriever'),
    'Detailed Results': results_page, 'Experiment History': history_page,
    'Reports & Export': reports_page
}[selection]()
