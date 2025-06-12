"""Command-line entrypoint for reproducible offline and Project 6 runs."""
import argparse
import json
from pathlib import Path
import sys
from rag_eval.config import ExperimentConfig
from rag_eval.datasets.loader import parse_bytes, diagnostics
from rag_eval.evaluation.statistics import summaries, paired_delta
from rag_eval.experiments.planner import matrix, preview
from rag_eval.experiments.runner import start_run, execute_run, plan
from rag_eval.reporting.report import markdown_report, json_export
from rag_eval.storage.repository import Repository


def execute(argv=None):
    p = argparse.ArgumentParser(prog='ragbench', description='RAGBench Studio: evaluate Project 6 (or explicitly labeled software fixture)')
    sub = p.add_subparsers(dest='command', required=True)
    ingest = sub.add_parser('import-dataset', help='validate and version a JSONL or CSV dataset')
    ingest.add_argument('path', type=Path)
    validate = sub.add_parser('validate-dataset')
    validate.add_argument('path', type=Path)
    run = sub.add_parser('run', help='launch one saved experiment from JSON')
    run.add_argument('--config', required=True, type=Path)
    run.add_argument('--execute', action='store_true', help='perform requests; default previews cost only')
    resume = sub.add_parser('resume')
    resume.add_argument('run_id')
    build = sub.add_parser('matrix', help='controlled chunk/retriever sweep')
    build.add_argument('--config', required=True, type=Path)
    build.add_argument('--sizes', required=True, help='e.g. 256,512')
    build.add_argument('--overlaps', required=True, help='fraction e.g. 0,0.1')
    build.add_argument('--retrievers', required=True, help='e.g. dense,keyword')
    build.add_argument('--execute', action='store_true')
    compare = sub.add_parser('compare')
    compare.add_argument('baseline')
    compare.add_argument('candidate')
    export = sub.add_parser('export')
    export.add_argument('run_id')
    export.add_argument('--format', choices=('markdown','json'), default='markdown')
    export.add_argument('--output', required=True, type=Path)
    sub.add_parser('history')
    args = p.parse_args(argv)
    repo = Repository()
    if args.command in ('import-dataset', 'validate-dataset'):
        records = parse_bytes(args.path.read_bytes(), args.path.name)
        print(json.dumps({'version': records[0].dataset_version, 'questions': len(records),
            'missing_reference': sum(x['missing_reference'] for x in diagnostics(records)),
            'missing_relevance_annotations': sum(x['missing_relevance_ids'] for x in diagnostics(records))}, indent=2))
        if args.command == 'import-dataset':
            repo.put_dataset(records)
    elif args.command in ('run', 'matrix'):
        config = ExperimentConfig.model_validate_json(args.config.read_text(encoding='utf-8'))
        configs = ([config] if args.command == 'run' else matrix(config,
            [int(x) for x in args.sizes.split(',')], [float(x) for x in args.overlaps.split(',')],
            [x.strip() for x in args.retrievers.split(',')]))
        print(json.dumps(preview(repo, configs), indent=2))
        if args.execute:
            # Running sequentially limits unexpected concurrent provider loads.
            for c in configs:
                run_id = start_run(repo, c)
                outcome = execute_run(repo, run_id)
                print(json.dumps({'id': run_id, 'status': outcome['status'], 'done': outcome['done'],
                                  'failed': outcome['failed'], 'error': outcome['error']}))
                if outcome['status'] != 'completed':
                    return 2
    elif args.command == 'resume':
        outcome = execute_run(repo, args.run_id)
        print(json.dumps({'status': outcome['status'], 'done': outcome['done'], 'error': outcome['error']}))
        return 0 if outcome['status'] == 'completed' else 2
    elif args.command == 'compare':
        a, b = repo.status(args.baseline), repo.status(args.candidate)
        if a['dataset_version'] != b['dataset_version'] or a['config']['split'] != b['config']['split']:
            print('ERROR: cannot pair different dataset versions/splits', file=sys.stderr)
            return 2
        print(json.dumps({metric: paired_delta(repo.results(args.baseline), repo.results(args.candidate), metric)
            for metric in sorted(set(a['config']['metrics']) & set(b['config']['metrics']))}, indent=2))
    elif args.command == 'export':
        if args.output.exists():
            print('ERROR: output exists; refusing to overwrite', file=sys.stderr)
            return 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        content = markdown_report(repo, args.run_id).encode() if args.format == 'markdown' else json_export(repo, args.run_id)
        args.output.write_bytes(content)
        print(f'exported {len(content)} bytes to {args.output}')
    elif args.command == 'history':
        print(json.dumps([{k: r[k] for k in ('id','status','created_at','done','total','failed','error')}
                          for r in repo.runs()], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(execute())
