"""Controlled matrix expansion; other variables held fixed by copy/update."""
from itertools import product
from rag_eval.config import ExperimentConfig
from rag_eval.experiments.runner import plan


def matrix(base: ExperimentConfig, sizes: list[int], overlaps: list[float], retrievers: list[str]) -> list[ExperimentConfig]:
    if len(sizes) * len(overlaps) * len(retrievers) > 100:
        raise ValueError('experiment matrix >100 configurations')
    configs = []
    for size, frac, retriever in product(sizes, overlaps, retrievers):
        if not 0 <= frac < 1:
            raise ValueError('overlaps must be fractions in [0,1)')
        configs.append(base.model_copy(update={'name': f'{base.name} | {size}/{frac:.0%}/{retriever}',
                'chunk_size': size, 'chunk_overlap': round(size * frac), 'retriever': retriever}))
    return [ExperimentConfig.model_validate(c.model_dump()) for c in configs]


def preview(repo, configs: list[ExperimentConfig]) -> list[dict]:
    return [{'config': c.name, 'fingerprint': c.fingerprint[:12], 'size': c.chunk_size,
             'overlap': c.chunk_overlap, 'retriever': c.retriever, **plan(repo, c)} for c in configs]
