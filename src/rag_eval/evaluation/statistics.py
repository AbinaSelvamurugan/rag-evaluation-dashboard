"""Question-level sampling uncertainty and properly paired comparisons."""
import numpy as np
from rag_eval.config import QuestionResult


def summaries(results: list[QuestionResult]) -> list[dict]:
    names = sorted(set(m for r in results for m in r.metrics))
    rows = []
    for name in names:
        good = [r.metrics[name].score for r in results if name in r.metrics and r.metrics[name].status == 'ok'
                and r.metrics[name].score is not None]
        n = len(good)
        rows.append({'metric': name, 'n': n, 'attempted': sum(name in r.metrics for r in results),
                     'missing_or_failed': sum(name in r.metrics and r.metrics[name].status != 'ok' for r in results),
                     'mean': float(np.mean(good)) if n else None,
                     'std': float(np.std(good, ddof=1)) if n > 1 else None})
    return rows


def paired_delta(baseline: list[QuestionResult], candidate: list[QuestionResult], metric: str, seed: int = 42) -> dict:
    def scores(results):
        return {r.question_id: r.metrics[metric].score for r in results
                if metric in r.metrics and r.metrics[metric].status == 'ok' and r.metrics[metric].score is not None}
    a, b = scores(baseline), scores(candidate)
    keys = sorted(a.keys() & b.keys())
    if not keys:
        return {'n_paired': 0, 'delta': None, 'ci_low': None, 'ci_high': None, 'note': 'no matched valid questions'}
    differences = np.asarray([b[k] - a[k] for k in keys], dtype=float)
    out = {'n_paired': len(keys), 'delta': float(differences.mean()), 'ci_low': None, 'ci_high': None,
           'note': 'Descriptive only: fewer than 5 paired questions; no uncertainty interval'}
    if len(keys) >= 5:
        rng = np.random.default_rng(seed)
        samples = rng.choice(differences, size=(2000, len(keys)), replace=True).mean(axis=1)
        lo, hi = np.percentile(samples, [2.5, 97.5])
        out.update(ci_low=float(lo), ci_high=float(hi), note='95% paired bootstrap CI; exploratory, not a superiority claim')
    return out
