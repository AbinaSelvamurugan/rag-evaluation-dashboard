import ast
from pathlib import Path


def test_streamlit_dashboard_source_parses_and_routes():
    path=Path(__file__).resolve().parents[1] / 'app' / 'streamlit_app.py'
    source=path.read_text()
    ast.parse(source)
    for page in ('Overview','Evaluation Dataset','Run Evaluation','Chunking Experiments',
                 'Retriever Experiments','Detailed Results','Experiment History','Reports & Export'):
        assert page in source
    assert 'st.fragment(run_every=' in source
    assert 'repo.status(' not in source or 'DB.status(' in source


def test_no_hardcoded_dashboard_metric_scores():
    source=(Path(__file__).resolve().parents[1] / 'app' / 'streamlit_app.py').read_text()
    assert 'mean' in source and 'DB.results(' in source
    assert 'hardcoded_score' not in source
