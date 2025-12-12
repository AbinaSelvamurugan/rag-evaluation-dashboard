# Troubleshooting

| Symptom | Diagnosis / action |
|---|---|
| No scores | No completed runs, missing judge credentials, or incomplete labels; check individual statuses. |
| `PROJECT6_EVAL_URL` missing | Export real bridge location implementing `ragbench-v1`. |
| Unsupported method / index config | Bridge capability mismatch. Select advertised strategy/units only. |
| Index fingerprint mismatch | Bridge reused incompatible index or returned an older index. Reindex, do not disable check. |
| `ModuleNotFoundError: streamlit` | Install `pip install -e '.[dashboard]'` on network-connected Python. |
| `ModuleNotFoundError: deepeval` | Install `pip install -e '.[judge]'` and provide external judge model credentials. |
| Context precision unavailable | Populate a source-validated `expected_answer`. |
| Retrieval metrics unavailable | Mark answerable labels with verified IDs and `sufficient_evidence=true`. |
| Run incomplete | Inspect question errors and resubmit as new run or resume pending records; errors are deliberately not erased. |
| Container cannot reach localhost backend | `localhost` inside container is not the host; point to a reachable HTTPS host/service. |
| Fixture numbers displayed | They test deterministic matching only; cannot validate Project 6 quality. |
| Deployment unavailable | Browser/Streamlit and live Project 6 tests need their external dependencies and running services. |
