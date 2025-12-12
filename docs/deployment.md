# Local and container deployment

From repository root with Python 3.11–3.13 and pip network access:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all]'
python -m pytest -q
export RAGBENCH_HOME=results
export PROJECT6_EVAL_URL=http://127.0.0.1:8000
# Configure external evaluator provider credentials securely before running judge metrics.
streamlit run app/streamlit_app.py --server.address=127.0.0.1
```

For Docker Compose:

```bash
cp .env.example .env
# Set PROJECT6_EVAL_URL, keys, RAGBENCH_ADMIN_TOKEN, RAGBENCH_HOME in .env
# Use a reachable HTTPS Project6 endpoint from inside the container (127.0.0.1 is the container).
docker compose build
docker compose up
```

Compose binds dashboard to `127.0.0.1:8501`. The Project 6 backend is **not** installed into the container. The host must provide it separately. For remote dashboard users, terminate HTTPS and use authenticated access control; the Streamlit token is only a secondary write gate. Docker commands were prepared but **not executed** in the authoring environment.
