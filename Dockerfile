FROM python:3.13-slim
WORKDIR /opt/ragbench
COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -e '.[dashboard,judge]'
COPY app ./app
COPY configs ./configs
COPY data ./data
COPY docs ./docs
ENV RAGBENCH_HOME=/opt/ragbench/results
RUN mkdir -p /opt/ragbench/results
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
