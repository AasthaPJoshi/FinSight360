# FinSight360 — Multi-stage production Dockerfile
# Stage 1: builder installs all deps into /install prefix
# Stage 2: runtime is lean (~60% smaller image)

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install all deps including optional ML packages
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Aastha Joshi <ajoshi8879@sdsu.edu>"
LABEL version="1.0.0"
LABEL description="FinSight360 — Autonomous Financial Anomaly Detection Platform"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY finsight360/ ./finsight360/
COPY ingestion/   ./ingestion/
COPY storage/     ./storage/
COPY config/      ./config/
COPY utils/       ./utils/
COPY ml/          ./ml/
COPY graph/       ./graph/
COPY genai/       ./genai/
COPY governance/  ./governance/
COPY dashboard/   ./dashboard/
COPY mlops/       ./mlops/
COPY dbt/         ./dbt/
COPY scripts/     ./scripts/
COPY data/reference/ ./data/reference/
COPY pyproject.toml .
COPY .streamlit/  ./.streamlit/

# Pre-create runtime directories
RUN mkdir -p data/raw data/processed data/chromadb \
             models mlruns reports/shap_plots docs \
             data/raw/filings logs data/reference

RUN useradd --create-home --shell /bin/bash finsight && \
    chown -R finsight:finsight /app
USER finsight

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DUCKDB_PATH=/app/data/finsight360.duckdb
ENV LOG_FORMAT=json
ENV LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
