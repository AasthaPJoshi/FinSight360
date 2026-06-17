# FinSight360 — Developer Makefile
# Usage: make <target>
.DEFAULT_GOAL := help
PYTHON ?= python3
VENV   ?= .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

.PHONY: help install venv seed run run-docker test lint clean reset docker-up docker-down

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

venv:  ## Create Python virtual environment
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv  ## Install all dependencies
	$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"

.env:
	cp .env.example .env
	@echo "✓ .env created — edit it to add your API keys"

dirs:
	mkdir -p data/raw data/processed data/chromadb models mlruns \
	         reports/shap_plots logs data/raw/filings

seed: install .env dirs  ## Seed 10-company demo dataset into DuckDB
	DUCKDB_PATH=data/finsight360.duckdb $(PY) scripts/seed_demo_data.py
	@echo "✓ Demo data seeded"

run: seed  ## Seed + launch Streamlit dashboard (quickstart)
	DUCKDB_PATH=data/finsight360.duckdb \
	$(VENV)/bin/streamlit run dashboard/app.py \
	  --server.port 8501 --server.headless false

run-no-seed:  ## Launch dashboard without re-seeding
	DUCKDB_PATH=data/finsight360.duckdb \
	$(VENV)/bin/streamlit run dashboard/app.py \
	  --server.port 8501 --server.headless false

ingest:  ## Ingest AAPL from SEC EDGAR
	DUCKDB_PATH=data/finsight360.duckdb $(PY) -m finsight360.cli ingest --ticker AAPL

ml-run:  ## Run ML anomaly detection pipeline
	DUCKDB_PATH=data/finsight360.duckdb $(PY) -m finsight360.cli ml-run

test:  ## Run unit tests (no integration tests)
	DUCKDB_PATH=data/test.duckdb $(VENV)/bin/pytest tests/ \
	  -v --tb=short -m "not integration" --timeout=60

test-all:  ## Run ALL tests including integration (needs Docker)
	$(VENV)/bin/pytest tests/ -v --tb=short --timeout=120

lint:  ## Lint with ruff
	$(VENV)/bin/ruff check . --select E,F,I --ignore E501

docker-up:  ## Start all Docker services (dashboard + Neo4j + MLflow)
	docker-compose up -d

docker-down:  ## Stop all Docker services
	docker-compose down

docker-logs:  ## Tail dashboard logs
	docker-compose logs -f finsight-dashboard

clean:  ## Clean pyc / cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache

reset:  ## DANGER: Delete DuckDB and re-seed from scratch
	rm -f data/finsight360.duckdb
	$(MAKE) seed
	@echo "✓ Database reset complete"
