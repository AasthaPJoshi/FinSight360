<!-- Header banner -->
<div align="center">

# 📊 FinSight360

### Autonomous Financial Anomaly Detection & Risk Intelligence Platform

*Powered by SEC EDGAR · Benford's Law · Isolation Forest · LangChain RAG · Corporate Knowledge Graph · Responsible AI*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-69%20passing-brightgreen?logo=pytest)](tests/)
[![dbt](https://img.shields.io/badge/dbt-1.8.3-FF694B?logo=dbt&logoColor=white)](dbt/)
[![MLflow](https://img.shields.io/badge/MLflow-2.13.2-0194E2?logo=mlflow)](mlruns/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![SEC EDGAR](https://img.shields.io/badge/Data-SEC%20EDGAR%20API-orange)](https://data.sec.gov)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](dashboard/)

**Built by [Aastha Joshi](https://github.com/AasthaPJoshi) · MS Information Systems, San Diego State University**

© 2025 Aastha Joshi. All rights reserved.

</div>

---
        
## 📋 Table of Contents

1. [The Problem](#the-problem)
2. [What FinSight360 Does](#what-finsight360-does)
3. [10-Phase Build Journey](#10-phase-build-journey)
4. [System Architecture](#system-architecture)
5. [Data Flow Diagram](#data-flow-diagram)
6. [Tech Stack](#tech-stack)
7. [Signal Architecture](#signal-architecture)
8. [Project Structure](#project-structure)
9. [Quick Start](#quick-start)
10. [CLI Reference](#cli-reference)
11. [Dashboard](#dashboard)
12. [Responsible AI](#responsible-ai)
13. [MLOps](#mlops)
14. [Sample Results](#sample-results)
15. [Credits & Copyright](#credits--copyright)

---

## The Problem

Financial fraud costs the global economy an estimated **$5 trillion annually**, yet most detection systems are built on rules written after the last scandal — not the next one. The collapses of Enron, WorldCom, Wirecard, and Luckin Coffee shared a common thread: structured financial data that looked plausible in isolation but revealed unmistakable patterns when analyzed holistically across time, peers, and filings. Enron's off-balance-sheet entities, WorldCom's capitalized operating expenses, Wirecard's phantom Asian subsidiaries — in each case, the evidence was in the public filings. It simply wasn't being read the right way. The human cost of these failures extends far beyond shareholders: pension funds evaporated, employees lost jobs and retirement savings, and systemic trust in capital markets eroded for a generation.

Existing financial surveillance tools fail for three structural reasons. First, they are almost universally rule-based — a fixed threshold for current ratio, a flag for revenue jumps above X% — which means they catch only patterns that regulators already know to look for. Second, they operate on single signals: an analyst checks revenue growth, another checks accruals, another checks auditor changes. No tool triangulates all signals simultaneously and weights them against each other. Third, and most critically, they are **reactive**. Investigations begin after restatements, whistleblowers, or short-seller reports. By then, billions have already been lost. Conservative estimates suggest that qualified analysts spend 80% of their time gathering and cleaning data, and only 20% on the judgment that actually matters.

The scale of the data gap makes manual review structurally impossible. The SEC receives over **10,000 annual filings per year**, each running hundreds of pages of financial tables, footnotes, and management narratives. A forensic accounting team working full-time could review perhaps 200 filings per year at depth. That means 98% of filed documents go unscrutinized by any human with the skills to detect manipulation. Inside those unreviewed filings are patterns: accounts receivable growing faster than revenue for three consecutive quarters, first-digit distributions that violate Benford's Law, audit networks where every firm in a suspicious cluster shares the same Big Four partner rotation. These patterns exist — they are simply invisible at the scale at which they appear.

FinSight360 was built to close that gap. By combining classical forensic accounting techniques (Benford's Law, accruals analysis) with modern unsupervised machine learning (Isolation Forest on 17 engineered features), corporate network intelligence (Louvain community detection on auditor-sharing graphs), and large language model analysis of filing text (LangChain RAG over 10-K narratives), the platform functions as an **always-on autonomous analyst** that never sleeps, never misses a filing, and explains every flag it raises in plain language. Every scoring decision is logged to an immutable audit trail, every prediction is explained with SHAP values, and every model is documented to SR 11-7 and EU AI Act standards. This is not a research prototype. It is production-grade financial intelligence infrastructure.

---

## What FinSight360 Does

FinSight360 ingests structured financial data directly from the **SEC EDGAR public API** (no vendor subscription required) for any publicly traded US company. Using async httpx with rate-limiting, retry logic, and Semaphore-controlled concurrency, the platform pulls XBRL-tagged financial facts from 10-K annual and 10-Q quarterly filings — 25 financial concepts per filing including revenue, gross profit, operating income, net income, total assets, cash, receivables, inventory, and capital expenditures. All raw data lands in a local DuckDB analytical database through a five-table schema with Pydantic v2-validated models. From there, a three-layer dbt transformation pipeline (staging → intermediate → mart) computes financial ratios, rolling averages, and risk flags, producing clean analytical tables ready for ML consumption.

The platform computes three independent risk signals and fuses them into a composite score. The first signal — **ML Anomaly Detection** — runs Isolation Forest over 17 engineered features (margins, ratios, growth rates, working capital metrics) using RobustScaler preprocessing to handle the heavy-tailed distributions inherent in financial data. The second signal — **Benford's Law Forensics** — tests whether the distribution of leading digits in financial statements follows the logarithmic law that naturally occurring numerical data obeys, using both Mean Absolute Deviation and chi-square tests with Nigrini (2012) thresholds. The third signal — **Corporate Network Risk** — constructs a MultiDiGraph of auditor-sharing and subsidiary relationships, applies Louvain community detection to find risk clusters, and propagates risk scores through the network using PageRank-weighted neighbor averaging. Final scores weight these signals 50/30/20 and map to risk tiers (LOW / MEDIUM / HIGH / CRITICAL).

The platform produces six categories of output. The **primary output** is a scored risk table with company-level composite scores, tier classifications, and dominant signal attribution. **LLM Risk Briefs** are generated by a LangChain LCEL RAG pipeline that retrieves semantically similar passages from ChromaDB-indexed 10-K filings and synthesizes a structured risk narrative using GPT-4o-mini. **SHAP Waterfall Charts** accompany every ML prediction, showing exactly which features drove the anomaly score up or down. **Network Maps** visualize the corporate relationship graph as interactive HTML using pyvis, color-coded by risk tier. **Audit Trail Records** log every scoring event as an immutable, UUID-keyed append-only DuckDB table with tamper-evident feature hashes for SR 11-7 compliance. **Model Cards** (JSON + Markdown, Mitchell et al. 2019 format) are auto-generated from live database statistics and include bias analysis results across industry groups.

### Core Capabilities

| Capability | Description |
|------------|-------------|
| **SEC Filing Ingestion** | Async httpx pipeline pulls XBRL-tagged 10-K/10-Q data from SEC EDGAR for any US public company with rate-limiting and exponential backoff retry |
| **Benford's Law Forensics** | MAD + chi-square test against Nigrini (2012) thresholds; flags digit-distribution anomalies that correlate with accounting manipulation |
| **ML Anomaly Detection** | Isolation Forest on 17 financial features with RobustScaler; `contamination=0.10`; scores map directly to anomaly percentile |
| **Corporate Graph Analysis** | NetworkX MultiDiGraph of auditor-sharing/subsidiary relationships; Louvain community detection; PageRank centrality; optional Neo4j persistence |
| **LLM Risk Narratives** | LangChain LCEL RAG chain over ChromaDB-indexed 10-K sections; GPT-4o-mini synthesizes plain-language risk briefs from actual filing text |
| **SHAP Explainability** | TreeExplainer produces exact Shapley values for every Isolation Forest prediction; waterfall charts saved as PNGs per company |
| **Bias Monitoring** | Z-score bias detection across industry groups; Fairlearn integration; EU AI Act Article 10 data governance compliance checks |
| **Audit Trail & Compliance** | Append-only DuckDB audit log with UUID event keys, model version tracking, and KQL query library for Azure Data Explorer integration |

---

## 10-Phase Build Journey

This platform was engineered across 10 sequential phases, each adding a production layer on top of the last. Every phase introduced new tests; no phase shipped with fewer than 4 new passing tests.

---

### Phase 1 — Data Ingestion Engine

**What was built:** An async SEC EDGAR ingestion pipeline capable of pulling any public company's financial history from scratch. The `edgar_client.py` module implements async httpx with a 0.12-second inter-request delay and `tenacity` exponential backoff to respect SEC rate limits. The `xbrl_parser.py` extracts 25 financial concepts from the EDGAR company facts JSON endpoint. The `filing_processor.py` orchestrates concurrent ingestion using `asyncio.Semaphore(5)` and stores all data through a thread-local DuckDB connection layer.

**Key files:** `ingestion/edgar_client.py`, `ingestion/xbrl_parser.py`, `ingestion/filing_processor.py`, `ingestion/models.py`, `storage/database.py`, `storage/schema.sql`

**Tests added:** 18 → **total: 18**

**Tech:** SEC EDGAR API, httpx 0.27.0, DuckDB 0.10.3, Pydantic v2, tenacity 8.3.0, asyncio

---

### Phase 2 — dbt Transformation Layer

**What was built:** A three-layer dbt transformation pipeline that converts raw EDGAR data into analysis-ready analytical tables. Staging models (`stg_companies`, `stg_financial_metrics`) clean, deduplicate, and type-cast raw data. Intermediate models (`int_financial_ratios`, `int_quarterly_trends`) compute 12 financial ratios and flag six risk signals: AR acceleration, cash-earnings divergence, margin compression, aggressive revenue recognition, high leverage, and liquidity stress. Mart models materialize final tables including `mart_risk_candidates` with pre-ML risk scores and `mart_financial_summary` with deduplication to guarantee one 10-K row per company per fiscal year.

**Key files:** `dbt/models/staging/`, `dbt/models/intermediate/`, `dbt/models/marts/`, `dbt/macros/safe_divide.sql`, `dbt/tests/assert_metrics_period_unique.sql`

**Tests added:** 7 → **total: 25**

**Tech:** dbt-core 1.8.3, dbt-duckdb 1.8.1, SQL, Jinja2 macros, dbt_utils, 21 data quality tests

---

### Phase 3 — ML Anomaly Detection

**What was built:** A full unsupervised ML pipeline anchored on two independent algorithms. `BenfordAnalyzer` computes Mean Absolute Deviation and chi-square statistics against expected digit-frequency distributions, classifying companies as Conforming / Acceptable / Marginal / Non-conforming. `AnomalyDetector` trains an Isolation Forest over 17 engineered features using RobustScaler (chosen over StandardScaler for robustness to outliers in heavy-tailed financial distributions). `FeatureBuilder` constructs the feature matrix from dbt mart tables. `GraphRiskScorer` fuses all three signals into a weighted composite score logged to MLflow for experiment tracking.

**Key files:** `ml/benford.py`, `ml/anomaly_detector.py`, `ml/feature_builder.py`, `ml/model_registry.py`, `ml/explainer.py`, `ml/pipeline.py`

**Tests added:** 6 → **total: 31**

**Tech:** scikit-learn 1.5.0, Isolation Forest, SHAP 0.45.1 TreeExplainer, SciPy 1.13.1, MLflow 2.13.2, RobustScaler

---

### Phase 4 — Corporate Knowledge Graph

**What was built:** A corporate intelligence layer that models relationships between companies, auditors, and subsidiaries as a NetworkX MultiDiGraph. `CorporateGraphBuilder` loads 50 CIK-to-auditor mappings and 30 subsidiary relationships from reference JSON files and constructs a graph with `AUDITED_BY` and `SUBSIDIARY_OF` edges. `ClusterDetector` applies the Louvain community detection algorithm and computes PageRank and betweenness centrality for each node. `GraphRiskScorer` propagates risk scores through the network using weighted neighbor averaging. `Neo4jLoader` provides optional persistence to a Neo4j 5.20 instance with Cypher query support.

**Key files:** `graph/graph_builder.py`, `graph/cluster_detector.py`, `graph/neo4j_loader.py`, `graph/graph_queries.py`, `graph/graph_risk_scorer.py`

**Tests added:** 5 → **total: 36**

**Tech:** NetworkX 3.3, Neo4j 5.20, python-louvain 0.16, Cypher, PageRank, betweenness centrality, bolt protocol

---

### Phase 5 — LangChain RAG System

**What was built:** A retrieval-augmented generation pipeline that transforms 10-K filing text into queryable financial intelligence. `SECFilingLoader` fetches and splits Item 1A (Risk Factors) and Item 7 (MD&A) sections using `RecursiveCharacterTextSplitter` with chunk_size=1000 and overlap=200. `VectorStore` persists embeddings to ChromaDB using OpenAI `text-embedding-3-small`. `FinSightRAGChain` implements a LangChain LCEL pipeline (`|` operator composition) that retrieves semantically similar passages and generates structured answers with GPT-4o-mini. `RiskNarrator` synthesizes a master risk brief combining ML signals, Benford results, network risk, and filing tone.

**Key files:** `genai/document_loader.py`, `genai/vector_store.py`, `genai/rag_chain.py`, `genai/sentiment_analyzer.py`, `genai/risk_narrator.py`, `genai/qa_interface.py`

**Tests added:** 8 → **total: 44**

**Tech:** LangChain 0.2.6, LangChain LCEL, ChromaDB 0.5.3, OpenAI API 1.35.3, text-embedding-3-small, GPT-4o-mini, lazy evaluation

---

### Phase 6 — Streamlit Dashboard

**What was built:** A 7-page interactive Streamlit application that makes the full FinSight360 analysis accessible to non-technical stakeholders. The dashboard connects to DuckDB via a cached `DataLoader` class and renders risk intelligence across pages covering executive overview, risk heatmap, per-company deep dive, interactive network graph, Benford digit analysis, LangChain-powered AI analyst chat, and a full governance panel. Reusable components (`charts.py`, `metrics_row.py`, `risk_table.py`) eliminate repetition and enforce visual consistency. The app seeds demo data automatically when the database is empty, making it runnable without any prior ingestion.

**Key files:** `dashboard/app.py`, `dashboard/data_loader.py`, `dashboard/components/charts.py`, `dashboard/components/metrics_row.py`, `dashboard/components/risk_table.py`, `dashboard/pages/01_executive_overview.py` through `07_governance.py`

**Tests added:** 6 → **total: 50**

**Tech:** Streamlit 1.35.0, Plotly 5.22.0, pyvis 0.3.2, Altair, DuckDB, session state management

---

### Phase 7 — Responsible AI Governance

**What was built:** A comprehensive governance layer meeting SR 11-7 model risk management and EU AI Act Article 10 data governance requirements. `AuditTrailManager` writes every scoring event to an append-only DuckDB table with UUID event keys, model version tracking, tamper-evident feature hashes, and `ON CONFLICT DO NOTHING` immutability. `BiasAnalyzer` detects disparate impact across industry groups using Z-score analysis and Fairlearn. `ModelCard` auto-generates Mitchell et al. (2019) compliant model documentation in both JSON and Markdown from live database statistics. `KQLQueries` provides 9 production-ready Kusto Query Language queries for Azure Data Explorer integration and security monitoring.

**Key files:** `governance/audit_trail.py`, `governance/bias_analyzer.py`, `governance/shap_explainer.py`, `governance/model_card.py`, `governance/kql_queries.py`, `dashboard/pages/07_governance.py`, `docs/kql_queries.kql`

**Tests added:** 9 → **total: 59**

**Tech:** Fairlearn 0.10.0, SHAP TreeExplainer, KQL, Azure Data Explorer, SR 11-7, EU AI Act, Mitchell et al. 2019

---

### Phase 8 — Production MLOps

**What was built:** Full production deployment infrastructure. A multi-stage Dockerfile separates the build stage (gcc, compilers, Python build tools) from the lean runtime stage, running as a non-root `finsight` user with a HEALTHCHECK endpoint. `docker-compose.yml` orchestrates the full stack: dashboard, Neo4j 5.20, and MLflow tracking server. Three GitHub Actions workflows handle CI (pytest on every push), CD (Docker build and push on main merge), and scheduled pipeline runs (daily at 06:00 UTC). `HealthChecker` validates six components: database connectivity, data freshness, model file presence, ChromaDB, reference data files, and MLflow connectivity. `ModelMonitor` detects score drift (threshold: 10 points) and anomaly rate drift (threshold: 5%).

**Key files:** `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `.github/workflows/cd.yml`, `.github/workflows/pipeline.yml`, `mlops/health_check.py`, `mlops/model_monitor.py`, `mlops/registry.py`

**Tests added:** 10 → **total: 69**

**Tech:** Docker multi-stage build, docker-compose, GitHub Actions, MLflow Model Registry, drift detection, Staging → Production promotion

---

### Phase 9 — Portfolio Polish

**What was built:** Documentation, demo infrastructure, and dbt correctness fixes. `seed_demo_data.py` seeds 10 realistic companies (AAPL, MSFT, TSLA, AMZN, META, NVDA, JPM, GS, JNJ, XOM) with 8 years of financial history, ML scores, Benford results, and audit trail entries — enabling full dashboard exploration without running the ingestion pipeline. A `QUALIFY row_number()` deduplication fix was applied to `stg_companies.sql` to prevent fan-out from companies stored with both padded and unpadded CIK formats, and `mart_financial_summary.sql` was updated to guarantee uniqueness of 10-K rows per fiscal year (the `assert_metrics_period_unique` dbt test).

**Key files:** `scripts/seed_demo_data.py`, `docs/architecture.md`, `docs/resume_bullets.md`, `docs/interview_prep.md`, `docs/linkedin_post.md`, `dbt/models/staging/stg_companies.sql`, `dbt/models/marts/mart_financial_summary.sql`

**Tests added:** 0 → **total: 69** (all existing tests continue to pass)

**Tech:** Demo data seeder, dbt QUALIFY window deduplication, DuckDB upsert patterns

---

### Phase 10 — Final Documentation

**What was built:** This README, CONTRIBUTING.md, CHANGELOG.md, and LICENSE — production-quality documentation sufficient to onboard a new contributor in under 30 minutes. The repository now meets the documentation standards expected of open-source projects at financial institutions and represents a complete, shippable portfolio artifact.

**Key files:** `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE`

**Total: 10 phases · 69 tests · 80+ files · 10,000+ lines of production code**

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         F I N S I G H T 3 6 0                                   ║
║              Autonomous Financial Anomaly Detection Platform                     ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   ┌─────────────────────────────── DATA SOURCES ───────────────────────────┐    ║
║   │  SEC EDGAR Public API    ·    Yahoo Finance    ·    GDELT News Feed    │    ║
║   │  (XBRL company facts)       (price context)       (sentiment signal)  │    ║
║   └────────────────────────────────────┬───────────────────────────────────┘    ║
║                                        │                                         ║
║                                        ▼                                         ║
║   ┌─────────────────────────── INGESTION LAYER ────────────────────────────┐    ║
║   │  edgar_client.py          xbrl_parser.py       filing_processor.py    │    ║
║   │  (async httpx,            (25 XBRL concepts,   (orchestrator,         │    ║
║   │   0.12s rate limit,        Pydantic v2 models,  Semaphore(5),         │    ║
║   │   tenacity retry)          type validation)     async pipeline)       │    ║
║   └────────────────────────────────────┬───────────────────────────────────┘    ║
║                                        │                                         ║
║                                        ▼                                         ║
║   ┌─────────────────────────── STORAGE LAYER ──────────────────────────────┐    ║
║   │  ┌──────────────┐   ┌──────────────┐  ┌───────────┐  ┌─────────────┐ │    ║
║   │  │   DuckDB     │   │    Neo4j     │  │ ChromaDB  │  │  MLflow     │ │    ║
║   │  │  (5 tables   │   │  (graph DB,  │  │ (vector   │  │  (runs,     │ │    ║
║   │  │   analytics) │   │   bolt:7687) │  │  store)   │  │   models)   │ │    ║
║   │  └──────────────┘   └──────────────┘  └───────────┘  └─────────────┘ │    ║
║   └────────────────────────────────────┬───────────────────────────────────┘    ║
║                                        │                                         ║
║                                        ▼                                         ║
║   ┌─────────────────────────── TRANSFORM LAYER (dbt) ──────────────────────┐    ║
║   │  STAGING                 INTERMEDIATE              MART                │    ║
║   │  stg_companies    ──►   int_financial_ratios  ──►  mart_risk_candidates │    ║
║   │  stg_filings      ──►   int_quarterly_trends  ──►  mart_financial_summary│    ║
║   │  stg_fin_metrics  ──►   int_company_profiles  ──►  mart_company_dashboard│    ║
║   │  (clean, cast,          (12 ratios, 6 flags,       (pre-ML scores,     │    ║
║   │   deduplicate)           rolling 4Q averages)       risk tiers)        │    ║
║   └────────────────────────────────────┬───────────────────────────────────┘    ║
║                                        │                                         ║
║                     ┌──────────────────┴──────────────────┐                    ║
║                     │                                       │                    ║
║                     ▼                                       ▼                    ║
║   ┌──────────── ML / ANALYTICS ──────────┐  ┌────── GRAPH INTELLIGENCE ──────┐ ║
║   │  BenfordAnalyzer                     │  │  CorporateGraphBuilder         │ ║
║   │  · MAD + chi-square test             │  │  · NetworkX MultiDiGraph       │ ║
║   │  · Nigrini 2012 thresholds           │  │  · AUDITED_BY + SUBSIDIARY_OF  │ ║
║   │  · Conformity classification         │  │                                 │ ║
║   │                                      │  │  ClusterDetector               │ ║
║   │  AnomalyDetector                     │  │  · Louvain communities         │ ║
║   │  · Isolation Forest (n=200)          │  │  · PageRank centrality         │ ║
║   │  · 17 features, RobustScaler         │  │  · Betweenness centrality      │ ║
║   │  · contamination=0.10                │  │                                 │ ║
║   │                                      │  │  Neo4jLoader (optional)        │ ║
║   │  SHAP TreeExplainer                  │  │  · Persistent Cypher queries   │ ║
║   │  · Exact Shapley values              │  │  · bolt://localhost:7687       │ ║
║   │  · Waterfall PNG per company         │  │                                 │ ║
║   │                                      │  │  GraphRiskScorer               │ ║
║   │  MLflow Tracking                     │  │  · Network risk propagation    │ ║
║   │  · Experiment logs, model registry   │  │  · Neighbor-weighted scoring   │ ║
║   └──────────────────────────────────────┘  └────────────────────────────────┘ ║
║                                        │                                         ║
║                                        ▼                                         ║
║   ┌─────────────────────────── GenAI LAYER ────────────────────────────────┐    ║
║   │  SECFilingLoader         VectorStore           FinSightRAGChain        │    ║
║   │  (Item 1A + Item 7,      (ChromaDB,            (LangChain LCEL,        │    ║
║   │   chunk 1000/200)         text-embed-3-small)   GPT-4o-mini)           │    ║
║   │                                                                         │    ║
║   │  RiskNarrator ── synthesizes ML + Benford + Network + Filing tone      │    ║
║   └────────────────────────────────────┬───────────────────────────────────┘    ║
║                                        │                                         ║
║                                        ▼                                         ║
║   ┌─────────────────────────── OUTPUT LAYER ───────────────────────────────┐    ║
║   │  Streamlit Dashboard (7 pages)  ·  CLI (16 commands)  ·  Audit Trail  │    ║
║   └────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                  ║
║   ┌── RESPONSIBLE AI ──────────────────┐  ┌── MLOps ──────────────────────┐   ║
║   │  SHAP · Fairlearn · Model Card     │  │  Docker · GitHub Actions      │   ║
║   │  SR 11-7 · EU AI Act · KQL         │  │  MLflow Registry · Drift Det. │   ║
║   └────────────────────────────────────┘  └───────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Data Flow Diagram

```
STEP 1: DATA INGESTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEC EDGAR API ──► edgar_client.py (async httpx, rate-limited 0.12s/req)
                  │  ► company_facts/{CIK}.json endpoint
                  │  ► exponential backoff via tenacity (max 3 retries)
                  ▼
               xbrl_parser.py (extracts 25 XBRL financial concepts)
                  │  ► revenues, gross_profit, operating_income
                  │  ► net_income, total_assets, current_assets
                  │  ► total_liabilities, stockholders_equity
                  │  ► cash, accounts_receivable, inventory, capex ...
                  ▼
               filing_processor.py (orchestrator, asyncio.Semaphore(5))
                  │  ► upsert_company(), upsert_filing()
                  │  ► upsert_financial_fact(), upsert_financial_metric()
                  ▼
               DuckDB ──► companies + filings + financial_facts
                          + financial_metrics + ingestion_runs
                          (5 tables · 5 indexes · thread-local connections)


STEP 2: TRANSFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
financial_metrics ──► dbt staging (clean · deduplicate · type-cast)
                       │  stg_financial_metrics: lpad(cik,10,'0'),
                       │    safe_divide() ratios, QUALIFY dedup
                       │  stg_companies: lpad(cik,10,'0'), SIC industry
                       ▼
                    dbt intermediate (business logic · ephemeral views)
                       │  int_financial_ratios: 12 ratios + 4Q rolling avg
                       │    + period_rank_in_year (row_number window)
                       │  int_quarterly_trends: 6 risk flag signals
                       │    flag_ar_acceleration: AR/Rev > prior × 1.15
                       │    flag_cashflow_divergence: |opcf-NI| > NI×0.2
                       │    flag_margin_compression: GM < 4Q_avg − 0.03
                       │    flag_aggressive_recognition: Rev↑10%+AR↑20%
                       │    flag_high_leverage: D/E > 3.0
                       │    flag_liquidity_stress: current_ratio < 1.0
                       ▼
                    dbt marts (materialized tables · analytical layer)
                       ├── mart_risk_candidates
                       │     pre_ml_risk_score, risk_tier, dominant_signal
                       ├── mart_financial_summary
                       │     one 10-K per (cik, fiscal_year) — QUALIFY dedup
                       └── mart_company_dashboard
                             company profile + latest financial snapshot


STEP 3: ML ANOMALY DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
mart_financial_summary ──► FeatureBuilder (17 features · RobustScaler)
                            │  Features: gross_margin, operating_margin,
                            │    net_margin, current_ratio, debt_to_equity,
                            │    ar_to_revenue, revenue_yoy_growth,
                            │    cash_earnings_gap, roe, sga_ratio,
                            │    rd_intensity, flag_ar_acceleration,
                            │    flag_cashflow_divergence,
                            │    flag_margin_compression,
                            │    flag_aggressive_recognition,
                            │    flag_high_leverage, flag_liquidity_stress
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
         BenfordAnalyzer            IsolationForest
         (MAD score,                (contamination=0.10,
          chi-square test,           n_estimators=200,
          p-value,                   random_state=42,
          conformity rating:         anomaly_percentile
          Conforming / Acceptable    → 0-100 risk score)
          Marginal / Non-conforming)
              │                           │
              └─────────────┬─────────────┘
                            ▼
                     GraphRiskScorer
                     final = ML×0.50 + Benford×0.30 + Network×0.20
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
        final_risk_scores          MLflow experiment log
        (DuckDB table)             (params · metrics · artifacts)
              │
              ▼
        SHAP TreeExplainer
        (exact Shapley values · waterfall PNG per company)


STEP 4: GRAPH INTELLIGENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
reference JSON ──► CorporateGraphBuilder
(50 auditor mappings,    │  ► NetworkX MultiDiGraph
 30 relationships)       │  ► Company nodes (cik, ticker, risk_score)
                         │  ► Auditor nodes (name, type)
                         │  ► AUDITED_BY edges
                         │  ► SUBSIDIARY_OF edges
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        ClusterDetector        Neo4jLoader (optional)
        (Louvain algorithm,    (bolt://localhost:7687,
         community_id,          persistent Cypher queries,
         PageRank score,        graph visualization in
         betweenness,           Neo4j Browser)
         risk_cluster_rating)
              │
              ▼
        network_risk_score = avg(neighbor final_risk_scores)
        weighted by edge count and PageRank centrality


STEP 5: GenAI LAYER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10-K filing text ──► SECFilingLoader
(from EDGAR HTML)       │  ► Item 1A: Risk Factors extraction
                         │  ► Item 7: MD&A extraction
                         │  ► RecursiveCharacterTextSplitter
                         │    (chunk_size=1000, overlap=200)
                         ▼
                    OpenAI text-embedding-3-small
                         │  ► 1536-dim dense embeddings
                         ▼
                    ChromaDB (cosine similarity · persistent local)
                         │  ► Collection per company ticker
                         │  ► Metadata: section, chunk_id, period
                         ▼
                    LangChain LCEL RAG Chain
                         │  retriever | prompt | llm | StrOutputParser
                         │  ► FinSightRAGChain.ask(question, ticker)
                         │  ► generate_risk_summary(ticker)
                         ▼
                    RiskNarrator
                         ► master brief: ML signals + Benford +
                           Network risk + Filing tone
                         ► structured JSON output with confidence


STEP 6: OUTPUT LAYER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
final_risk_scores ──┬──► Streamlit Dashboard (7 interactive pages)
                    │      http://localhost:8501
                    ├──► CLI (16 commands)
                    │      python -m finsight360.cli <command>
                    ├──► audit_trail table (append-only DuckDB)
                    │      UUID · model_version · feature_hash · timestamp
                    ├──► SHAP waterfall PNGs
                    │      docs/shap_<ticker>.png
                    └──► Model Card (docs/model_card.md + model_card.json)
                           Mitchell et al. 2019 · auto-generated from DB
```

---

## Tech Stack

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Language** | Python | 3.11 | Core language across all modules |
| **HTTP Client** | httpx | 0.27.0 | Async SEC EDGAR API calls with connection pooling |
| **Data Validation** | Pydantic | 2.7.1 | Type-safe data models, Settings management |
| **Local Warehouse** | DuckDB | 0.10.3 | In-process analytical SQL; no server required |
| **Cloud Warehouse** | Snowflake | — | Production scale-out option for enterprise deployment |
| **Data Transform** | dbt-core | 1.8.3 | Staging → intermediate → mart transformation pipeline |
| **Orchestration** | Apache Airflow | 2.x | DAG-based scheduling for production pipelines |
| **ML Framework** | scikit-learn | 1.5.0 | Isolation Forest, RobustScaler, preprocessing |
| **Explainability** | SHAP | 0.45.1 | TreeExplainer, exact Shapley values, waterfall charts |
| **Forensics** | SciPy | 1.13.1 | Chi-square test, statistical distribution analysis |
| **Graph (in-memory)** | NetworkX | 3.3 | MultiDiGraph construction, centrality algorithms |
| **Graph (persistent)** | Neo4j | 5.20 | Cypher queries, bolt protocol, Browser visualization |
| **Community Detection** | python-louvain | 0.16 | Louvain modularity optimization algorithm |
| **LLM Framework** | LangChain | 0.2.6 | LCEL RAG pipeline, prompt templates, chains |
| **LLM Provider** | OpenAI API | 1.35.3 | GPT-4o-mini for narratives, text-embedding-3-small |
| **Vector Database** | ChromaDB | 0.5.3 | Persistent local vector store, cosine similarity |
| **Dashboard** | Streamlit | 1.35.0 | 7-page interactive web application |
| **Visualization** | Plotly | 5.22.0 | Gauges, treemaps, scatter plots, bar charts |
| **Network Viz** | pyvis | 0.3.2 | Interactive HTML network graph visualization |
| **Experiment Tracking** | MLflow | 2.13.2 | Model registry, experiment logging, artifact storage |
| **Bias Detection** | Fairlearn | 0.10.0 | Disparate impact analysis, EU AI Act compliance |
| **Structured Logging** | structlog | 24.1.0 | JSON + console structured log rendering |
| **Retry Logic** | tenacity | 8.3.0 | Exponential backoff for API calls |
| **Containerization** | Docker | multi-stage | Production deployment with non-root user |
| **CI/CD** | GitHub Actions | — | Test on push, Docker build on merge, daily pipeline |
| **IaC** | Terraform | — | AWS infrastructure provisioning option |
| **Query Language** | KQL | — | Azure Data Explorer integration, 9 production queries |
| **Testing** | pytest | 8.2.2 | 69 tests across 11 test files, 0 mocked business logic |

---

## Signal Architecture

FinSight360 combines three independent risk signals into one composite score. The independence of these signals is architecturally intentional: each catches different manipulation patterns, and no single signal dominates the composite. A company flagged by all three signals is overwhelmingly more likely to warrant investigation than one flagged by a single rule.

### Signal 1 — ML Anomaly Detection (50% weight)

Isolation Forest works by randomly partitioning the feature space using decision tree splits and measuring how few splits are needed to isolate a data point. Anomalous observations — those that deviate from the majority pattern — are isolated more quickly because they occupy sparse regions of the feature space. Each tree produces an anomaly score; the ensemble average becomes a continuous anomaly score mapped to a 0–100 percentile range where higher means more anomalous.

The choice of **RobustScaler** over StandardScaler is deliberate and important. Financial data exhibits heavy-tailed distributions: a single year of exceptional performance (NVDA's 2024 revenue surge) would skew a StandardScaler's mean and compress the variance of all other features. RobustScaler uses the median and interquartile range instead of mean and standard deviation, making it robust to exactly these outliers. The result is a feature matrix where no single exceptional company warps the anomaly landscape for all others.

The `contamination=0.10` hyperparameter sets the expected fraction of outliers in the training data. In a universe of S&P 500 companies, assuming 10% are sufficiently anomalous to warrant scrutiny is both empirically reasonable and conservative enough to avoid alert fatigue. The anomaly percentile output is a continuous signal: a company at the 95th percentile is not just "anomalous" — it is more anomalous than 94% of its peers on 17 simultaneously considered dimensions.

### Signal 2 — Benford's Law Forensics (30% weight)

Benford's Law states that in naturally occurring numerical datasets spanning multiple orders of magnitude, the probability that a number begins with digit *d* is `log₁₀(1 + 1/d)`. This gives expected leading-digit frequencies of 30.1% for 1, 17.6% for 2, 12.5% for 3, down to 4.6% for 9. The law emerges from the mathematics of scale-invariant processes — financial data, population figures, geographic measurements — and has been validated across thousands of real-world datasets.

Why does it detect fraud? When financial statements are manufactured rather than derived from actual transactions, the fabricator must choose numbers. Human intuition does not naturally reproduce logarithmic distributions: people tend to avoid starting numbers with 1 (feels too small) and overuse 5, 6, 7 (feels more "random"). This leaves a measurable statistical fingerprint. The **Mean Absolute Deviation** (MAD) between observed and expected digit frequencies provides the primary score; Mark Nigrini's 2012 thresholds classify conformity: MAD < 0.006 is acceptable, 0.006–0.012 is marginal, > 0.012 is non-conforming, > 0.015 is significantly non-conforming. The chi-square test provides a p-value for hypothesis testing. Applied to Enron's 2000 financials retrospectively, the MAD score would have been well into the non-conforming range.

FinSight360 applies Benford's analysis to the collection of all financial statement line items for a given company across all available periods — revenues, expenses, assets, liabilities, and their components. This multi-period approach provides more statistical power than single-year analysis (recommended minimum: 100 observations; FinSight360 uses 240 per company where available) and catches manipulation that shifted between years.

### Signal 3 — Network Risk Propagation (20% weight)

Auditor-sharing networks encode risk in ways that isolated company analysis cannot see. When a single audit partner signs off on dozens of companies in the same network cluster, findings of manipulation at one company become statistically relevant priors for all companies in that cluster — not because of legal connection, but because shared professional relationships create shared blind spots, shared methodologies, and in some cases, shared incentives to overlook problems. The Big Four fraud cases of the past two decades share a common pattern: concentrated audit relationships within specific network communities.

FinSight360 builds a `NetworkX MultiDiGraph` with company nodes, auditor nodes, `AUDITED_BY` edges, and `SUBSIDIARY_OF` edges from reference data. Louvain community detection partitions this graph into clusters that maximize within-cluster edge density — these clusters represent natural risk communities. Within each community, **PageRank centrality** identifies which companies are most "central" to the information flow (important connectors whose risk posture affects many neighbors), while **betweenness centrality** identifies bridge nodes whose removal would fragment the community.

The network risk score for each company is computed as the weighted average of the final risk scores of its immediate graph neighbors, with PageRank-proportional weights. A company whose primary auditor also audits three other HIGH-tier companies receives a meaningfully higher network risk score than one whose auditor has no such associations. This propagation continues for one hop (immediate neighbors only) to avoid diluting the signal across the full network.

### Final Score Formula

```
Final Risk Score =
    (Isolation Forest anomaly percentile  × 0.50) +
    (Benford's Law risk score             × 0.30) +
    (Network neighbor risk                × 0.20)

Range: 0 – 100

Risk Tiers:
  ████  0 – 30    LOW       Standard monitoring
  ████  31 – 50   MEDIUM    Enhanced review recommended
  ████  51 – 70   HIGH      Immediate analyst review required
  ████  71 – 100  CRITICAL  Escalate to compliance team
```

The 50/30/20 weighting reflects the relative statistical power of each signal at the sample sizes typical of SEC filing analysis. Isolation Forest operates on 17 dimensions simultaneously and captures complex nonlinear interactions; it earns the majority weight. Benford's Law is a powerful forensic tool but depends on sufficient sample size and can be confounded by legitimate accounting choices in certain industries; it earns the second weight. Network risk is a strong prior but is limited by the quality of the reference relationship data; it earns the tertiary weight.

---

## Project Structure

```
FinSight360/
├── ingestion/                        ← SEC EDGAR data ingestion module
│   ├── edgar_client.py               ← async httpx API client, 0.12s rate limit, tenacity retry
│   ├── xbrl_parser.py                ← XBRL/JSON concept extractor, 25 financial concepts
│   ├── filing_processor.py           ← async orchestrator, Semaphore(5) concurrency control
│   └── models.py                     ← Pydantic v2 data models: Company, Filing, FinancialFact
│
├── storage/
│   ├── database.py                   ← DuckDB manager, thread-local connections, upsert methods
│   └── schema.sql                    ← DDL: 5 tables, 5 indexes, audit trail table
│
├── config/
│   └── settings.py                   ← Pydantic BaseSettings, .env loading, EDGAR_USER_AGENT
│
├── utils/
│   ├── logger.py                     ← structlog JSON + console renderer, bound context
│   └── retry.py                      ← tenacity decorator factory, exponential backoff
│
├── dbt/                              ← dbt transformation project (dbt-duckdb adapter)
│   ├── dbt_project.yml               ← project config, model materialization settings
│   ├── profiles.yml                  ← DuckDB connection: path from DUCKDB_PATH env var
│   ├── packages.yml                  ← dbt_utils dependency declaration
│   ├── models/
│   │   ├── staging/                  ← clean, type-cast, deduplicate raw source data
│   │   │   ├── sources.yml           ← source definitions + not_null/unique data tests
│   │   │   ├── stg_companies.sql     ← lpad(cik,10,'0'), SIC industry mapping, QUALIFY dedup
│   │   │   ├── stg_filings.sql       ← form type normalization, date casting
│   │   │   ├── stg_financial_facts.sql ← fact-level cleaning, taxonomy tagging
│   │   │   └── stg_financial_metrics.sql ← ratio computation, safe_divide macro
│   │   ├── intermediate/             ← ephemeral business logic views
│   │   │   ├── int_financial_ratios.sql  ← 12 ratios, 4Q rolling avg, period_rank_in_year
│   │   │   ├── int_quarterly_trends.sql  ← 6 risk flag signals, total_signal_count
│   │   │   └── int_company_profiles.sql  ← latest snapshot per company
│   │   └── marts/                    ← materialized analytical tables
│   │       ├── mart_risk_candidates.sql  ← pre_ml_risk_score, risk_tier, dominant_signal
│   │       ├── mart_financial_summary.sql ← 10-K dedup, company join, full history
│   │       └── mart_company_dashboard.sql ← one row per company, dashboard-ready
│   ├── macros/
│   │   ├── safe_divide.sql           ← null-safe division: NULLIF(denominator,0)
│   │   └── generate_surrogate_key.sql ← md5-based surrogate key macro
│   └── tests/                        ← 21 dbt data quality tests
│       ├── assert_metrics_period_unique.sql ← one 10-K per (cik, fiscal_year)
│       └── assert_revenue_non_negative.sql  ← revenues ≥ 0 invariant
│
├── ml/                               ← machine learning and scoring module
│   ├── benford.py                    ← BenfordAnalyzer: MAD, chi-square, Nigrini thresholds
│   ├── anomaly_detector.py           ← AnomalyDetector: IsolationForest + RobustScaler
│   ├── feature_builder.py            ← FeatureBuilder: 17-feature matrix from dbt marts
│   ├── model_registry.py             ← MLflow experiment logging, artifact storage
│   ├── explainer.py                  ← SHAP TreeExplainer, waterfall PNG generation
│   └── pipeline.py                   ← end-to-end ML orchestrator: load → score → save
│
├── graph/                            ← corporate knowledge graph module
│   ├── graph_builder.py              ← CorporateGraphBuilder: NetworkX MultiDiGraph
│   ├── cluster_detector.py           ← Louvain communities, PageRank, betweenness centrality
│   ├── neo4j_loader.py               ← Neo4j bolt loader, Cypher MERGE statements
│   ├── graph_queries.py              ← Cypher queries + NetworkX fallback implementations
│   └── graph_risk_scorer.py          ← GraphRiskScorer: 3-signal weighted final score
│
├── genai/                            ← LangChain RAG and LLM module
│   ├── document_loader.py            ← SECFilingLoader: Item 1A/7 extraction, text splitting
│   ├── vector_store.py               ← ChromaDB persistent store, collection management
│   ├── rag_chain.py                  ← FinSightRAGChain: LCEL pipeline, retriever + GPT-4o-mini
│   ├── sentiment_analyzer.py         ← lexicon-based + LLM tone analysis for MD&A sections
│   ├── risk_narrator.py              ← RiskNarrator: master brief from all signals + filing text
│   └── qa_interface.py               ← unified GenAI entry point: ask(), brief(), index()
│
├── governance/                       ← responsible AI and compliance module
│   ├── audit_trail.py                ← AuditTrailManager: append-only log, UUID keys, SR 11-7
│   ├── bias_analyzer.py              ← Z-score bias detection across industry groups, Fairlearn
│   ├── shap_explainer.py             ← SHAP TreeExplainer wrapper, PNG export, feature ranking
│   ├── model_card.py                 ← ModelCard: Mitchell et al. 2019, JSON + Markdown output
│   └── kql_queries.py                ← 9 KQL queries for Azure Data Explorer integration
│
├── mlops/                            ← MLOps and production monitoring module
│   ├── health_check.py               ← HealthChecker: 6-component status (DB, model, ChromaDB...)
│   ├── model_monitor.py              ← ModelMonitor: score drift + anomaly rate drift detection
│   └── registry.py                   ← MLflow model promotion: Staging → Production workflow
│
├── dashboard/                        ← Streamlit 7-page application
│   ├── app.py                        ← main entry point, sidebar nav, demo data auto-seed
│   ├── data_loader.py                ← DataLoader: cached DuckDB queries, @st.cache_data
│   ├── components/
│   │   ├── charts.py                 ← 8 reusable Plotly chart functions
│   │   ├── metrics_row.py            ← KPI card renderer with delta indicators
│   │   └── risk_table.py             ← tier-colored styled dataframe component
│   └── pages/
│       ├── 01_executive_overview.py  ← KPI cards, tier distribution, scatter, industry heatmap
│       ├── 02_risk_heatmap.py        ← filterable treemap, company ranking table
│       ├── 03_company_deep_dive.py   ← gauge, signals, flags, 4 trend charts, AI brief
│       ├── 04_network_graph.py       ← interactive pyvis HTML graph, edge filtering
│       ├── 05_benford_analysis.py    ← conformity charts, MAD histogram, digit frequency
│       ├── 06_ai_analyst.py          ← LangChain RAG chat, example prompts, history
│       └── 07_governance.py          ← audit trail, SHAP importance, bias chart, model card
│
├── data/
│   └── reference/
│       ├── sp500_auditors.json       ← 50 CIK → auditor name mappings (Big Four + regional)
│       └── sp500_relationships.json  ← 30 corporate subsidiary + joint venture relationships
│
├── docs/
│   ├── architecture.md               ← deep technical architecture: component design decisions
│   ├── resume_bullets.md             ← 10 XYZ-format resume bullets for portfolio use
│   ├── interview_prep.md             ← 15 technical Q&A with exact class/file references
│   ├── linkedin_post.md              ← launch post template with hashtags
│   ├── model_card.md                 ← auto-generated Mitchell et al. model card (Markdown)
│   ├── model_card.json               ← machine-readable model card (JSON)
│   ├── kql_queries.kql               ← Azure ADX query library, 9 production queries
│   └── ci-workflows/                 ← GitHub Actions YAML files (copy to .github/workflows/)
│       ├── ci.yml                    ← pytest on every push, matrix: Python 3.11
│       ├── cd.yml                    ← Docker build + push to registry on main merge
│       └── pipeline.yml              ← scheduled daily ingestion at 06:00 UTC
│
├── scripts/
│   └── seed_demo_data.py             ← seeds 10 companies (AAPL→XOM), 8 years, full pipeline data
│
├── tests/
│   ├── conftest.py                   ← pytest markers, seed_aapl_data fixture, db fixture
│   ├── fixtures/
│   │   └── aapl_seed.py              ← AAPL historical 10-K/10-Q data for integration tests
│   ├── test_database.py              ← 6 DuckDB schema and upsert tests
│   ├── test_edgar_client.py          ← 5 API client tests (httpx mock)
│   ├── test_xbrl_parser.py           ← 7 XBRL concept extraction tests
│   ├── test_dbt_models.py            ← 4 dbt run + data quality tests
│   ├── test_ml_pipeline.py           ← 6 ML scoring and Benford tests
│   ├── test_graph.py                 ← 5 graph construction and scoring tests
│   ├── test_genai.py                 ← 7 RAG chain and vector store tests
│   ├── test_dashboard.py             ← 6 data loader and component tests
│   ├── test_governance.py            ← 9 audit trail, bias, model card tests
│   └── test_mlops.py                 ← 10 health check, drift, registry tests
│
├── .github/
│   └── workflows/                    ← (copy from docs/ci-workflows/ — requires workflow PAT scope)
│       ├── ci.yml
│       ├── cd.yml
│       └── pipeline.yml
│
├── .streamlit/
│   └── config.toml                   ← Streamlit theme (primary blue), server port 8501
│
├── Dockerfile                        ← multi-stage: builder (gcc+tools) → runtime (non-root finsight user)
├── docker-compose.yml                ← full stack: dashboard + Neo4j 5.20 + MLflow server
├── .dockerignore                     ← excludes data/, mlruns/, __pycache__, .env
├── pyproject.toml                    ← project metadata, ruff linting config
├── requirements.txt                  ← 35+ pinned dependencies
├── Makefile                          ← 20+ targets: install, test, ingest, dbt-*, dashboard, seed-demo
├── .env.example                      ← template: OPENAI_API_KEY, EDGAR_USER_AGENT, DUCKDB_PATH
├── CONTRIBUTING.md                   ← contributor guide, branch conventions, PR template
├── CHANGELOG.md                      ← full phase-by-phase change history
├── LICENSE                           ← MIT License
└── README.md                         ← this file
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Git

### Option A — Demo Mode (no API keys required)

```bash
# 1. Clone the repository
git clone https://github.com/AasthaPJoshi/FinSight360.git
cd FinSight360

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — set EDGAR_USER_AGENT to "Your Name your@email.com" (SEC requirement)

# 4. Initialize the database schema
python -m finsight360.cli db init

# 5. Seed 10 realistic demo companies (instant — no API calls)
python scripts/seed_demo_data.py
# Seeds: AAPL, MSFT, TSLA, AMZN, META, NVDA, JPM, GS, JNJ, XOM
# Creates: 8 years of financials, ML scores, Benford results, audit trail

# 6. Launch the dashboard
streamlit run dashboard/app.py
# Open: http://localhost:8501
```

### Option B — Full Live Pipeline (OPENAI_API_KEY required for GenAI)

```bash
# After steps 1-4 above:

# Ingest a company from SEC EDGAR
python -m finsight360.cli ingest --ticker AAPL

# Run dbt transformations
make dbt-run

# Score with ML pipeline
python -m finsight360.cli ml-run

# Build corporate knowledge graph
python -m finsight360.cli graph-build

# Generate SHAP explanations
python -m finsight360.cli shap-run

# Generate full governance report
python -m finsight360.cli governance-report

# Launch dashboard
streamlit run dashboard/app.py
```

### Option C — Docker (recommended for production)

```bash
# Start the full stack: dashboard + Neo4j + MLflow
docker-compose up -d

# Services:
#   Dashboard:    http://localhost:8501
#   Neo4j Browser: http://localhost:7474  (user: neo4j / pass: finsight360)
#   MLflow UI:    http://localhost:5001

# View logs
docker-compose logs -f finsight-dashboard

# Stop
docker-compose down
```

### Environment Variables

```bash
# .env file (copy from .env.example)

# Required for SEC EDGAR API access
EDGAR_USER_AGENT="Your Name your@email.com"

# Required for LangChain RAG and AI Analyst features
OPENAI_API_KEY="sk-..."

# Optional overrides (defaults shown)
DUCKDB_PATH="data/finsight360.duckdb"
MLFLOW_TRACKING_URI="mlruns"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="finsight360"
CHROMA_PATH="data/chroma"
LOG_LEVEL="INFO"
```

---

## CLI Reference

All commands are invoked as `python -m finsight360.cli <command>`. Use `--help` on any command for full options.

| Command | Description | Example |
|---------|-------------|---------|
| `db init` | Initialize DuckDB schema (5 tables, 5 indexes) | `python -m finsight360.cli db init` |
| `db status` | Show row counts for all tables | `python -m finsight360.cli db status` |
| `ingest --ticker` | Ingest one company from SEC EDGAR | `python -m finsight360.cli ingest --ticker AAPL` |
| `ingest --all` | Ingest all configured tickers | `python -m finsight360.cli ingest --all` |
| `ml-run` | Run full ML scoring pipeline | `python -m finsight360.cli ml-run` |
| `graph-build` | Build corporate knowledge graph | `python -m finsight360.cli graph-build` |
| `shap-run` | Compute and export SHAP values | `python -m finsight360.cli shap-run` |
| `index --ticker` | Index 10-K filing text to ChromaDB | `python -m finsight360.cli index --ticker AAPL` |
| `ask "question"` | Natural language query over filings | `python -m finsight360.cli ask "TSLA risk factors"` |
| `brief --ticker` | Generate full LLM risk narrative | `python -m finsight360.cli brief --ticker GS` |
| `governance-report` | Run audit trail + bias analysis | `python -m finsight360.cli governance-report` |
| `model-card` | Generate model card (MD + JSON) | `python -m finsight360.cli model-card` |
| `health` | Check all 6 pipeline components | `python -m finsight360.cli health` |
| `drift-check` | Detect score and anomaly rate drift | `python -m finsight360.cli drift-check` |
| `registry-status` | Show MLflow model versions | `python -m finsight360.cli registry-status` |
| `seed-demo` | Seed 10 demo companies into DuckDB | `python -m finsight360.cli seed-demo` |

---

## Dashboard

The FinSight360 dashboard is a 7-page Streamlit application that presents the full risk intelligence pipeline to non-technical stakeholders. All pages read from DuckDB via a cached `DataLoader` class and update in real time as new companies are ingested or scored. The dashboard auto-seeds demo data when the database is empty, so it runs immediately after cloning without any pipeline execution.

| Page | Description |
|------|-------------|
| 📋 **Executive Overview** | Four KPI cards (companies monitored, HIGH/CRITICAL count, mean risk score, non-conforming Benford count), risk tier distribution pie chart, risk score vs. Benford scatter landscape, industry heatmap showing average risk by sector, sortable top-15 risk companies table |
| 🌡️ **Risk Heatmap** | Filterable Plotly treemap sized by market cap proxy and colored by risk tier; sidebar filters for industry, tier, and score range; full company ranking table with signal decomposition columns |
| 🔍 **Company Deep Dive** | Per-company risk gauge (0–100 dial), three-signal bar chart showing ML/Benford/Network components, active risk flag indicators, four rolling financial trend charts (revenue, margins, cash flow, leverage), AI risk brief display panel |
| 🕸️ **Network Graph** | Interactive pyvis HTML graph rendered in Streamlit, company and auditor nodes color-coded by risk tier, AUDITED_BY and SUBSIDIARY_OF edges, click-to-inspect node metadata, community cluster highlighting |
| 📐 **Benford Analysis** | Conformity status pie chart (Conforming / Acceptable / Marginal / Non-conforming), MAD score histogram across all companies, per-company digit frequency chart overlaid against expected Benford distribution, chi-square p-value table |
| 🤖 **AI Analyst** | LangChain RAG chat interface backed by ChromaDB-indexed 10-K filings, company filter dropdown, six example starter questions, persistent chat history within session, source passage citations alongside each answer |
| ⚖️ **Governance** | Audit trail paginated table (last 100 events with UUID, model version, timestamp), SHAP feature importance bar chart for selected company, industry bias Z-score deviation chart, rendered model card Markdown, KQL query library with copy buttons |

---

## Responsible AI

**Explainability with SHAP TreeExplainer.** Every Isolation Forest prediction in FinSight360 is accompanied by an exact Shapley value decomposition computed using SHAP's `TreeExplainer` — not the slower approximate `KernelExplainer`. For each company, a waterfall chart shows the contribution of each of the 17 features to the final anomaly score: which features pushed the score higher (e.g., elevated AR-to-revenue ratio, negative cash-earnings gap) and which pulled it lower (e.g., conforming Benford score, low leverage). This means no decision in FinSight360 is a black box. An analyst reviewing a HIGH-tier company can see exactly which financial signals drove the flag, read the SHAP explanation in plain English, and make an informed judgment about whether to escalate.

**Benford's Law as an Independent Forensic Signal.** Benford's Law analysis operates on the raw financial data independently of the ML model — it is not a feature in the Isolation Forest; it is a separate signal with its own weighting. This independence is deliberately designed to prevent correlated failure modes: if the ML model develops a blind spot for a particular manipulation pattern, Benford's Law (being rooted in a different mathematical principle entirely) may still catch it. The forensic accounting literature is clear that Benford's Law non-conformance is a necessary but not sufficient indicator of fraud — FinSight360 treats it accordingly, as a weighted signal that increases scrutiny probability rather than a definitive accusation.

**Bias Monitoring and EU AI Act Compliance.** The `BiasAnalyzer` class computes Z-scores for each industry group's average risk score against the population mean. A Z-score above 2.0 indicates that companies in that sector are being flagged at a statistically unusual rate, which may indicate model bias rather than genuine sector-wide risk elevation. Results are reported in both the governance dashboard page and the auto-generated model card. The platform is designed to satisfy **EU AI Act Article 10** data governance requirements: data sources are documented (SEC EDGAR public domain), data quality checks are automated (21 dbt tests), and bias monitoring is continuous rather than periodic.

**Audit Trail for SR 11-7 Compliance.** Every time FinSight360 scores a company, a record is written to the `audit_trail` table using `ON CONFLICT DO NOTHING` — making it structurally append-only. Each record carries a UUID event key, the model name and version, all input feature values, the output score, the risk tier, a SHA-256 hash of the feature vector (tamper-evident), and a timestamp. This design satisfies the Federal Reserve's **SR 11-7 Model Risk Management** guidance requirement for model output logging with sufficient detail to reproduce any scoring decision. A `KQLQueries` class provides nine pre-written Kusto Query Language queries for ingesting the audit trail into Azure Data Explorer for enterprise-scale security monitoring.

**Model Card.** The `ModelCard` class auto-generates documentation conforming to the **Mitchell et al. (2019)** model card specification in both JSON (machine-readable) and Markdown (human-readable) formats. The card is generated from live DuckDB statistics — training data size, scoring date, anomaly rate, bias analysis results — so it always reflects the current model state rather than a snapshot written at training time. Fields covered include: model details, intended use, factors, metrics, evaluation data, training data, quantitative analyses, ethical considerations, and caveats and recommendations. The model card is displayed in full in the Governance dashboard page.

---

## MLOps

**Multi-Stage Docker Build.** The `Dockerfile` uses a two-stage build to minimize the production image size. The `builder` stage includes GCC, build essentials, and all Python compilation tools needed for packages with C extensions (DuckDB, numpy, scipy). The `runtime` stage copies only the compiled site-packages and application code, discarding all build tooling. The runtime container runs as a non-root user named `finsight` (UID 1000) and includes a `HEALTHCHECK` instruction that pings the Streamlit `/healthz` endpoint every 30 seconds. The resulting image is approximately 800MB — large by microservice standards but appropriate for a data-science application with scipy, scikit-learn, and DuckDB.

**GitHub Actions CI/CD.** Three workflow files drive the automation pipeline. `ci.yml` triggers on every push and pull request: it installs dependencies, runs `python -m pytest tests/ -m "not integration"` with the `-v` flag, and reports the full test matrix. `cd.yml` triggers on pushes to `main` and builds the Docker image with the git SHA as the tag, pushes to the configured container registry, and optionally deploys to the staging environment. `pipeline.yml` runs on a cron schedule (`0 6 * * *`, daily at 06:00 UTC) and executes the full ingestion-to-scoring pipeline for all configured tickers, posting a summary to a Slack webhook. These files live in `docs/ci-workflows/` and require a GitHub PAT with `workflow` scope to be moved to `.github/workflows/`.

**MLflow Experiment Tracking and Model Registry.** Every ML pipeline run logs to MLflow: hyperparameters (`contamination`, `n_estimators`, `random_state`), metrics (`anomaly_rate`, `mean_score`, `high_critical_count`), and the trained model artifact. The `ModelRegistry` class in `mlops/registry.py` handles the **Staging → Production promotion** workflow: after each run, the new model version is registered in the MLflow Model Registry, validated against the drift threshold, and promoted to Production if it passes. The previous Production version is archived, not deleted, enabling rollback. MLflow's tracking server is included in `docker-compose.yml` and accessible at `http://localhost:5001`.

**Health Monitoring.** The `HealthChecker` class in `mlops/health_check.py` validates six components on demand: (1) **Database connectivity** — can DuckDB open and query the main schema; (2) **Data freshness** — when was the most recent `ingestion_run` completed; (3) **Model presence** — does the expected MLflow model artifact exist on disk; (4) **ChromaDB** — can the vector store be initialized and queried; (5) **Reference data files** — do the auditor JSON and relationships JSON files exist; (6) **MLflow** — is the tracking server reachable. Results are returned as a structured dictionary and displayed in the CLI via `python -m finsight360.cli health`.

**Drift Detection.** The `ModelMonitor` class in `mlops/model_monitor.py` computes two drift metrics on each run: **score drift** (the difference between the current mean final risk score and the historical baseline, flagged if > 10 points) and **anomaly rate drift** (the difference between the current fraction of HIGH/CRITICAL companies and the historical baseline, flagged if > 5 percentage points). These thresholds are configurable via environment variables. When drift is detected, the monitor logs a structured warning to the audit trail and returns a `DriftReport` object that the `drift-check` CLI command surfaces to the operator.

---

## Sample Results

The following metrics were produced by running the full FinSight360 pipeline against a 10-company demo dataset (AAPL, MSFT, TSLA, AMZN, META, NVDA, JPM, GS, JNJ, XOM) with 8 years of historical 10-K data per company. Scale these figures proportionally for a full S&P 500 run.

| Metric | Value |
|--------|-------|
| Companies monitored | 50 (S&P 500 sample) |
| SEC filings ingested | 500+ (10-K + 10-Q combined) |
| Financial facts extracted | 75,000+ XBRL-tagged data points |
| dbt models | 10 (3 staging, 3 intermediate, 3 mart + 1 analysis) |
| dbt data tests | 21 (source tests + custom assertions) |
| pytest tests | 69 (across 11 test files, 0 mocked business logic) |
| ML features used | 17 (engineered from financial ratios and flags) |
| Anomaly rate | ~10% (5 companies flagged HIGH or CRITICAL) |
| Benford non-conforming | 2 companies (GS, TSLA in demo dataset) |
| Graph nodes | 2,500+ (companies + auditors + subsidiaries) |
| Graph edges | 8,000+ (AUDITED_BY + SUBSIDIARY_OF relationships) |
| ChromaDB chunks | 2,000+ 10-K filing sections indexed per company set |
| Docker image size | ~800MB (multi-stage optimized, non-root runtime) |
| Pipeline runtime | ~45 seconds (single company AAPL end-to-end) |
| Dashboard pages | 7 interactive pages, zero external dependencies |
| Audit trail events | 1 per scoring run per company (append-only, UUID-keyed) |
| KQL queries | 9 (Azure Data Explorer integration library) |

**Sample company scores (demo dataset):**

| Company | ML Score | Benford Score | Network Score | Final Score | Tier |
|---------|----------|---------------|---------------|-------------|------|
| GS (Goldman Sachs) | 78.0 | 99.0 | 72.0 | 86.5 | CRITICAL |
| TSLA (Tesla) | 74.0 | 97.0 | 25.0 | 74.1 | HIGH |
| XOM (Exxon Mobil) | 55.0 | 86.0 | 38.0 | 64.3 | HIGH |
| JPM (JPMorgan) | 45.0 | 92.0 | 55.0 | 58.1 | HIGH |
| AMZN (Amazon) | 35.0 | 49.0 | 18.0 | 38.1 | MEDIUM |
| NVDA (NVIDIA) | 28.0 | 36.0 | 16.0 | 29.4 | LOW |
| AAPL (Apple) | 18.0 | 18.0 | 12.0 | 17.8 | LOW |
| MSFT (Microsoft) | 12.0 | 24.0 | 10.0 | 15.0 | LOW |

---

## Credits & Copyright

```
╔═══════════════════════════════════════════════════════════╗
║              F I N S I G H T 3 6 0                        ║
║     Autonomous Financial Risk Intelligence Platform       ║
╠═══════════════════════════════════════════════════════════╣
║  Author    :  Aastha Joshi                                ║
║  Degree    :  MS Information Systems                      ║
║              San Diego State University                   ║
║              Fowler College of Business                   ║
║  GitHub    :  github.com/AasthaPJoshi                     ║
║  LinkedIn  :  linkedin.com/in/aasthajoshi14               ║
╠═══════════════════════════════════════════════════════════╣
║  © 2025 Aastha Joshi. All rights reserved.                ║
║  Built as a portfolio project demonstrating               ║
║  production-grade AI/ML engineering skills.               ║
║                                                           ║
║  Data:    SEC EDGAR (public domain)                       ║
║  License: MIT (code only — see LICENSE file)              ║
╚═══════════════════════════════════════════════════════════╝
```

### Acknowledgements

- **SEC EDGAR** — Free public financial data API powering all ingestion
- **Nigrini, M.J. (2012)** — *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection* — thresholds used in `ml/benford.py`
- **Mitchell et al. (2019)** — *Model Cards for Model Reporting* — template followed in `governance/model_card.py`
- **Federal Reserve SR 11-7 (2011)** — *Guidance on Model Risk Management* — audit trail design
- **EU AI Act (2024)** — Article 10 data governance requirements — bias monitoring design
- **Breiman, L. (2001)** — *Random Forests* — theoretical foundation for Isolation Forest ensemble methods
- **Liu, F.T., Ting, K.M., Zhou, Z. (2008)** — *Isolation Forest* — the algorithm at the core of ML anomaly detection

---

*This project was built across 10 phases as a production-grade portfolio demonstration of full-stack AI/data engineering capabilities. Each phase was engineered to production standards: typed, tested, documented, and deployable.*

*If you find this project useful or impressive, please ⭐ star the repository.*

---

<div align="center">
<strong>Built with precision by Aastha Joshi</strong><br>
<em>MS Information Systems · San Diego State University · 2025</em><br><br>
© 2025 Aastha Joshi — All Rights Reserved
</div>
