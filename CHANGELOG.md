# Changelog

All notable changes to FinSight360 are documented here.

Format: `[Phase X] — YYYY-MM — Summary`

Each phase section lists what was **Added**, **Changed**, and **Fixed**.
Phases were built sequentially; later phases depend on all prior phases being complete.

---

## [Phase 10] — 2025-05 — Final Documentation

The tenth and final phase completed the public-facing documentation layer of the repository, making it self-sufficient for a new contributor or reviewer to understand the full system without any prior context.

### Added
- `README.md` — complete 700+ line production-grade README with architecture diagram, data flow walkthrough, full tech stack table, signal architecture deep-dive, 80-file project structure, CLI reference, dashboard page descriptions, Responsible AI section, MLOps section, sample results table, and credits
- `CONTRIBUTING.md` — contributor guide covering fork/clone workflow, branch naming conventions, test requirements, code style (ruff + type hints), and PR template
- `CHANGELOG.md` — this file; full phase-by-phase history of all 10 phases
- `LICENSE` — MIT License file with full legal text

### Changed
- Nothing — documentation-only phase

### Fixed
- Nothing — documentation-only phase

---

## [Phase 9] — 2025-05 — Portfolio Polish

The ninth phase added demo infrastructure, documentation artifacts, and correctness fixes to the dbt transformation layer that had been exposed by the interaction between the demo seeder and the AAPL integration test fixture.

### Added
- `scripts/seed_demo_data.py` — seeds 10 realistic companies (AAPL, MSFT, TSLA, AMZN, META, NVDA, JPM, GS, JNJ, XOM) with 8 years of historical 10-K financial data, ML anomaly scores, Benford results, final risk scores, and audit trail entries; enables full dashboard exploration without running the ingestion pipeline
- `Makefile` target `seed-demo` — `python scripts/seed_demo_data.py`
- `finsight360/cli.py` command `seed-demo` — CLI entry point for the seeder
- `docs/architecture.md` — deep technical architecture document covering all 8 module components with class-level detail and 10 documented design decisions
- `docs/resume_bullets.md` — 10 XYZ-format resume bullets for portfolio use
- `docs/interview_prep.md` — 15 technical Q&A with exact class and file name references from the codebase
- `docs/linkedin_post.md` — 350-word LinkedIn launch post template

### Changed
- `dbt/models/staging/stg_companies.sql` — added `QUALIFY row_number() OVER (PARTITION BY cik ORDER BY updated_at DESC) = 1` to deduplicate companies that exist with both padded (`0000320193`) and unpadded (`320193`) CIK formats after `lpad()` normalization; prevents 2× fan-out in intermediate joins
- `dbt/models/marts/mart_financial_summary.sql` — rewrote to deduplicate 10-K rows per `(cik, fiscal_year)` using `UNION ALL` + `QUALIFY row_number()` pattern; removed redundant `stg_companies` join in the mart (company fields now sourced from the `int_quarterly_trends` CTE which already carries them); eliminates 4–8× row fan-out caused by duplicate company records

### Fixed
- `assert_metrics_period_unique` dbt test was failing with "Got 8 results" when the demo seeder had been run against a database that also contained the AAPL integration test fixture data; the `QUALIFY` deduplication in the mart now guarantees at most one 10-K row per `(cik, fiscal_year)` regardless of raw data duplicates

---

## [Phase 8] — 2025-05 — Production MLOps

The eighth phase built the full production deployment and operations infrastructure: Docker packaging, CI/CD automation, MLflow model lifecycle management, health monitoring, and drift detection.

### Added
- `Dockerfile` — multi-stage build: `builder` stage with GCC and Python build tools; `runtime` stage with lean image, non-root `finsight` user (UID 1000), and `HEALTHCHECK` on Streamlit `/healthz`
- `docker-compose.yml` — orchestrates three services: `finsight-dashboard` (app), `neo4j` (graph DB on ports 7474/7687), `mlflow` (tracking server on port 5001)
- `.github/workflows/ci.yml` — triggers on push/PR; runs `pytest -m "not integration"` with Python 3.11 matrix
- `.github/workflows/cd.yml` — triggers on `main` merge; builds Docker image tagged with git SHA and pushes to configured container registry
- `.github/workflows/pipeline.yml` — cron `0 6 * * *`; runs full ingestion-to-scoring pipeline for all configured tickers
- `mlops/health_check.py` — `HealthChecker` class validating 6 components: database connectivity, data freshness, model artifact presence, ChromaDB availability, reference data files, MLflow connectivity
- `mlops/model_monitor.py` — `ModelMonitor` class detecting score drift (threshold: 10 points) and anomaly rate drift (threshold: 5 percentage points); returns structured `DriftReport`
- `mlops/registry.py` — `ModelRegistry` class wrapping MLflow Model Registry: `register_model()`, `promote_to_production()`, `archive_old_versions()`, `get_production_model_uri()`
- `finsight360/cli.py` commands: `health`, `drift-check`, `registry-status`
- `Makefile` targets: `health`, `drift`, `registry`, `docker-build`, `docker-run`, `docker-stop`, `docker-logs`, `docker-clean`
- `tests/test_mlops.py` — 10 tests covering health checker, drift detection, and registry operations

### Changed
- `requirements.txt` — pinned `mlflow==2.13.2`, added `docker` SDK reference in comments
- `Makefile` — added `full-pipeline` target chaining ml-run → graph-build → governance-report → dashboard

### Fixed
- `structlog` reserved keyword conflict: `log.info(..., event=...)` raises `TypeError` in `make_filtering_bound_logger`; all call sites updated to use `ev=` instead of `event=`
- `numpy.bool_` identity check: `assert rows.iloc[0]["is_anomaly"] is True` fails for DuckDB numpy boolean returns; changed to `== True`

---

## [Phase 7] — 2025-05 — Responsible AI Governance

The seventh phase built the complete responsible AI governance layer satisfying SR 11-7 model risk management guidance and EU AI Act Article 10 data governance requirements.

### Added
- `governance/audit_trail.py` — `AuditTrailManager` class with `AuditEntry` dataclass; writes every scoring event to append-only `audit_trail` DuckDB table with UUID, model version, all feature values, SHA-256 feature hash, timestamp; `ON CONFLICT DO NOTHING` enforces immutability
- `governance/bias_analyzer.py` — `BiasAnalyzer` class computing Z-score disparate impact across industry groups; Fairlearn integration for demographic parity metrics; flags groups with |Z| > 2.0
- `governance/shap_explainer.py` — `SHAPExplainer` wrapper for `shap.TreeExplainer`; `explain_company()` returns per-feature Shapley values; `save_waterfall_plot()` exports PNG; `get_feature_importance_df()` returns ranked DataFrame
- `governance/model_card.py` — `ModelCard` class generating Mitchell et al. (2019) compliant documentation; fields: model details, intended use, factors, metrics, evaluation data, training data, quantitative analyses, ethical considerations, caveats; exports both `model_card.json` and `model_card.md`
- `governance/kql_queries.py` — `KQLQueries` class with 9 pre-written Kusto Query Language queries for Azure Data Explorer: high-risk alerts, audit trail ingestion, anomaly trend analysis, Benford non-conformance report, model drift detection, false positive analysis, industry risk distribution, weekly scoring summary, compliance audit export
- `dashboard/pages/07_governance.py` — Governance dashboard page: paginated audit trail table, SHAP feature importance bar chart, industry bias Z-score chart, rendered model card, KQL query library with copy buttons
- `finsight360/cli.py` commands: `governance-report`, `model-card`
- `Makefile` target: `governance-report`
- `docs/kql_queries.kql` — standalone KQL query file for Azure ADX
- `tests/test_governance.py` — 9 tests covering audit trail append-only behavior, bias Z-score computation, SHAP value shape, model card JSON schema, KQL query string validity

### Changed
- `ml/pipeline.py` — integrated `AuditTrailManager.log_scoring_event()` call after each company is scored; audit entry carries full feature vector
- `scripts/seed_demo_data.py` — extended to seed audit trail entries for all 10 demo companies

### Fixed
- Nothing prior to this phase required fixing

---

## [Phase 6] — 2025-05 — Streamlit Dashboard

The sixth phase built the 7-page interactive Streamlit application connecting all analysis layers to a visual interface accessible to non-technical stakeholders.

### Added
- `dashboard/app.py` — main application entry point; sidebar navigation; database connection management; auto-seeds demo data when DB is empty
- `dashboard/data_loader.py` — `DataLoader` class with `@st.cache_data` decorated methods for all DuckDB queries: `get_risk_scores()`, `get_financial_summary()`, `get_benford_results()`, `get_audit_trail()`, `get_company_profile()`
- `dashboard/components/charts.py` — 8 reusable Plotly chart functions: `risk_gauge()`, `signal_bar()`, `trend_line()`, `treemap()`, `scatter_landscape()`, `industry_heatmap()`, `benford_bar()`, `shap_waterfall()`
- `dashboard/components/metrics_row.py` — `render_kpi_row()` function rendering 4 Streamlit metric cards with delta indicators in a single row
- `dashboard/components/risk_table.py` — `render_risk_table()` function applying tier-colored row styling to a Streamlit dataframe
- `dashboard/pages/01_executive_overview.py` — KPI row, risk tier pie chart, risk vs Benford scatter, industry heatmap, top-15 risk companies sortable table
- `dashboard/pages/02_risk_heatmap.py` — Plotly treemap sized by revenue proxy and colored by risk tier; sidebar filters for industry, tier, score range; full company ranking table
- `dashboard/pages/03_company_deep_dive.py` — company selector, risk gauge, three-signal decomposition bar chart, active flag indicators, four rolling trend charts (revenue, gross margin, current ratio, debt-to-equity), AI brief display
- `dashboard/pages/04_network_graph.py` — interactive pyvis HTML graph embedded in Streamlit, node color by tier, edge type filter, click-to-inspect metadata panel
- `dashboard/pages/05_benford_analysis.py` — conformity status pie, MAD histogram, per-company observed vs. expected digit frequency chart, chi-square p-value summary table
- `dashboard/pages/06_ai_analyst.py` — LangChain RAG chat interface, company context filter, six example starter questions, session state chat history, source passage display
- `.streamlit/config.toml` — theme configuration: primary color `#1f77b4`, dark background, port 8501
- `tests/test_dashboard.py` — 6 tests covering DataLoader query results, chart function return types, and component rendering

### Changed
- `Makefile` — added `dashboard` target: `streamlit run dashboard/app.py`

### Fixed
- Nothing prior to this phase required fixing

---

## [Phase 5] — 2025-05 — LangChain RAG System

The fifth phase built the GenAI layer: an LCEL-based retrieval-augmented generation pipeline that transforms 10-K filing text into queryable financial intelligence.

### Added
- `genai/document_loader.py` — `SECFilingLoader` class fetching 10-K HTML from EDGAR, extracting Item 1A (Risk Factors) and Item 7 (MD&A) sections using regex, splitting with `RecursiveCharacterTextSplitter(chunk_size=1000, overlap=200)`
- `genai/vector_store.py` — `FinSightVectorStore` class wrapping ChromaDB persistent client; `index_company()` embeds chunks with `text-embedding-3-small` and stores with metadata; `query()` returns top-k similar passages
- `genai/rag_chain.py` — `FinSightRAGChain` class implementing LangChain LCEL pipeline: `retriever | prompt_template | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()`; `ask(question, ticker)` and `generate_risk_summary(ticker)` methods
- `genai/sentiment_analyzer.py` — `SentimentAnalyzer` class combining lexicon-based tone scoring (positive/negative/uncertainty word lists from Loughran-McDonald) with optional LLM sentiment classification
- `genai/risk_narrator.py` — `RiskNarrator` class synthesizing a structured master risk brief from ML scores, Benford results, network risk, and RAG-retrieved filing passages; returns JSON with `summary`, `key_risks`, `red_flags`, `confidence`
- `genai/qa_interface.py` — `GenAIInterface` unified entry point: `ask()`, `brief()`, `index()` routing to appropriate underlying class
- `finsight360/cli.py` commands: `index --ticker`, `ask`, `brief --ticker`
- `tests/test_genai.py` — 8 tests covering document loader, vector store operations, RAG chain response structure, sentiment scoring, and narrator output schema

### Changed
- `requirements.txt` — added `langchain`, `langchain-openai`, `langchain-chroma`, `chromadb`, `openai` with pinned versions

### Fixed
- Nothing prior to this phase required fixing

---

## [Phase 4] — 2025-04 — Corporate Knowledge Graph

The fourth phase built the corporate intelligence layer modeling relationships between companies, auditors, and subsidiaries.

### Added
- `graph/graph_builder.py` — `CorporateGraphBuilder` class loading `sp500_auditors.json` (50 CIK→auditor mappings) and `sp500_relationships.json` (30 subsidiary/JV relationships); builds `networkx.MultiDiGraph` with `AUDITED_BY` and `SUBSIDIARY_OF` typed edges
- `graph/cluster_detector.py` — `ClusterDetector` class applying `python-louvain` Louvain algorithm; computes `community_id`, PageRank score (via `nx.pagerank()`), and betweenness centrality (via `nx.betweenness_centrality()`); assigns `risk_cluster_rating` based on within-cluster average risk
- `graph/neo4j_loader.py` — `Neo4jLoader` class connecting to Neo4j via bolt protocol; `load_graph()` issues `MERGE` Cypher statements for all nodes and edges; `run_query()` executes arbitrary Cypher
- `graph/graph_queries.py` — `GraphQueries` class providing both Cypher implementations (for Neo4j) and NetworkX fallback implementations for: `get_company_neighbors()`, `get_audit_network()`, `get_risk_cluster()`, `get_high_risk_neighbors()`
- `graph/graph_risk_scorer.py` — `GraphRiskScorer` class computing `network_risk_score` as PageRank-weighted average of immediate neighbor final risk scores; exposed as `score_company_network_risk(cik)`
- `data/reference/sp500_auditors.json` — 50 CIK to Big Four / regional auditor name mappings
- `data/reference/sp500_relationships.json` — 30 SUBSIDIARY_OF and JOINT_VENTURE relationships
- `finsight360/cli.py` command: `graph-build --skip-neo4j`
- `Makefile` target: `graph-build`
- `tests/test_graph.py` — 5 tests covering graph construction, Louvain communities, PageRank computation, Neo4j loader mock, and network risk score calculation

### Changed
- `ml/pipeline.py` — calls `GraphRiskScorer` after Benford + Isolation Forest to produce the 3-signal weighted final score
- `storage/schema.sql` — added `ml_anomaly_scores` table columns: `community_id`, `late_filings_count`, `total_signal_count`

### Fixed
- Nothing prior to this phase required fixing

---

## [Phase 3] — 2025-04 — ML Anomaly Detection

The third phase built the machine learning scoring engine: Benford's Law forensics, Isolation Forest anomaly detection, SHAP explainability, and MLflow experiment tracking.

### Added
- `ml/benford.py` — `BenfordAnalyzer` class computing first-digit distribution from a list of financial values; calculates MAD (Mean Absolute Deviation from Benford expected frequencies), chi-square statistic and p-value via `scipy.stats.chisquare()`; maps to conformity ratings using Nigrini (2012) thresholds; outputs `BenfordResult` dataclass and populates `benford_results` DuckDB table
- `ml/anomaly_detector.py` — `AnomalyDetector` class wrapping `sklearn.ensemble.IsolationForest(n_estimators=200, contamination=0.10, random_state=42)` with `RobustScaler` preprocessing; `fit_transform()` returns normalized feature matrix; `score_companies()` returns anomaly percentile (0–100) per company
- `ml/feature_builder.py` — `FeatureBuilder` class querying `mart_financial_summary` from DuckDB and constructing a 17-column feature matrix: gross_margin, operating_margin, net_margin, current_ratio, debt_to_equity, ar_to_revenue, revenue_yoy_growth, cash_earnings_gap, roe, sga_ratio, rd_intensity, and 6 binary flag columns
- `ml/model_registry.py` — `MLflowRegistry` class logging experiments: `start_run()`, `log_params()`, `log_metrics()`, `log_model()`, `end_run()`; also handles `mlflow.sklearn.log_model()` artifact storage
- `ml/explainer.py` — `SHAPExplainer` class using `shap.TreeExplainer(model)` for exact Shapley values; `explain(X)` returns SHAP values array; `save_waterfall(shap_values, features, ticker)` exports PNG to `docs/`
- `ml/pipeline.py` — `MLPipeline` class orchestrating the full scoring run: load features → fit Benford → fit Isolation Forest → compute SHAP → combine signals → write `ml_anomaly_scores` and `final_risk_scores` to DuckDB → log to MLflow
- `finsight360/cli.py` command: `ml-run`, `shap-run`
- `Makefile` targets: `ml-run`, `shap-run`
- `tests/test_ml_pipeline.py` — 6 tests covering Benford MAD calculation, Isolation Forest score range, feature matrix shape, SHAP value dimensions, pipeline end-to-end on AAPL fixture data

### Changed
- `storage/schema.sql` — added `ml_anomaly_scores` table (cik, ticker, isolation_forest_score, anomaly_percentile, ml_risk_score, is_anomaly, risk_tier, computed_at) and `benford_results` table (cik, n_observations, mad_score, chi_square_statistic, chi_square_p_value, conformity_rating, benford_risk_score, suspicious_digits, interpretation, computed_at)
- `requirements.txt` — added `scikit-learn`, `shap`, `scipy`, `mlflow`, `fairlearn`

### Fixed
- Nothing prior to this phase required fixing

---

## [Phase 2] — 2025-04 — dbt Transformation Layer

The second phase built the three-layer dbt transformation pipeline converting raw EDGAR data into analysis-ready analytical tables with 21 automated data quality tests.

### Added
- `dbt/dbt_project.yml` — project configuration: `finsight360`, DuckDB adapter, schema names `main_staging` / `main_intermediate` / `main_marts`, model materialization settings (staging: view, intermediate: ephemeral, marts: table)
- `dbt/profiles.yml` — DuckDB connection profile reading `DUCKDB_PATH` environment variable; `dev` target
- `dbt/packages.yml` — `dbt-labs/dbt_utils` dependency for `safe_divide` macro pattern
- `dbt/models/staging/sources.yml` — source definitions for `raw.companies`, `raw.filings`, `raw.financial_facts`, `raw.financial_metrics`; 14 not_null and unique source tests
- `dbt/models/staging/stg_companies.sql` — `lpad(cik,10,'0')` normalization, SIC code to industry_group CASE mapping, `QUALIFY row_number()` CIK deduplication
- `dbt/models/staging/stg_filings.sql` — form type normalization, date casting, accession number cleaning
- `dbt/models/staging/stg_financial_facts.sql` — concept label standardization, taxonomy tagging
- `dbt/models/staging/stg_financial_metrics.sql` — `safe_divide()` macro for 5 ratio columns, `lpad()` CIK normalization, `start_date` variable filter
- `dbt/models/intermediate/int_financial_ratios.sql` — 12 computed ratios, 4-quarter rolling gross margin average, `ar_to_revenue_prior` lag, `period_rank_in_year` row_number window
- `dbt/models/intermediate/int_quarterly_trends.sql` — 6 binary risk flag columns, `total_signal_count` sum
- `dbt/models/intermediate/int_company_profiles.sql` — latest-period snapshot per company
- `dbt/models/marts/mart_risk_candidates.sql` — companies with `total_signal_count >= 1`; `pre_ml_risk_score` weighted sum; `risk_tier` CASE; `dominant_signal` column identifying highest flag
- `dbt/models/marts/mart_financial_summary.sql` — full history with 10-K deduplication; company fields from trends CTE
- `dbt/models/marts/mart_company_dashboard.sql` — one row per company for dashboard KPI display
- `dbt/macros/safe_divide.sql` — `CASE WHEN denominator = 0 THEN NULL ELSE numerator / denominator END` macro
- `dbt/tests/assert_metrics_period_unique.sql` — custom test: `(cik, fiscal_year)` unique for `form='10-K'`
- `dbt/tests/assert_revenue_non_negative.sql` — custom test: revenues ≥ 0
- `finsight360/cli.py` dbt commands delegated to subprocess (dbt-run, dbt-test)
- `Makefile` targets: `dbt-deps`, `dbt-run`, `dbt-test`, `dbt-docs`
- `tests/test_dbt_models.py` — 4 tests: `test_dbt_run_succeeds`, `test_dbt_staging_models_exist`, `test_dbt_mart_models_exist`, `test_dbt_tests_pass`

### Changed
- `requirements.txt` — added `dbt-core==1.8.3`, `dbt-duckdb==1.8.1`

### Fixed
- Nothing prior to this phase required fixing

---

## [Phase 1] — 2025-04 — Data Ingestion Engine

The first phase built the complete SEC EDGAR data ingestion pipeline from scratch: async HTTP client, XBRL parser, filing processor, Pydantic data models, DuckDB schema, and configuration layer.

### Added
- `ingestion/edgar_client.py` — `EDGARClient` async class; `get_company_facts(cik)` hits `https://data.sec.gov/api/xbrl/companyfacts/{CIK}.json`; rate limiting via `asyncio.sleep(0.12)` between requests; `@retry` decorator from `utils/retry.py`
- `ingestion/xbrl_parser.py` — `XBRLParser` class; `parse_company_facts()` extracts 25 financial concepts from EDGAR JSON; handles both `us-gaap` and `dei` taxonomies; deduplicates by `(concept, period_end, accn)` preferring annual over instantaneous units
- `ingestion/filing_processor.py` — `FilingProcessor` async class; `process_company(ticker)` orchestrates full ingestion for one ticker; `process_all(tickers)` uses `asyncio.Semaphore(5)` for concurrency; writes to `ingestion_runs` table on start and completion
- `ingestion/models.py` — Pydantic v2 `BaseModel` classes: `Company`, `Filing`, `FinancialFact`, `FinancialMetric`; all fields typed and validated; `model_config = ConfigDict(from_attributes=True)`
- `storage/database.py` — `Database` class with thread-local DuckDB connections; `get_connection()`, `upsert_company()`, `upsert_filing()`, `upsert_financial_fact()`, `upsert_financial_metric()`, `close()`; uses `ON CONFLICT DO UPDATE SET` for PK tables
- `storage/schema.sql` — DDL for 5 tables: `companies`, `filings`, `financial_facts`, `financial_metrics`, `ingestion_runs`; 5 indexes; all timestamp columns with `DEFAULT current_timestamp`
- `config/settings.py` — `Settings` class extending `pydantic_settings.BaseSettings`; reads `EDGAR_USER_AGENT`, `DUCKDB_PATH`, `OPENAI_API_KEY`, `NEO4J_*`, `MLFLOW_TRACKING_URI` from environment / `.env` file
- `utils/logger.py` — `get_logger(name)` factory returning `structlog` bound logger with JSON renderer in production and console renderer in development
- `utils/retry.py` — `retry_with_backoff(max_attempts=3, wait_min=1, wait_max=10)` decorator factory wrapping `tenacity.retry`
- `finsight360/__init__.py`, `finsight360/cli.py` — `click`-based CLI entry point with `db` command group and `ingest` command
- `tests/conftest.py` — pytest `integration` marker; `seed_aapl_data` session-scoped fixture calling `tests/fixtures/aapl_seed.py`
- `tests/fixtures/aapl_seed.py` — AAPL historical data: 5 annual 10-K periods (2019–2023) + 5 quarterly 10-Q periods; real SEC figures used
- `tests/test_database.py` — 6 tests: schema creation, company upsert, filing upsert, fact upsert, metric upsert, duplicate handling
- `tests/test_edgar_client.py` — 5 tests: client initialization, rate limiting, retry on 429, CIK padding, response parsing
- `tests/test_xbrl_parser.py` — 7 tests: concept extraction, deduplication, missing concept handling, unit filtering, period type detection
- `Makefile` — initial targets: `install`, `test`, `test-all`, `ingest`, `ingest-company`
- `.env.example` — template with all required and optional environment variables documented
- `pyproject.toml` — project metadata, ruff linting configuration (line-length=100, target=py311)
- `requirements.txt` — initial pinned dependencies: httpx, pydantic, pydantic-settings, duckdb, tenacity, structlog, click

### Changed
- Nothing — initial phase

### Fixed
- Nothing — initial phase

---

*FinSight360 — 10 phases · 69 tests · 80+ files · 10,000+ lines · Built by Aastha Joshi · © 2025*
