# FinSight360 System Architecture

## Overview

FinSight360 is a layered data platform designed around a single constraint: every component must be independently testable, replaceable, and observable. The system is structured as a directed pipeline — data flows in one direction from ingestion through transformation, ML, graph analytics, GenAI enrichment, and governance — with no circular dependencies between layers. Each layer exposes a clean interface to the next, which means Isolation Forest can be swapped for XGBoost without touching the graph layer, or DuckDB can be replaced with MotherDuck without changing the ML code.

The second design principle is defense-in-depth for risk scoring. Financial fraud is adversarial — a motivated actor can learn to game any single detection signal. By combining three mathematically independent signals (statistical ML, digit-distribution forensics, and network graph topology), FinSight360 requires an anomaly to be visible from multiple angles before it reaches a HIGH or CRITICAL tier. This dramatically reduces false positives while preserving recall on genuine manipulation patterns.

---

## Data Flow

```
SEC EDGAR API
    │
    ▼ (httpx + tenacity, 0.12s rate limit)
EdgarClient.get_submissions() / get_xbrl_facts()
    │
    ▼
XBRLParser.extract_financial_facts()          ← structured XBRL → Python dicts
    │
    ▼
FilingProcessor.process_ticker()              ← orchestrates ingestion + metric calc
    │  writes to DuckDB: companies, filings,
    │  financial_facts, financial_metrics
    ▼
dbt run (staging → intermediate → marts)      ← SQL transformations inside DuckDB
    │  produces: mart_risk_candidates,
    │            mart_company_summary
    ▼
    ├── ML Pipeline ──────────────────────────
    │   BenfordAnalyzer.analyze_all_companies()   ← Benford score per company
    │   FeatureBuilder.build_features()           ← 17-feature matrix
    │   IsolationForestModel.train_and_score()    ← ML anomaly scores
    │   SHAPExplainer.explain_all_companies()     ← per-feature attribution
    │
    ├── Graph Pipeline ───────────────────────
    │   CorporateGraphBuilder.build()             ← NetworkX G(V,E)
    │   ClusterDetector.detect_communities()      ← Louvain partition
    │   ClusterDetector.compute_centrality()      ← PageRank + betweenness
    │   Neo4jLoader.load_graph()                  ← persist to Neo4j (optional)
    │
    └── Risk Scoring ─────────────────────────
        GraphRiskScorer.compute_final_scores()    ← 50% ML + 30% Benford + 20% net
            │  writes: final_risk_scores table
            ▼
        AuditTrailManager.log_all_scores()        ← append-only compliance log
            │
            ▼
        Streamlit Dashboard                       ← read-only DuckDB queries
```

---

## Component Architecture

### 1. Data Ingestion Layer

**`ingestion/edgar_client.py`** — Rate-limited HTTP client for SEC EDGAR's public REST API. Uses `httpx` (async-capable, connection pooling) with `tenacity` for exponential-backoff retry on 429/503 responses. The 0.12-second inter-request delay is derived from SEC's published fair-use guidelines (10 requests/second maximum). All requests include a required `User-Agent` header identifying the application and contact email per SEC policy.

**`ingestion/xbrl_parser.py`** — Extracts structured financial facts from EDGAR's XBRL JSON format. Handles the taxonomy complexity: the same concept (e.g., revenue) can appear under dozens of GAAP concept names (`Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`, `SalesRevenueNet`). The parser resolves this through a priority-ordered concept list and returns the first matching non-null value per period.

**`ingestion/filing_processor.py`** — Orchestration layer that coordinates `EdgarClient` and `XBRLParser`, writes to DuckDB using upsert-safe SQL, and computes derived financial metrics (gross margin, AR-to-revenue ratio, YoY growth) immediately after ingestion. This avoids a separate ETL job for simple ratio calculations.

**Why `httpx` over `requests`?** Connection pooling and keep-alive reduce latency for sequential API calls. The async interface (`httpx.AsyncClient`) is available if we move to concurrent ingestion in Phase 10.

**Why `tenacity` over manual retry?** Declarative retry logic (`@retry(stop=stop_after_attempt(5), wait=wait_exponential(...))`) keeps the HTTP client code clean and makes retry behavior configurable without changing call sites.

---

### 2. Storage Layer

**DuckDB** (`storage/database.py`) — In-process analytical database running embedded inside the Python process. Schema: `companies`, `filings`, `financial_facts`, `financial_metrics`, `ingestion_runs` (core tables) + `ml_anomaly_scores`, `benford_results`, `final_risk_scores`, `shap_values`, `audit_trail` (ML output tables). Key indexes: `(cik, period_of_report)` on financial_metrics for time-series queries; `(cik)` and `(generated_at)` on audit_trail for compliance reporting.

**Why DuckDB over PostgreSQL?** For analytical workloads (aggregations, window functions, percentiles across all companies), DuckDB's columnar execution engine is 10–100× faster than row-oriented PostgreSQL on the same queries. It requires zero infrastructure — the database is a single file. For a 500-company portfolio, DuckDB handles all queries sub-second. At 50,000 companies, the migration path is MotherDuck (DuckDB cloud) with minimal code changes.

**Neo4j 5.20** (`graph/neo4j_loader.py`) — Property graph database for persistent corporate relationship storage. Nodes: `Company` (with ML risk scores), `Auditor`. Edges: `AUDITED_BY`, `SUBSIDIARY_OF`. Neo4j is optional — all graph analytics run in NetworkX; Neo4j provides Cypher query access and graph visualization via the Neo4j Browser.

**ChromaDB** (`genai/vector_store.py`) — Local vector store for SEC filing text chunks. Embeddings: `text-embedding-3-small` (1536 dimensions). Documents chunked at 1000 tokens with 200-token overlap. Persisted to `data/chromadb/` for reuse across sessions. **Why ChromaDB over Pinecone?** No API key, no monthly cost, full local control, no data leaving the environment. At scale (>1M vectors), the migration path to Pinecone or Weaviate requires changing only `vector_store.py`.

---

### 3. Transformation Layer

dbt-duckdb runs SQL transformations directly inside the DuckDB process (no external warehouse). The three-layer pattern:

- **Staging models** (`stg_companies`, `stg_financial_metrics`, `stg_xbrl_facts`) — type casting, null handling, deduplication, column renaming to consistent snake_case. No business logic.
- **Intermediate models** (`int_financial_ratios`) — derived ratio calculations (current ratio = current assets / current liabilities). Joins staging tables. No aggregation.
- **Mart models** (`mart_risk_candidates`, `mart_company_summary`) — final analytical tables optimized for dashboard consumption. Adds risk signal flags: `flag_ar_acceleration`, `flag_cashflow_divergence`, `flag_margin_compression`, `flag_aggressive_recognition`, `flag_high_leverage`, `flag_liquidity_stress`.

**Why dbt over raw SQL?** Version-controlled transformations, documented lineage (`dbt docs generate`), built-in data testing (`dbt test`), and Jinja2 templating for DRY SQL. The staging/mart separation means the dashboard never queries raw tables directly — it queries tested, documented models.

---

### 4. ML Layer

**Benford's Law** (`ml/benford.py`) — For any naturally occurring financial dataset, the first significant digit d appears with probability log₁₀(1 + 1/d). Financial manipulation often violates this distribution because fabricated numbers cluster around psychologically convenient values (round numbers, numbers just below thresholds). FinSight360 computes both the Mean Absolute Deviation (MAD) and chi-square p-value against Benford's expected distribution. Conformity thresholds follow Nigrini (2012): MAD < 0.006 = conforming, 0.006–0.012 = acceptable, 0.012–0.015 = marginal, > 0.015 = non-conforming.

**Isolation Forest** (`ml/isolation_forest.py`) — Unsupervised anomaly detection algorithm. Randomly partitions the feature space; anomalous points require fewer partitions to isolate (they sit in sparse regions). Trained on 17 features including financial ratios, binary risk flags, and filing behavior metrics. Parameters: `contamination=0.10` (expects ~10% anomalies), `n_estimators=100`, `random_state=42`. `RobustScaler` normalizes features using median and IQR — robust to the extreme outliers common in financial data (unlike `StandardScaler` which is sensitive to outliers).

**Why unsupervised?** No labeled training data for financial fraud exists at scale. The SEC's AAER (Accounting and Auditing Enforcement Releases) database has ~1,000 confirmed fraud cases over 30 years — far too sparse for supervised learning on a 500-company universe. Isolation Forest requires no labels and adapts to the data distribution automatically.

**SHAP TreeExplainer** (`governance/shap_explainer.py`) — Exact Shapley value computation for tree-based models (O(TLD) complexity where T=trees, L=leaves, D=depth). Produces a signed contribution for each of the 17 features per company, explaining how much each feature pushed the anomaly score above or below the baseline. Stored in the `shap_values` DuckDB table and visualized as waterfall charts. **Why TreeExplainer over KernelExplainer?** TreeExplainer is exact (not approximate) and 100–1000× faster for tree ensembles.

---

### 5. Graph Intelligence Layer

**`graph/graph_builder.py`** — Constructs a NetworkX heterogeneous graph G(V, E) where V = {Company nodes, Auditor nodes} and E = {AUDITED_BY edges, SUBSIDIARY_OF edges}. Company nodes are annotated with ML risk scores, Benford scores, and tier labels. Built from reference JSON files (`data/reference/sp500_auditors.json`, `sp500_relationships.json`).

**`graph/cluster_detector.py`** — Runs Louvain community detection (`python-louvain`) to partition the graph into communities. Communities of companies sharing auditors or subsidiaries are identified; high-risk communities (avg ML score > threshold) are flagged. Also computes PageRank (which nodes are most "important" in the graph) and betweenness centrality (which nodes are critical connectors between communities — often systemically important intermediaries).

**Risk propagation**: Network risk score for a company is a function of its community's average ML score and its own centrality. A company with moderate standalone ML risk gets amplified if it's deeply embedded in a high-risk auditor community — reflecting real-world contagion dynamics (cf. Arthur Andersen's simultaneous collapse of multiple client relationships in 2002).

**`graph/neo4j_loader.py`** — Bulk-loads the NetworkX graph into Neo4j using the Python driver. Supports `MERGE` operations (idempotent). The `GraphQueryEngine` provides a unified interface: if Neo4j is available, Cypher queries run there; otherwise, NetworkX in-memory queries are used as fallback.

---

### 6. GenAI Layer

**LangChain LCEL pipeline** (`genai/rag_chain.py`) — Uses LangChain Expression Language (`|` operator) to compose: `retriever | format_docs | prompt | llm | StrOutputParser`. LCEL chains are lazy-evaluated (no execution until `.invoke()`) and streaming-capable. The retriever pulls the top-4 most relevant document chunks from ChromaDB using cosine similarity on OpenAI embeddings.

**`genai/sentiment_analyzer.py`** — Lexicon-based + LLM tone analysis of MD&A sections. Lexicon scan uses a curated word list (going concern language, litigation terms, restatement language) before calling the LLM, avoiding unnecessary API costs when no risk language is detected. The `FilingToneAnalyzer.llm` property is lazy-initialized — no `ChatOpenAI` object is created until the first actual LLM call.

**Why GPT-4o-mini over GPT-4o?** For summarization and risk brief generation tasks, GPT-4o-mini achieves 90%+ of GPT-4o quality at approximately 15% of the cost. The structured prompt template constrains the output format, which reduces the quality gap further.

---

### 7. Responsible AI Layer

**`governance/audit_trail.py`** — Append-only DuckDB table (`audit_trail`) with UUID primary key. Every scoring event writes one row: model name + version, all three risk scores, final tier, is_anomaly flag, top SHAP features (JSON), data hash (SHA-256 of input features for tamper detection), and human review fields. Satisfies SR 11-7 requirements for model documentation, validation traceability, and ongoing monitoring. The `ON CONFLICT (audit_id) DO NOTHING` pattern ensures idempotency without allowing record modification.

**`governance/bias_analyzer.py`** — Computes z-scores of group mean risk scores vs. overall mean. Groups flagged at |z| > 1.5 standard deviations are reported as potentially biased. Covers industry groups (SIC codes) and geographic regions (state of incorporation). Aligned to EU AI Act Article 10 (data governance) requirements for high-risk AI systems in financial services.

**`governance/model_card.py`** — Generates model documentation in Mitchell et al. (2019) format. Reads live stats from DuckDB (companies scored, anomaly rate, human reviews) and writes `docs/model_card.json` + `docs/model_card.md`. Includes intended use, out-of-scope uses, performance metrics, ethical considerations, and regulatory alignment table (SR 11-7, EU AI Act, NIST AI RMF, SOC2).

**`governance/kql_queries.py`** — 9 pre-built KQL queries targeting the `FinSightRiskScores`, `FinSightAuditTrail`, `FinSightBenfordResults`, and `FinSightSHAPValues` tables in Azure Data Explorer. Enables enterprise security teams to integrate FinSight360 audit data with Microsoft Sentinel SIEM workflows.

---

### 8. MLOps Layer

**Multi-stage Dockerfile** — Builder stage installs all Python dependencies (including compilers for C extensions); runtime stage copies only the installed packages and application code. The non-root `finsight` user runs the application. `HEALTHCHECK` verifies DuckDB connectivity every 30 seconds. Image size reduction vs. single-stage: ~60%.

**GitHub Actions** — Three workflows: `ci.yml` runs `pytest -m "not integration"` on every push to `main`, `develop`, and `claude/*` branches. `cd.yml` builds and pushes the Docker image to GitHub Container Registry on main merges and version tags, using layer caching via `type=gha`. `pipeline.yml` runs the full ingestion + ML + governance pipeline on a 6am UTC weekday cron schedule with manual dispatch support.

**MLflow** — Every `IsolationForestModel.train()` call logs parameters (contamination, n_estimators), metrics (anomaly_rate, mean_score, n_companies), and the trained model artifact. The `ModelRegistryManager` in `mlops/registry.py` handles version promotion: None → Staging → Production → Archived.

**`mlops/health_check.py`** — `PipelineHealthChecker.check_all()` runs 6 checks: DuckDB file exists and has required tables; last ingestion run is recent (< 7 days); Isolation Forest model file exists and is fresh (< 7 days); ChromaDB directory has files; reference JSON files present; MLflow experiment directory exists.

**`mlops/model_monitor.py`** — `ModelMonitor.compare_runs()` queries the audit trail grouped by hour, compares the two most recent scoring runs, and raises alerts if: average risk score shifts > 10 points (potential data distribution shift) or anomaly rate changes > 5 percentage points (potential model drift).

---

## Design Decisions

**1. DuckDB over PostgreSQL** — Financial analytics queries (percentiles, window functions, multi-table aggregations) run 10–100× faster in DuckDB's columnar engine. Zero infrastructure setup accelerates development. The `dbt-duckdb` adapter enables full dbt functionality without a separate server. Migration to MotherDuck (cloud DuckDB) when needed requires changing only the connection string.

**2. Isolation Forest over supervised ML** — No labeled fraud dataset exists at the scale needed. Isolation Forest adapts to any data distribution without labels, making it appropriate for a universe that changes over time. The 10% contamination parameter is tunable; it represents our prior on anomaly prevalence, not a fixed fraud rate.

**3. Benford's Law as second signal** — Benford's Law catches a fundamentally different signal than ML: digit-distribution manipulation that occurs even when all ratio-level metrics look normal. A company can engineer ratios to appear acceptable while the underlying digit distributions betray rounding or fabrication. The two signals are statistically independent, making the combination more powerful than either alone.

**4. NetworkX + Neo4j dual-layer** — NetworkX provides in-memory graph analytics without infrastructure dependencies (critical for testing and CI). Neo4j provides Cypher query access for ad-hoc analysis and production persistence. The `GraphQueryEngine` uses Neo4j when available and falls back to NetworkX — same interface, pluggable backend.

**5. LangChain LCEL over legacy LangChain** — LCEL (LangChain Expression Language) chains are composable, streamable, and inspectable. Lazy evaluation means no LLM calls during object construction — critical for test suites that import the module without API keys. The `|` pipe syntax makes the chain structure visually clear.

**6. RobustScaler over StandardScaler** — Financial data contains legitimate extreme outliers (a company growing 500% YoY, negative equity, extreme leverage). `StandardScaler` would pull the mean toward these outliers and compress the variance for normal companies. `RobustScaler` uses median and IQR, making it insensitive to outliers and appropriate for heavy-tailed financial distributions.

**7. ChromaDB over Pinecone** — No API key, no monthly cost, no data leaving the environment. Full local control of the embedding index. For a 10-company demo or a 500-company portfolio, ChromaDB's performance is more than adequate. The migration path to Pinecone requires changing only the `Chroma(...)` call in `vector_store.py`.

**8. Multi-stage Docker** — Builder stage includes gcc, g++, and build-essential for compiling Python C extensions (numpy, scipy, scikit-learn). Runtime stage needs none of these, reducing image size by ~60% and attack surface. The non-root user (`finsight`) satisfies container security best practices required in enterprise deployments.

**9. dbt over raw SQL** — dbt provides version-controlled, documented, tested SQL transformations. `dbt test` catches data quality issues (nulls, referential integrity) before they reach the ML layer. `dbt docs generate` produces a browsable data lineage graph. The staging/mart separation enforces a clean contract between raw data and analytical outputs.

**10. GPT-4o-mini over GPT-4o** — For structured summarization tasks (risk brief generation, MD&A tone analysis), GPT-4o-mini achieves ~90% of GPT-4o quality at ~15% of the cost. The structured prompt template constrains the output format, which reduces the quality gap. GPT-4o is reserved for tasks requiring deep reasoning — none of which appear in the current feature set.
