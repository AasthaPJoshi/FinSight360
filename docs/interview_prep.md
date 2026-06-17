# FinSight360 — Technical Interview Prep

15 Q&A covering architecture, ML decisions, data engineering, and responsible AI. Answers use exact class and file names from the codebase.

---

## Q1: Walk me through the FinSight360 architecture end to end.

FinSight360 is a layered data pipeline. `EdgarClient` pulls structured XBRL data from SEC's public REST API; `XBRLParser` normalizes it into financial facts; `FilingProcessor` writes to DuckDB and computes derived metrics. dbt runs staging and mart transformations inside DuckDB, producing `mart_risk_candidates`. Three risk signals run in parallel: `BenfordAnalyzer` scores digit distributions, `IsolationForestModel` scores 17 financial features, and `CorporateGraphBuilder` + `ClusterDetector` score network risk. `GraphRiskScorer` combines these 50/30/20 into a final score. `AuditTrailManager` logs every scoring event append-only. A 7-page Streamlit dashboard reads from DuckDB; the GenAI layer adds LangChain RAG Q&A over filing text.

---

## Q2: Why did you choose Isolation Forest over supervised ML?

No reliable labeled fraud dataset exists at scale. The SEC's AAER database has roughly 1,000 confirmed fraud cases over 30 years — far too sparse for a supervised model on a 500-company portfolio. Isolation Forest is unsupervised: it learns what "normal" financial reporting looks like from the data itself, then flags companies that are statistically anomalous. The `contamination=0.10` parameter encodes our prior that roughly 10% of filings warrant scrutiny — it's tunable. A supervised alternative (XGBoost with AAER labels) would suffer severe class imbalance and would only detect patterns similar to past frauds, missing novel manipulation techniques.

---

## Q3: Explain Benford's Law and why it's used for fraud detection.

Benford's Law states that in naturally occurring numerical data, the first significant digit d appears with probability log₁₀(1 + 1/d): "1" appears ~30% of the time, "9" appears ~4.6%. Revenue, expense, and asset figures in legitimate financial statements follow this distribution because they arise from multiplicative growth processes. Fabricated numbers tend to cluster around psychologically convenient values — round numbers, numbers just below reporting thresholds — which violates the distribution. `BenfordAnalyzer` computes the Mean Absolute Deviation (MAD) between observed and expected first-digit frequencies, and a chi-square p-value. Thresholds follow Nigrini (2012): MAD > 0.015 = non-conforming. The signal is independent of the ML layer — a company can engineer its ratios to look normal while its underlying digit distributions betray manipulation.

---

## Q4: How does your RAG system work technically?

The RAG pipeline lives in `genai/rag_chain.py` and uses LangChain LCEL. SEC 10-K filing text is extracted and chunked at 1,000 tokens with 200-token overlap in `document_loader.py`. Chunks are embedded with `text-embedding-3-small` (1,536 dimensions) and stored in ChromaDB via `vector_store.py`. At query time, the user's question is embedded and the top-4 most semantically similar chunks are retrieved using cosine similarity. The LCEL chain is: `retriever | format_docs_with_citations | risk_prompt | ChatOpenAI(gpt-4o-mini) | StrOutputParser`. The prompt template instructs the model to answer only from the retrieved context and cite sources. `FilingToneAnalyzer` runs a lexicon pre-scan before LLM calls to avoid API costs when no risk language is detected.

---

## Q5: What is SHAP and how did you use it for explainability?

SHAP (SHapley Additive exPlanations) computes each feature's contribution to a model's prediction using game-theoretic Shapley values. For a company's anomaly score, SHAP answers: "how much did high AR-to-revenue ratio push the score up, and how much did low leverage pull it down?" `SHAPExplainer` in `governance/shap_explainer.py` uses `shap.TreeExplainer` — the exact (non-approximate) algorithm for tree ensembles like Isolation Forest. It runs in O(TLD) time (trees × leaves × depth). For each company, 17 signed SHAP values are stored in the `shap_values` DuckDB table and visualized as waterfall charts. Global feature importance is the mean absolute SHAP value across all companies. This satisfies SR 11-7's requirement that model decisions be explainable to model risk management.

---

## Q6: Why DuckDB instead of a traditional database?

Financial analytics queries — percentiles, window functions, multi-join aggregations across all companies — are analytical workloads, not transactional ones. DuckDB's columnar execution engine runs these 10–100× faster than row-oriented PostgreSQL on the same hardware. It runs in-process (no server, no network round trips), making it ideal for a development setup and a dashboard that needs sub-second query response. The `dbt-duckdb` adapter runs full dbt transformations inside DuckDB without an external warehouse. The tradeoff is concurrency: DuckDB supports one writer at a time. For a system with many concurrent writers, PostgreSQL or ClickHouse would be appropriate. At the current scale (500 companies, one pipeline run at a time), DuckDB is strictly better.

---

## Q7: How did you handle the dbt staging/mart pattern?

The three-layer pattern enforces a clean data contract at each stage. Staging models (`stg_companies`, `stg_financial_metrics`, `stg_xbrl_facts`) do only type casting, null handling, and column renaming — no business logic. Intermediate models (`int_financial_ratios`) compute derived ratios (current ratio, debt-to-equity) by joining staging tables — no aggregation. Mart models (`mart_risk_candidates`, `mart_company_summary`) are the final analytical tables with risk signal flags (`flag_ar_acceleration`, `flag_cashflow_divergence`, etc.) that the ML layer reads. The dashboard never queries raw tables. `dbt test` runs referential integrity and not-null tests on every model before the ML pipeline starts. This means if ingestion produced bad data, the pipeline fails loudly at the dbt layer rather than silently propagating corrupt scores.

---

## Q8: What is community detection and why Louvain algorithm?

Community detection partitions a graph into groups of nodes that are more densely connected to each other than to the rest of the graph. In FinSight360's corporate graph, communities represent clusters of companies sharing auditors or subsidiary relationships. A community of companies all audited by the same firm is a risk contagion unit — if one company in the community commits fraud, the auditor relationship raises scrutiny for all. `ClusterDetector` in `graph/cluster_detector.py` uses the Louvain algorithm (`python-louvain` / `community` package) because it is efficient (O(n log n)), parameterless (no need to specify k), and maximizes modularity — the standard quality metric for community structure. The alternative (spectral clustering, k-means on embeddings) requires specifying k and doesn't respect the graph topology.

---

## Q9: How does your audit trail ensure compliance?

`AuditTrailManager` in `governance/audit_trail.py` writes one record per scoring event to an append-only DuckDB table (`audit_trail`). Each record includes: UUID (`audit_id`), event type, company identifiers, model name + version, all three risk scores, final tier, `is_anomaly` flag, top SHAP features as JSON, a SHA-256 hash of the input feature values (for tamper detection), timestamp, and human review fields. The `ON CONFLICT (audit_id) DO NOTHING` pattern makes writes idempotent while preventing record modification — the audit record is immutable once written. Three DuckDB indexes on `cik`, `generated_at`, and `final_risk_tier` enable fast compliance queries. This design satisfies SR 11-7 (model documentation and ongoing monitoring), SOC2 (complete change log), and the EU AI Act Article 13 (transparency) requirements.

---

## Q10: What drift detection did you implement?

`ModelMonitor` in `mlops/model_monitor.py` queries the `audit_trail` table, groups scoring events by hour, and compares the two most recent pipeline runs. It raises two types of alerts: a `SCORE_DRIFT` alert if the average final risk score shifts more than 10 points between runs (which could indicate a data distribution shift — new companies, different economic conditions, or a data pipeline bug), and an `ANOMALY_RATE_DRIFT` alert if the fraction of companies flagged as anomalies changes by more than 5 percentage points. Both thresholds are configurable class attributes (`SCORE_DRIFT_THRESHOLD`, `ANOMALY_RATE_DRIFT_THRESHOLD`). Drift alerts are surfaced in the CLI via `finsight360 drift-check` and can be wired to Slack/PagerDuty alerts in production.

---

## Q11: Explain your Docker multi-stage build approach.

The Dockerfile has two stages. The `builder` stage starts from `python:3.11-slim` and installs system build tools (gcc, g++, build-essential) needed to compile C extensions for numpy, scipy, and scikit-learn. All Python packages are installed with `pip install --prefix=/install` into an isolated directory. The `runtime` stage starts from a fresh `python:3.11-slim` with only `libgomp1` and `curl` (no compilers). It copies `/install` from the builder stage using `COPY --from=builder`. This produces an image with all compiled packages but without the build toolchain, reducing image size by ~60% and reducing attack surface. The `finsight` non-root user satisfies container security requirements in enterprise Kubernetes deployments.

---

## Q12: How would you scale this to 5,000 companies?

Four changes needed: First, move DuckDB to MotherDuck (cloud DuckDB) or ClickHouse — connection string change only in `database.py`. Second, parallelize ingestion using `asyncio.gather()` with the existing `httpx.AsyncClient` in `EdgarClient`, respecting SEC's 10 req/s rate limit using a semaphore. Third, batch the Isolation Forest training: instead of retraining on every run, train weekly on the full 5,000-company universe and score incrementally against the frozen model. Fourth, replace ChromaDB with Pinecone or Weaviate for the vector store — a 5,000-company filing corpus at 500 chunks per company is 2.5M vectors, beyond ChromaDB's performance sweet spot. The LCEL chain in `rag_chain.py` requires no changes — only the retriever instantiation in `vector_store.py` changes.

---

## Q13: What is the KQL integration and why is it relevant?

KQL (Kusto Query Language) is the query language for Azure Data Explorer, Azure Monitor, and Microsoft Sentinel. `governance/kql_queries.py` contains 9 pre-built queries targeting tables `FinSightRiskScores`, `FinSightAuditTrail`, `FinSightBenfordResults`, and `FinSightSHAPValues` — the ADX schema is defined in `KQL_TABLE_SCHEMA`. This is relevant because financial institutions that use Microsoft's security stack (Sentinel for SIEM, Azure Monitor for observability) can stream FinSight360's audit trail into their existing compliance dashboards without building custom connectors. The `audit_trail_summary` and `anomaly_trend_over_time` queries enable compliance teams to monitor model behavior at enterprise scale. It demonstrates awareness of where ML models actually land in production financial environments.

---

## Q14: How did you handle SEC EDGAR rate limiting?

SEC EDGAR's fair-use policy limits requests to 10 per second. `EdgarClient` in `ingestion/edgar_client.py` enforces a 0.12-second minimum delay between requests (slightly above the 0.10-second limit for safety margin) using `time.sleep()` in a synchronous context. The `tenacity` library wraps all HTTP calls with exponential backoff: on a 429 (rate limit) or 503 (service unavailable) response, the client waits 2, 4, 8... seconds up to a maximum of 5 attempts before raising. The `User-Agent` header includes the application name and a contact email per SEC's published requirements — requests without a compliant User-Agent are blocked. All of this is transparent to `FilingProcessor` — it calls `client.get_xbrl_facts(cik)` and gets a structured result back regardless of how many retries occurred.

---

## Q15: What responsible AI measures did you implement?

Four distinct layers. **Explainability**: SHAP TreeExplainer computes exact Shapley values for all 17 features per company; every dashboard score shows the top contributing factors; `governance/shap_explainer.py` stores results in DuckDB for audit access. **Bias monitoring**: `BiasAnalyzer` computes z-scores of mean risk scores by industry group and geography; groups more than 1.5 standard deviations from the overall mean are flagged — aligned to EU AI Act Article 10 requirements. **Audit trail**: `AuditTrailManager` logs every scoring event with a tamper-evident feature hash; records are immutable (`ON CONFLICT DO NOTHING`); satisfies SR 11-7 ongoing monitoring requirements. **Model card**: `generate_model_card()` produces a Mitchell et al. (2019)-format document including intended use, out-of-scope uses, caveats, performance metrics, and a regulatory alignment table (SR 11-7, EU AI Act, NIST AI RMF, SOC2).
