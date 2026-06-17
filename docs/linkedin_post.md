# LinkedIn Post Template

---

**Post:**

I've spent the past few months building something I wish existed when I started studying financial data engineering: a fully open-source, end-to-end financial anomaly detection platform.

**FinSight360** ingests SEC EDGAR 10-K filings, runs three independent risk signals, and delivers explainable scores — all from public data, all local, no vendor lock-in.

Here's what the system actually does:

→ Pulls structured XBRL financial data from SEC's public API for any public company
→ Runs dbt transformations to compute financial ratios and flag risk signals
→ Scores companies with Isolation Forest (17 features), Benford's Law (digit forensics), and corporate network risk (Louvain community detection on auditor-sharing graphs)
→ Explains every decision with SHAP — no black boxes
→ Generates LLM risk briefs via a LangChain RAG pipeline over the actual 10-K text
→ Logs every scoring event in an SR 11-7 compliant audit trail with tamper-evident feature hashes

The governance layer is what I'm most proud of. Bias analysis, model cards (Mitchell et al. 2019), KQL queries for Azure Sentinel integration, EU AI Act Article 10 compliance — built in from the start, not bolted on.

A few numbers that show it's production-serious:
- **69 tests** across all 8 modules (0 skipped, 0 mocked business logic)
- **10 phases** from raw ingestion to full MLOps (Docker, GitHub Actions CI/CD, MLflow registry, drift detection)
- **7-page Streamlit dashboard** — executive overview through per-company SHAP waterfall charts
- 100% free data sources (SEC EDGAR public API)

I built this to answer a question I kept running into: *What does a production-grade ML system in finance actually look like, built by one person?*

The answer involves a lot of choices that aren't in tutorials — RobustScaler over StandardScaler for heavy-tailed financial data, DuckDB over PostgreSQL for in-process analytical queries, TreeExplainer over KernelExplainer for exact SHAP values, append-only audit tables with ON CONFLICT DO NOTHING for regulatory immutability.

If you work in fintech, quant research, risk analytics, or data engineering — I'd love to hear what you'd build next on top of this foundation.

GitHub link in comments. Happy to chat about any of the technical decisions.

What's one signal you'd add to catch financial manipulation that most people don't think about?

---

#DataEngineering #MachineLearning #FinTech #OpenSource #Python #MLOps #ResponsibleAI #SEC #QuantFinance #DataScience
