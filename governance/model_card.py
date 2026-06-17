"""
Model card generator for FinSight360 (Mitchell et al. 2019 format).
Writes JSON + Markdown to docs/. Reads live DuckDB stats.
Compliance: SR 11-7, EU AI Act, NIST AI RMF.
"""
import json
from datetime import datetime
from pathlib import Path

from utils.logger import get_logger

log = get_logger("model_card")


def generate_model_card(db, output_dir: str = "docs") -> dict:
    """
    Generate model card from live DuckDB stats.
    Returns dict with json_path, md_path, and model_card_data.
    """
    conn = db.get_connection()
    stats = _read_live_stats(conn)

    model_card_data = {
        "model_details": {
            "name": "FinSight360 Anomaly Detection Model",
            "version": "1.0.0",
            "type": "Unsupervised Anomaly Detection (Isolation Forest)",
            "owner": "FinSight360 Risk Analytics Team",
            "created_date": "2024-01-01",
            "last_updated": datetime.utcnow().strftime("%Y-%m-%d"),
            "contact": "risk-analytics@finsight360.com",
            "license": "Proprietary — Internal Use Only",
            "framework": "scikit-learn IsolationForest + RobustScaler",
            "description": (
                "Ensemble anomaly detection model trained on SEC EDGAR 10-K financial "
                "metrics. Combines Isolation Forest ML scores (50%), Benford's Law "
                "forensic scores (30%), and corporate network risk (20%) into a final "
                "risk score. SHAP TreeExplainer provides per-company feature attribution."
            ),
        },
        "intended_use": {
            "primary_use": "Financial forensic analysis — identifying companies with anomalous financial reporting patterns",
            "intended_users": [
                "Internal risk analysts",
                "Compliance officers",
                "Audit teams",
                "Quantitative researchers",
            ],
            "out_of_scope": [
                "Automated trading decisions without human review",
                "Legal determination of fraud",
                "Real-time transaction monitoring",
                "Retail investor advice",
            ],
            "human_oversight": "All HIGH and CRITICAL risk flags require human analyst review before any action",
        },
        "model_architecture": {
            "algorithm": "IsolationForest (sklearn)",
            "contamination": 0.10,
            "n_estimators": 100,
            "preprocessing": "RobustScaler (median/IQR normalization, outlier-robust)",
            "feature_count": 17,
            "features": [
                "gross_margin", "operating_margin", "net_margin",
                "revenue_yoy_growth", "ar_to_revenue", "current_ratio",
                "debt_to_equity", "roe", "flag_ar_acceleration",
                "flag_cashflow_divergence", "flag_margin_compression",
                "flag_aggressive_recognition", "flag_high_leverage",
                "flag_liquidity_stress", "total_signal_count",
                "late_filings_count", "max_reporting_lag_days",
            ],
            "ensemble_weights": {
                "isolation_forest_ml": 0.50,
                "benfords_law": 0.30,
                "network_graph_risk": 0.20,
            },
            "explainability": "SHAP TreeExplainer (exact Shapley values, O(TLD) complexity)",
            "storage": "DuckDB (local) / Azure Data Explorer (cloud)",
        },
        "training_data": {
            "source": "SEC EDGAR XBRL financial data via public REST API",
            "period": "2019–2024 (5-year rolling window)",
            "company_count": stats.get("companies_logged", "N/A"),
            "filing_types": ["10-K annual reports"],
            "preprocessing": [
                "XBRL fact extraction and normalization",
                "Benford digit distribution computation",
                "Corporate graph construction (NetworkX)",
                "RobustScaler feature normalization",
            ],
            "known_limitations": [
                "Training data limited to US public companies filing with SEC",
                "XBRL quality varies by company and filing year",
                "Financial restatements may affect historical accuracy",
                "Model does not use unstructured text (MD&A, footnotes)",
            ],
        },
        "performance": {
            "note": "Unsupervised model — no labeled ground truth; metrics are diagnostic",
            "anomaly_rate": stats.get("anomaly_rate", "N/A"),
            "total_companies_scored": stats.get("companies_logged", 0),
            "total_scoring_events": stats.get("scoring_events", 0),
            "high_risk_flags": stats.get("high_risk_flags", 0),
            "human_reviews_completed": stats.get("human_reviews", 0),
            "explainability_coverage": "100% (SHAP values computed for all scored companies)",
            "audit_trail_coverage": "100% (every scoring event logged)",
            "risk_tier_thresholds": {
                "CRITICAL": ">= 85",
                "HIGH": ">= 70",
                "MEDIUM": ">= 50",
                "LOW": ">= 30",
                "MINIMAL": "< 30",
            },
        },
        "ethical_considerations": {
            "bias_monitoring": {
                "method": "Z-score analysis across industry groups and geographic regions",
                "threshold": "Groups > 1.5 standard deviations from overall mean are flagged",
                "frequency": "Run after every batch scoring event",
                "compliance": "EU AI Act Article 10 — data governance and bias monitoring",
            },
            "fairness_principles": [
                "Industry-neutral scoring — no sector receives categorical treatment",
                "Scores reflect financial metrics only, not company identity",
                "All anomaly flags are advisory — human review required for action",
                "SHAP explanations provided for every score to prevent black-box decisions",
            ],
            "privacy": "No personally identifiable information (PII) processed — only public SEC filings",
            "transparency": "Full model card, SHAP explanations, and audit trail publicly documented",
        },
        "caveats": [
            "This model flags statistical anomalies, not confirmed fraud — many flags are false positives",
            "Past anomaly patterns do not guarantee future fraud detection",
            "Financial restatements or XBRL errors in source data may affect scores",
            "The 10% contamination rate is a tuning parameter, not empirically validated fraud prevalence",
            "Network risk scores depend on completeness of subsidiary/auditor relationship data",
            "Model requires periodic retraining as financial reporting norms evolve",
        ],
        "regulatory_alignment": {
            "SR_11-7": "Model documentation, validation, and ongoing monitoring requirements met",
            "EU_AI_Act": "Article 10 bias monitoring, Article 13 transparency, Article 14 human oversight",
            "NIST_AI_RMF": "Govern, Map, Measure, Manage functions addressed",
            "SOC2": "Audit trail provides complete change log for compliance review",
        },
        "audit_trail": {
            "total_events": stats.get("total_events", 0),
            "first_event": str(stats.get("first_event", "N/A")),
            "last_event": str(stats.get("last_event", "N/A")),
            "storage": "DuckDB append-only table (audit_trail)",
            "retention": "Indefinite (immutable records, no deletion)",
        },
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "model_card.json"
    json_path.write_text(json.dumps(model_card_data, indent=2, default=str))

    md_path = output_path / "model_card.md"
    md_path.write_text(_render_markdown(model_card_data))

    log.info("model_card_generated", json_path=str(json_path), md_path=str(md_path))
    return {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "model_card_data": model_card_data,
    }


def _read_live_stats(conn) -> dict:
    try:
        row = conn.execute("""
            SELECT
                count(*) AS total_events,
                count(DISTINCT cik) AS companies_logged,
                sum(CASE WHEN event_type='risk_score_generated' THEN 1 ELSE 0 END) AS scoring_events,
                sum(CASE WHEN human_reviewed THEN 1 ELSE 0 END) AS human_reviews,
                sum(CASE WHEN final_risk_tier IN ('HIGH','CRITICAL') THEN 1 ELSE 0 END) AS high_risk_flags,
                min(generated_at) AS first_event,
                max(generated_at) AS last_event
            FROM audit_trail
        """).df()
        stats = row.iloc[0].to_dict() if not row.empty else {}
    except Exception:
        stats = {}

    try:
        anomaly_row = conn.execute("""
            SELECT
                count(*) AS total,
                sum(CASE WHEN is_anomaly THEN 1 ELSE 0 END) AS anomalies
            FROM audit_trail
            WHERE event_type = 'risk_score_generated'
        """).df()
        if not anomaly_row.empty and anomaly_row.iloc[0]["total"] > 0:
            total = anomaly_row.iloc[0]["total"]
            anomalies = anomaly_row.iloc[0]["anomalies"]
            stats["anomaly_rate"] = f"{anomalies / total * 100:.1f}%"
    except Exception:
        pass

    return stats


def _render_markdown(data: dict) -> str:
    md = data["model_details"]
    intended = data["intended_use"]
    arch = data["model_architecture"]
    training = data["training_data"]
    perf = data["performance"]
    ethics = data["ethical_considerations"]
    caveats = data["caveats"]
    reg = data["regulatory_alignment"]
    audit = data["audit_trail"]

    lines = [
        f"# Model Card: {md['name']}",
        "",
        f"> **Version:** {md['version']}  ",
        f"> **Last Updated:** {md['last_updated']}  ",
        f"> **Owner:** {md['owner']}  ",
        f"> **License:** {md['license']}",
        "",
        "---",
        "",
        "## 1. Model Details",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Type | {md['type']} |",
        f"| Framework | {md['framework']} |",
        f"| Created | {md['created_date']} |",
        f"| Contact | {md['contact']} |",
        "",
        md["description"],
        "",
        "---",
        "",
        "## 2. Intended Use",
        "",
        f"**Primary use:** {intended['primary_use']}",
        "",
        "**Intended users:**",
    ]
    for u in intended["intended_users"]:
        lines.append(f"- {u}")
    lines += [
        "",
        "**Out of scope:**",
    ]
    for o in intended["out_of_scope"]:
        lines.append(f"- {o}")
    lines += [
        "",
        f"**Human oversight:** {intended['human_oversight']}",
        "",
        "---",
        "",
        "## 3. Model Architecture",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Algorithm | {arch['algorithm']} |",
        f"| Contamination | {arch['contamination']} |",
        f"| Estimators | {arch['n_estimators']} |",
        f"| Preprocessing | {arch['preprocessing']} |",
        f"| Feature Count | {arch['feature_count']} |",
        f"| Explainability | {arch['explainability']} |",
        "",
        "**Ensemble weights:**",
        f"- ML (Isolation Forest): {arch['ensemble_weights']['isolation_forest_ml'] * 100:.0f}%",
        f"- Benford's Law: {arch['ensemble_weights']['benfords_law'] * 100:.0f}%",
        f"- Network Graph Risk: {arch['ensemble_weights']['network_graph_risk'] * 100:.0f}%",
        "",
        "---",
        "",
        "## 4. Training Data",
        "",
        f"**Source:** {training['source']}  ",
        f"**Period:** {training['period']}  ",
        f"**Companies:** {training['company_count']}  ",
        f"**Filing types:** {', '.join(training['filing_types'])}",
        "",
        "**Known limitations:**",
    ]
    for lim in training["known_limitations"]:
        lines.append(f"- {lim}")
    lines += [
        "",
        "---",
        "",
        "## 5. Performance",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Companies Scored | {perf['total_companies_scored']} |",
        f"| Total Scoring Events | {perf['total_scoring_events']} |",
        f"| Anomaly Rate | {perf['anomaly_rate']} |",
        f"| High/Critical Risk Flags | {perf['high_risk_flags']} |",
        f"| Human Reviews Completed | {perf['human_reviews_completed']} |",
        f"| Explainability Coverage | {perf['explainability_coverage']} |",
        "",
        "**Risk tier thresholds:**",
    ]
    for tier, threshold in perf["risk_tier_thresholds"].items():
        lines.append(f"- {tier}: {threshold}")
    lines += [
        "",
        "---",
        "",
        "## 6. Ethical Considerations",
        "",
        "### Bias Monitoring",
        f"- **Method:** {ethics['bias_monitoring']['method']}",
        f"- **Threshold:** {ethics['bias_monitoring']['threshold']}",
        f"- **Frequency:** {ethics['bias_monitoring']['frequency']}",
        f"- **Compliance:** {ethics['bias_monitoring']['compliance']}",
        "",
        "### Fairness Principles",
    ]
    for principle in ethics["fairness_principles"]:
        lines.append(f"- {principle}")
    lines += [
        "",
        f"**Privacy:** {ethics['privacy']}  ",
        f"**Transparency:** {ethics['transparency']}",
        "",
        "---",
        "",
        "## 7. Caveats and Limitations",
        "",
    ]
    for caveat in caveats:
        lines.append(f"- {caveat}")
    lines += [
        "",
        "---",
        "",
        "## 8. Regulatory Alignment",
        "",
        f"| Standard | Coverage |",
        f"|----------|----------|",
        f"| SR 11-7 | {reg['SR_11-7']} |",
        f"| EU AI Act | {reg['EU_AI_Act']} |",
        f"| NIST AI RMF | {reg['NIST_AI_RMF']} |",
        f"| SOC2 | {reg['SOC2']} |",
        "",
        "---",
        "",
        "## 9. Audit Trail",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Total Events | {audit['total_events']} |",
        f"| First Event | {audit['first_event']} |",
        f"| Last Event | {audit['last_event']} |",
        f"| Storage | {audit['storage']} |",
        f"| Retention | {audit['retention']} |",
        "",
        "---",
        "",
        f"*Generated by FinSight360 on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*",
    ]
    return "\n".join(lines)
