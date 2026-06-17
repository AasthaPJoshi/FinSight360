"""
SHAP explainability for anomaly detection results.
Answers: WHY is this company flagged as high-risk?
"""
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
from pathlib import Path
from utils.logger import get_logger
from ml.anomaly_detector import AnomalyDetector

log = get_logger("explainer")

PLOTS_DIR = Path("reports/shap_plots")

FEATURE_LABELS = {
    "flag_ar_acceleration": "accounts receivable growing faster than revenue",
    "flag_cashflow_divergence": "cash flow diverging from net income",
    "flag_margin_compression": "sustained gross margin compression",
    "flag_aggressive_recognition": "aggressive revenue recognition pattern",
    "flag_high_leverage": "high debt-to-equity ratio",
    "flag_liquidity_stress": "current ratio below 1.0 (liquidity risk)",
    "ar_to_revenue": "elevated AR-to-revenue ratio",
    "debt_to_equity": "leverage level",
    "gross_margin": "gross margin level",
    "revenue_yoy_growth": "revenue growth rate",
    "total_signal_count": "multiple concurrent risk signals",
    "late_filings_count": "history of late SEC filings",
    "max_reporting_lag_days": "filing lag severity",
}


class AnomalyExplainer:
    """
    Uses TreeExplainer (fast, exact) for Isolation Forest SHAP values.
    Generates per-company waterfall charts showing which features
    push risk score up or down.
    """

    def __init__(self, detector: AnomalyDetector):
        self.detector = detector
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        # TreeExplainer works directly with sklearn IsolationForest
        self.explainer = shap.TreeExplainer(
            detector.pipeline.named_steps["isolation_forest"]
        )

    def explain_company(self, feature_df: pd.DataFrame, cik: str) -> dict:
        """Generate SHAP explanation for a single company."""
        row = feature_df[feature_df["cik"] == cik]
        if row.empty:
            log.warning("company_not_found_for_shap", cik=cik)
            return {}

        X_raw = row[self.detector.feature_cols].values
        X_scaled = self.detector.pipeline.named_steps["scaler"].transform(X_raw)

        shap_values = self.explainer.shap_values(X_scaled)

        # Handle different SHAP output shapes (2D array or list)
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values

        if sv.ndim == 2:
            sv = sv[0]

        explanation = {
            col: float(sv[i])
            for i, col in enumerate(self.detector.feature_cols)
        }

        # Sort by absolute impact descending
        return dict(
            sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)
        )

    def generate_waterfall_plot(self, feature_df: pd.DataFrame,
                                cik: str, company_name: str) -> str:
        """Save a SHAP waterfall chart for a company. Returns file path."""
        explanation = self.explain_company(feature_df, cik)
        if not explanation:
            return ""

        features = list(explanation.keys())[:10]
        values = [explanation[f] for f in features]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#d32f2f" if v > 0 else "#388e3c" for v in values]
        ax.barh(features, values, color=colors, edgecolor="none")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on anomaly score)")
        ax.set_title(
            f"Risk Factor Explanation: {company_name}\n"
            f"Red = increases risk, Green = decreases risk"
        )
        ax.invert_yaxis()
        plt.tight_layout()

        filepath = PLOTS_DIR / f"{cik}_shap_waterfall.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        log.info("shap_plot_saved", cik=cik, path=str(filepath))
        return str(filepath)

    def generate_summary_explanation(self, explanation: dict,
                                     company_name: str,
                                     ml_risk_score: float) -> str:
        """Generate plain-English risk narrative from SHAP values."""
        if not explanation:
            return "Insufficient data for explanation."

        top_risk_factors = [(k, v) for k, v in explanation.items() if v > 0][:3]
        mitigating = [(k, v) for k, v in explanation.items() if v < 0][:2]

        risk_text = "; ".join([
            FEATURE_LABELS.get(k, k.replace("_", " "))
            for k, _ in top_risk_factors
        ]) or "none identified"

        mitigation_text = "; ".join([
            FEATURE_LABELS.get(k, k.replace("_", " "))
            for k, _ in mitigating
        ]) if mitigating else "none identified"

        escalate = (
            "Recommend escalation for detailed analyst review."
            if ml_risk_score > 70
            else "Monitor for deterioration next quarter."
        )

        return (
            f"{company_name} scores {ml_risk_score:.0f}/100 on the FinSight360 "
            f"ML risk model. Primary risk drivers: {risk_text}. "
            f"Mitigating factors: {mitigation_text}. {escalate}"
        )
