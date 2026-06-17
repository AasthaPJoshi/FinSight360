"""
SHAP (SHapley Additive exPlanations) integration for FinSight360.

SHAP answers: WHY did this company receive this risk score?
Uses TreeExplainer for Isolation Forest (fast + exact).
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from utils.logger import get_logger

log = get_logger("shap_explainer")

SHAP_PLOTS_DIR = Path("reports/shap_plots")
SHAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FRIENDLY_NAMES = {
    "gross_margin": "Gross Margin",
    "operating_margin": "Operating Margin",
    "net_margin": "Net Margin",
    "revenue_yoy_growth": "Revenue YoY Growth",
    "ar_to_revenue": "AR / Revenue Ratio",
    "current_ratio": "Current Ratio",
    "debt_to_equity": "Debt / Equity",
    "roe": "Return on Equity",
    "flag_ar_acceleration": "AR Acceleration Flag",
    "flag_cashflow_divergence": "Cash Flow Divergence Flag",
    "flag_margin_compression": "Margin Compression Flag",
    "flag_aggressive_recognition": "Aggressive Rev. Recognition",
    "flag_high_leverage": "High Leverage Flag",
    "flag_liquidity_stress": "Liquidity Stress Flag",
    "total_signal_count": "Total Risk Signal Count",
    "late_filings_count": "Late Filings Count",
    "max_reporting_lag_days": "Max Reporting Lag (Days)",
}


class SHAPExplainer:
    """Generates SHAP explanations for FinSight360 Isolation Forest."""

    def __init__(self, db):
        self.db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        conn = self.db.get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shap_values (
                cik              VARCHAR NOT NULL,
                feature_name     VARCHAR NOT NULL,
                friendly_name    VARCHAR,
                shap_value       DOUBLE,
                feature_value    DOUBLE,
                abs_shap_value   DOUBLE,
                direction        VARCHAR,
                rank_in_company  INTEGER,
                computed_at      TIMESTAMP DEFAULT current_timestamp,
                PRIMARY KEY (cik, feature_name)
            )
        """)

    def explain_all_companies(
        self,
        feature_df: pd.DataFrame,
        feature_cols: list[str],
        model_path: str = "models/isolation_forest.joblib",
    ) -> pd.DataFrame:
        """Compute SHAP values for all companies. Returns per-feature DataFrame."""
        if not Path(model_path).exists():
            log.warning("model_not_found", path=model_path)
            return pd.DataFrame()

        model_data = joblib.load(model_path)
        pipeline = model_data["pipeline"]

        scaler = pipeline.named_steps["scaler"]
        X_scaled = scaler.transform(feature_df[feature_cols].values)

        iso_forest = pipeline.named_steps["isolation_forest"]
        explainer = shap.TreeExplainer(iso_forest)
        shap_values = explainer.shap_values(X_scaled)

        log.info(
            "shap_computed",
            shape=shap_values.shape,
            companies=len(feature_df),
        )

        rows = []
        conn = self.db.get_connection()

        for i, (_, company_row) in enumerate(
            feature_df.reset_index(drop=True).iterrows()
        ):
            cik = str(company_row["cik"])
            company_shap = shap_values[i]
            sorted_indices = np.argsort(np.abs(company_shap))[::-1]

            for rank, feat_idx in enumerate(sorted_indices):
                if feat_idx >= len(feature_cols):
                    continue
                feat_name = feature_cols[feat_idx]
                shap_val = float(company_shap[feat_idx])
                feat_val = float(company_row.get(feat_name, 0))
                friendly = FEATURE_FRIENDLY_NAMES.get(
                    feat_name, feat_name.replace("_", " ").title()
                )

                row = {
                    "cik": cik,
                    "feature_name": feat_name,
                    "friendly_name": friendly,
                    "shap_value": round(shap_val, 6),
                    "feature_value": round(feat_val, 4),
                    "abs_shap_value": round(abs(shap_val), 6),
                    "direction": "Risk ↑" if shap_val > 0 else "Risk ↓",
                    "rank_in_company": rank + 1,
                }
                rows.append(row)

                conn.execute(
                    """
                    INSERT INTO shap_values
                        (cik, feature_name, friendly_name, shap_value,
                         feature_value, abs_shap_value, direction,
                         rank_in_company, computed_at)
                    VALUES (?,?,?,?,?,?,?,?,current_timestamp)
                    ON CONFLICT (cik, feature_name) DO UPDATE SET
                        shap_value = excluded.shap_value,
                        feature_value = excluded.feature_value,
                        abs_shap_value = excluded.abs_shap_value,
                        direction = excluded.direction,
                        rank_in_company = excluded.rank_in_company,
                        computed_at = excluded.computed_at
                    """,
                    [
                        cik, feat_name, friendly, row["shap_value"],
                        row["feature_value"], row["abs_shap_value"],
                        row["direction"], row["rank_in_company"],
                    ],
                )

        log.info("shap_values_stored", rows=len(rows))
        return pd.DataFrame(rows)

    def get_company_explanation(
        self, cik: str, top_n: int = 10
    ) -> pd.DataFrame:
        conn = self.db.get_connection()
        try:
            return conn.execute(
                """
                SELECT feature_name, friendly_name, shap_value,
                       feature_value, abs_shap_value, direction,
                       rank_in_company
                FROM shap_values
                WHERE cik = ?
                ORDER BY rank_in_company
                LIMIT ?
                """,
                [cik, top_n],
            ).df()
        except Exception:
            return pd.DataFrame()

    def plot_waterfall(self, cik: str, company_name: str) -> str:
        """Generate and save SHAP waterfall chart PNG. Returns file path."""
        shap_df = self.get_company_explanation(cik, top_n=12)
        if shap_df.empty:
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))
        features = shap_df["friendly_name"].tolist()
        values = shap_df["shap_value"].tolist()
        colors = ["#C62828" if v > 0 else "#2E7D32" for v in values]

        ax.barh(
            features[::-1],
            values[::-1],
            color=colors[::-1],
            edgecolor="none",
            height=0.6,
        )
        ax.axvline(x=0, color="#37474F", linewidth=1.0)
        for i, val in enumerate(values[::-1]):
            ax.text(
                val + (0.001 if val >= 0 else -0.001),
                i,
                f"{val:+.4f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9,
            )

        ax.set_xlabel("SHAP Value (impact on anomaly score)", fontsize=11)
        ax.set_title(
            f"FinSight360 Risk Explanation: {company_name}\n"
            "Red = increases anomaly risk | Green = reduces risk",
            fontsize=12,
            pad=15,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        filepath = SHAP_PLOTS_DIR / f"{cik}_waterfall.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        log.info("shap_waterfall_saved", cik=cik, path=str(filepath))
        return str(filepath)

    def plot_summary(self, shap_df_all: pd.DataFrame) -> str:
        """Global feature importance bar chart. Returns file path."""
        if shap_df_all.empty:
            return ""

        importance = (
            shap_df_all.groupby("friendly_name")["abs_shap_value"]
            .mean()
            .sort_values(ascending=True)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            importance.index,
            importance.values,
            color="#1565C0",
            edgecolor="none",
            height=0.6,
        )
        ax.set_xlabel(
            "Mean |SHAP Value| (average impact on risk score)", fontsize=11
        )
        ax.set_title(
            "FinSight360 — Global Feature Importance\n"
            "Average impact of each feature across all companies",
            fontsize=12,
            pad=15,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        filepath = SHAP_PLOTS_DIR / "global_feature_importance.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        log.info("shap_summary_saved", path=str(filepath))
        return str(filepath)
