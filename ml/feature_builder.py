"""Builds the ML feature matrix from DuckDB mart tables."""
import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("feature_builder")

# Features used for Isolation Forest — chosen based on academic literature
# on financial fraud detection (Beneish M-score, Dechow et al. 2011).
ISOLATION_FOREST_FEATURES = [
    # Revenue quality signals
    "gross_margin", "operating_margin", "net_margin",
    "revenue_yoy_growth", "ar_to_revenue",
    # Cash flow quality
    "current_ratio", "debt_to_equity", "roe",
    # Rule-based risk signals (from dbt mart)
    "flag_ar_acceleration", "flag_cashflow_divergence",
    "flag_margin_compression", "flag_aggressive_recognition",
    "flag_high_leverage", "flag_liquidity_stress",
    "total_signal_count",
    # Filing behavior
    "late_filings_count", "max_reporting_lag_days",
]

# Schemas to try (dbt-duckdb prefixes with 'main_')
_MART_SCHEMAS = ["main_marts", "marts"]


def _query_mart(conn, table: str) -> pd.DataFrame:
    """Try multiple schema prefixes to find dbt mart tables."""
    for schema in _MART_SCHEMAS:
        try:
            return conn.execute(f"SELECT * FROM {schema}.{table}").df()
        except Exception:
            continue
    log.warning("mart_table_not_found", table=table)
    return pd.DataFrame()


class FeatureBuilder:
    def __init__(self, db):
        self.db = db

    def build_feature_matrix(self) -> pd.DataFrame:
        """
        Pull mart_risk_candidates from DuckDB and build ML feature matrix.
        Handles nulls, clips outliers, returns clean DataFrame.
        """
        conn = self.db.get_connection()
        df = _query_mart(conn, "mart_risk_candidates")

        if df.empty:
            log.warning("feature_matrix_empty")
            return df

        log.info("feature_matrix_loaded", rows=len(df), cols=len(df.columns))

        available = [f for f in ISOLATION_FOREST_FEATURES if f in df.columns]
        missing = [f for f in ISOLATION_FOREST_FEATURES if f not in df.columns]
        if missing:
            log.warning("features_missing", missing=missing)

        meta_cols = ["cik", "ticker", "company_name", "industry_group",
                     "risk_tier", "pre_ml_risk_score"]
        keep_cols = [c for c in meta_cols if c in df.columns] + available
        feature_df = df[keep_cols].copy()

        # Fill nulls: flags → 0, ratios → industry median
        flag_cols = [
            c for c in available
            if c.startswith("flag_")
            or c in ("total_signal_count", "late_filings_count")
        ]
        ratio_cols = [c for c in available if c not in flag_cols]

        feature_df[flag_cols] = feature_df[flag_cols].fillna(0)

        for col in ratio_cols:
            median = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(median)

        # Clip extreme outliers at 1st/99th percentile for ratio cols
        for col in ratio_cols:
            p1 = feature_df[col].quantile(0.01)
            p99 = feature_df[col].quantile(0.99)
            feature_df[col] = feature_df[col].clip(p1, p99)

        log.info("feature_matrix_ready",
                 rows=len(feature_df), feature_count=len(available))
        return feature_df

    def get_feature_names(self) -> list:
        return ISOLATION_FOREST_FEATURES

    def build_time_series_features(self) -> pd.DataFrame:
        """
        For each company, compute trend features across last 4 quarters:
        slope of gross_margin, ar_to_revenue, revenues, signal_count.
        """
        conn = self.db.get_connection()
        df = _query_mart(conn, "mart_financial_summary")

        if df.empty:
            return df

        # Filter to quarterly data
        df = df[df["form"] == "10-Q"].sort_values(["cik", "period_end"])

        trends = []
        for cik, group in df.groupby("cik"):
            last4 = group.tail(4)
            if len(last4) < 2:
                continue

            x = np.arange(len(last4))

            def slope(series):
                clean = series.fillna(series.median() if not series.isna().all() else 0)
                if len(clean) < 2:
                    return 0.0
                return float(np.polyfit(x[:len(clean)], clean, 1)[0])

            trends.append({
                "cik": cik,
                "ticker": group["ticker"].iloc[0],
                "company_name": group["company_name"].iloc[0],
                "gm_slope_4q": slope(last4["gross_margin"]),
                "om_slope_4q": slope(last4["operating_margin"]),
                "ar_slope_4q": slope(last4["ar_to_revenue"]),
                "revenue_slope_4q": slope(last4["revenues"]),
                "signal_count_trend": slope(last4["total_signal_count"]),
            })

        return pd.DataFrame(trends)
