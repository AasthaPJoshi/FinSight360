"""
Cached data layer for Streamlit dashboard.
All DuckDB queries run here with @st.cache_data.
Cache TTL: 5 minutes (data refreshes after new pipeline run).
"""
import os

import duckdb
import pandas as pd
import streamlit as st

DB_PATH = os.environ.get(
    "DUCKDB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "finsight360.duckdb"),
)


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH, read_only=True)


@st.cache_data(ttl=300)
def load_final_risk_scores() -> pd.DataFrame:
    """Top-level risk table: all companies, all signals combined."""
    try:
        conn = get_connection()
        df = conn.execute("""
            SELECT
                f.cik, f.ticker, f.company_name,
                f.ml_risk_score, f.benford_risk_score,
                f.network_risk_score AS network_risk,
                f.final_risk_score, f.final_risk_tier,
                m.is_anomaly, m.anomaly_percentile, m.isolation_forest_score,
                b.conformity_rating, b.mad_score,
                r.dominant_signal, r.total_signal_count,
                r.flag_ar_acceleration, r.flag_cashflow_divergence,
                r.flag_margin_compression, r.flag_aggressive_recognition,
                r.flag_high_leverage, r.flag_liquidity_stress,
                r.latest_period, r.revenue_yoy_growth,
                r.gross_margin, r.operating_margin, r.current_ratio,
                r.debt_to_equity, r.ar_to_revenue,
                r.late_filings_count, r.has_late_filings,
                r.industry_group
            FROM final_risk_scores f
            LEFT JOIN ml_anomaly_scores m ON f.cik = m.cik
            LEFT JOIN benford_results b ON f.cik = b.cik
            LEFT JOIN main_marts.mart_risk_candidates r ON f.cik = r.cik
            ORDER BY f.final_risk_score DESC
        """).df()
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_financial_summary() -> pd.DataFrame:
    """Time-series financial data for trend charts."""
    try:
        conn = get_connection()
        df = conn.execute("""
            SELECT * FROM main_marts.mart_financial_summary
            ORDER BY cik, period_end
        """).df()
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_company_dashboard() -> pd.DataFrame:
    try:
        conn = get_connection()
        df = conn.execute("SELECT * FROM main_marts.mart_company_dashboard").df()
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_benford_results() -> pd.DataFrame:
    try:
        conn = get_connection()
        df = conn.execute("""
            SELECT * FROM benford_results
            ORDER BY benford_risk_score DESC
        """).df()
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_ingestion_runs() -> pd.DataFrame:
    try:
        conn = get_connection()
        df = conn.execute("""
            SELECT * FROM ingestion_runs
            ORDER BY started_at DESC LIMIT 10
        """).df()
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_risk_briefs() -> pd.DataFrame:
    try:
        conn = get_connection()
        df = conn.execute("""
            SELECT cik, company_name, final_risk_score,
                   generated_at,
                   LEFT(brief_text, 200) AS brief_preview
            FROM risk_briefs
            ORDER BY final_risk_score DESC
        """).df()
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_company_brief(cik: str) -> str:
    """Load full brief text for one company (not cached — on demand)."""
    try:
        conn = get_connection()
        result = conn.execute(
            "SELECT brief_text FROM risk_briefs WHERE cik = ?", [cik]
        ).df()
        conn.close()
        if not result.empty:
            return str(result.iloc[0]["brief_text"])
        return ""
    except Exception:
        return ""


def get_db_stats() -> dict:
    """Pipeline health stats for sidebar."""
    try:
        conn = get_connection()
        stats: dict = {}
        for table in [
            "companies", "filings", "financial_facts",
            "ml_anomaly_scores", "final_risk_scores",
        ]:
            try:
                count = conn.execute(
                    f"SELECT count(*) FROM {table}"
                ).fetchone()[0]
                stats[table] = count
            except Exception:
                stats[table] = 0
        conn.close()
        return stats
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_shap_global() -> pd.DataFrame:
    """Global SHAP feature importance across all companies."""
    try:
        conn = get_connection()
        df = conn.execute("""
            SELECT
                feature_name,
                friendly_name,
                avg(abs_shap_value) AS mean_abs_shap,
                avg(shap_value)     AS mean_shap,
                count(*)            AS n_companies
            FROM shap_values
            GROUP BY feature_name, friendly_name
            ORDER BY mean_abs_shap DESC
        """).df()
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_shap_company(cik: str) -> pd.DataFrame:
    """Per-company SHAP breakdown for explainability tab."""
    try:
        conn = get_connection()
        df = conn.execute("""
            SELECT
                feature_name, friendly_name, shap_value,
                feature_value, abs_shap_value, direction, rank_in_company
            FROM shap_values
            WHERE cik = ?
            ORDER BY abs_shap_value DESC
        """, [cik]).df()
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()
