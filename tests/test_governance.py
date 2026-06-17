"""
Tests for Phase 7: Responsible AI Governance Layer.
Covers: audit trail, bias analysis, model card, KQL queries, SHAP mappings.
"""
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from governance.audit_trail import AuditEntry, AuditTrailManager
from governance.bias_analyzer import BiasAnalyzer
from governance.kql_queries import KQL_QUERIES, export_kql_file, get_all_queries
from governance.shap_explainer import FEATURE_FRIENDLY_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db():
    """In-memory DuckDB via Database wrapper."""
    import duckdb
    from storage.database import Database

    class _InMemDB:
        def __init__(self):
            self.conn = duckdb.connect(":memory:")

        def get_connection(self):
            return self.conn

        def close(self):
            pass

    return _InMemDB()


# ---------------------------------------------------------------------------
# Audit Trail tests
# ---------------------------------------------------------------------------

def test_audit_entry_creates_valid_id():
    entry = AuditEntry()
    assert len(entry.audit_id) == 36  # UUID4 format
    assert "-" in entry.audit_id
    assert entry.event_type == "risk_score_generated"


def test_audit_trail_table_created(tmp_db):
    manager = AuditTrailManager(tmp_db)
    # Table should exist after init
    result = tmp_db.conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'audit_trail'"
    ).fetchone()
    assert result[0] == 1


def test_log_scoring_event_inserts_row(tmp_db):
    manager = AuditTrailManager(tmp_db)
    entry = AuditEntry(
        event_type="risk_score_generated",
        cik="0001234567",
        ticker="TEST",
        company_name="Test Corp",
        ml_risk_score=65.0,
        final_risk_score=72.5,
        final_risk_tier="HIGH",
        is_anomaly=True,
    )
    audit_id = manager.log_scoring_event(entry)
    assert audit_id == entry.audit_id

    rows = tmp_db.conn.execute(
        "SELECT * FROM audit_trail WHERE audit_id = ?", [audit_id]
    ).df()
    assert len(rows) == 1
    assert rows.iloc[0]["ticker"] == "TEST"
    assert rows.iloc[0]["final_risk_tier"] == "HIGH"
    assert rows.iloc[0]["is_anomaly"] == True


# ---------------------------------------------------------------------------
# Bias Analyzer tests
# ---------------------------------------------------------------------------

def test_bias_analyzer_no_bias():
    analyzer = BiasAnalyzer()
    df = pd.DataFrame({
        "industry_group": ["Tech", "Finance", "Healthcare", "Energy", "Tech",
                           "Finance", "Healthcare", "Energy"],
        "final_risk_score": [50.0, 51.0, 49.0, 50.5, 48.5, 52.0, 50.0, 49.5],
    })
    report = analyzer.analyze_by_group(df, "industry_group")
    assert report.bias_severity == "NONE"
    assert report.bias_detected is False
    assert len(report.flagged_groups) == 0


def test_bias_analyzer_detects_skew():
    analyzer = BiasAnalyzer()
    # Small "Outlier" group at extreme scores vs many "Normal" companies at 50
    # Creates a high z-score for Outlier group (score 100 vs mean ~55, std ~15)
    df = pd.DataFrame({
        "industry_group": (["Outlier"] * 3) + (["Normal"] * 27),
        "final_risk_score": ([100.0] * 3) + ([50.0] * 27),
    })
    report = analyzer.analyze_by_group(df, "industry_group")
    assert report.bias_detected is True
    assert report.bias_severity in ("LOW", "MEDIUM", "HIGH")
    assert len(report.flagged_groups) > 0


# ---------------------------------------------------------------------------
# Model Card tests
# ---------------------------------------------------------------------------

def test_model_card_generates_files(tmp_db):
    from governance.model_card import generate_model_card

    # Create a minimal audit_trail table so live stats don't error
    tmp_db.conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_trail (
            audit_id VARCHAR PRIMARY KEY,
            event_type VARCHAR,
            cik VARCHAR,
            ticker VARCHAR,
            company_name VARCHAR,
            period_of_report VARCHAR,
            model_name VARCHAR,
            model_version VARCHAR,
            mlflow_run_id VARCHAR,
            ml_risk_score DOUBLE,
            benford_risk_score DOUBLE,
            network_risk DOUBLE,
            final_risk_score DOUBLE,
            final_risk_tier VARCHAR,
            is_anomaly BOOLEAN,
            top_features TEXT,
            total_signals INTEGER,
            generated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            pipeline_version VARCHAR,
            data_hash VARCHAR,
            human_reviewed BOOLEAN DEFAULT false,
            human_reviewer VARCHAR,
            review_action VARCHAR,
            review_notes TEXT
        )
    """)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = generate_model_card(tmp_db, output_dir=tmpdir)

    assert "json_path" in result
    assert "md_path" in result
    assert "model_card_data" in result

    data = result["model_card_data"]
    assert "model_details" in data
    assert "intended_use" in data
    assert "ethical_considerations" in data
    assert "regulatory_alignment" in data
    assert data["model_details"]["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# KQL Query tests
# ---------------------------------------------------------------------------

def test_kql_all_queries_non_empty():
    queries = get_all_queries()
    assert len(queries) >= 9
    for name, query in queries.items():
        assert isinstance(query, str)
        assert len(query.strip()) > 0, f"Query '{name}' is empty"
        assert "FinSight" in query, f"Query '{name}' missing FinSight table reference"


def test_kql_export_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/test_queries.kql"
        result_path = export_kql_file(output_path)
        assert Path(result_path).exists()
        content = Path(result_path).read_text()
        assert "FinSight360" in content
        assert len(content) > 500


# ---------------------------------------------------------------------------
# SHAP Feature Name tests
# ---------------------------------------------------------------------------

def test_shap_feature_names_mapping():
    expected_features = [
        "gross_margin", "operating_margin", "net_margin",
        "revenue_yoy_growth", "ar_to_revenue", "current_ratio",
        "debt_to_equity", "roe",
        "flag_ar_acceleration", "flag_cashflow_divergence",
        "flag_margin_compression", "flag_aggressive_recognition",
        "flag_high_leverage", "flag_liquidity_stress",
        "total_signal_count", "late_filings_count", "max_reporting_lag_days",
    ]
    for feat in expected_features:
        assert feat in FEATURE_FRIENDLY_NAMES, f"Missing friendly name for '{feat}'"
        friendly = FEATURE_FRIENDLY_NAMES[feat]
        assert isinstance(friendly, str) and len(friendly) > 0
