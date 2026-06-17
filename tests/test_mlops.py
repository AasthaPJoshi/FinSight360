"""
Tests for Phase 8: Production MLOps.
Covers: health checks, model monitor, model registry.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlops.health_check import HealthStatus, PipelineHealthChecker
from mlops.model_monitor import ModelMonitor
from mlops.registry import ModelRegistryManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db():
    """Minimal in-memory DB stub."""
    import duckdb

    class _InMemDB:
        def __init__(self):
            self.conn = duckdb.connect(":memory:")

        def get_connection(self):
            return self.conn

        def close(self):
            pass

    return _InMemDB()


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------

def test_health_checker_returns_list():
    checker = PipelineHealthChecker()
    results = checker.check_all()
    assert isinstance(results, list)
    assert len(results) >= 5


def test_health_status_fields():
    checker = PipelineHealthChecker()
    results = checker.check_all()
    for status in results:
        assert isinstance(status, HealthStatus)
        assert hasattr(status, "component")
        assert hasattr(status, "status")
        assert hasattr(status, "message")
        assert status.status in ("OK", "WARN", "ERROR")


def test_health_database_warn_when_missing(tmp_path):
    checker = PipelineHealthChecker()
    checker.db_path = str(tmp_path / "nonexistent.duckdb")
    result = checker.check_database()
    assert result.status in ("ERROR", "WARN")
    assert result.component == "DuckDB"


def test_health_model_warn_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checker = PipelineHealthChecker()
    result = checker.check_model_exists()
    assert result.status == "WARN"
    assert result.component == "ML Model"


def test_health_reference_files_ok():
    checker = PipelineHealthChecker()
    result = checker.check_reference_files()
    # In actual repo these files exist; just verify structure
    assert result.component == "Reference Files"
    assert result.status in ("OK", "WARN")


def test_health_check_mlflow_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checker = PipelineHealthChecker()
    result = checker.check_mlflow_runs()
    assert result.component == "MLflow"
    # No mlruns in tmp_path → WARN
    assert result.status == "WARN"


# ---------------------------------------------------------------------------
# Model Monitor tests
# ---------------------------------------------------------------------------

def test_model_monitor_no_data(tmp_db):
    monitor = ModelMonitor(tmp_db)
    result = monitor.compare_runs()
    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] in ("no_data", "ok", "alerts")


def test_model_monitor_with_final_scores(tmp_db):
    tmp_db.conn.execute("""
        CREATE TABLE final_risk_scores (
            cik VARCHAR,
            ticker VARCHAR,
            final_risk_score DOUBLE,
            is_anomaly BOOLEAN,
            final_risk_tier VARCHAR,
            computed_at TIMESTAMP DEFAULT current_timestamp
        )
    """)
    tmp_db.conn.execute("""
        INSERT INTO final_risk_scores VALUES
            ('001', 'AAPL', 65.0, false, 'MEDIUM', current_timestamp),
            ('002', 'MSFT', 72.0, true,  'HIGH',   current_timestamp),
            ('003', 'GOOG', 45.0, false, 'LOW',    current_timestamp)
    """)
    monitor = ModelMonitor(tmp_db)
    summary = monitor.get_run_summary()
    assert isinstance(summary, dict)
    # avg_score should be around 60.67
    assert "avg_score" in summary


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_registry_manager_no_crash():
    registry = ModelRegistryManager()
    versions = registry.get_latest_versions()
    assert isinstance(versions, list)


def test_registry_production_uri_returns_none_or_str():
    registry = ModelRegistryManager()
    uri = registry.get_production_model_uri()
    assert uri is None or isinstance(uri, str)
