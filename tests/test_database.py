"""Tests for DuckDB schema and database operations."""
import os
import tempfile
from datetime import datetime, date

import pytest

from storage.database import Database
from ingestion.models import Company, Filing, FinancialFact, FinancialMetric, IngestionRun


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "test.duckdb")
    database = Database(db_path=path)
    yield database
    database.close()


def test_database_creation(db):
    """Database object should be created without error."""
    assert db is not None
    assert db.conn is not None


def test_tables_exist(db):
    """All five required tables must be present after schema creation."""
    for table in ("companies", "filings", "financial_facts", "financial_metrics", "ingestion_runs"):
        assert db.table_exists(table), f"Table '{table}' not found"


def test_insert_company(db):
    """Inserting a company record and retrieving it should round-trip correctly."""
    company = Company(
        cik="320193",
        name="Apple Inc.",
        ticker="AAPL",
        sic_code="3674",
        sic_description="Semiconductors and Related Devices",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.upsert_company(company)
    result = db.get_company("320193")
    assert result is not None
    assert result["name"] == "Apple Inc."
    assert result["ticker"] == "AAPL"


def test_insert_filing(db):
    """Filing records must persist and be queryable."""
    company = Company(cik="320193", name="Apple Inc.", created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.upsert_company(company)

    filing = Filing(
        accession_number="0000320193-23-000106",
        cik="320193",
        form_type="10-K",
        filing_date=date(2023, 11, 3),
        period_of_report=date(2023, 9, 30),
        document_count=88,
        primary_document="aapl-20230930.htm",
        ingested_at=datetime.utcnow(),
    )
    db.upsert_filing(filing)

    rows = db.conn.execute(
        "SELECT * FROM filings WHERE accession_number = ?",
        ["0000320193-23-000106"],
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][2] == "10-K"


def test_insert_financial_fact(db):
    """Financial facts must be stored and retrievable by CIK."""
    company = Company(cik="320193", name="Apple Inc.", created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.upsert_company(company)

    fact = FinancialFact(
        id="test-fact-001",
        cik="320193",
        concept="NetIncomeLoss",
        period_end="2023-09-30",
        value=96995000000.0,
        unit="USD",
        form="10-K",
        created_at=datetime.utcnow(),
    )
    db.upsert_financial_fact(fact)

    rows = db.conn.execute(
        "SELECT value FROM financial_facts WHERE id = 'test-fact-001'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(96995000000.0)


def test_insert_financial_metrics(db):
    """Financial metrics must persist with all numeric fields."""
    company = Company(cik="320193", name="Apple Inc.", created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.upsert_company(company)

    metric = FinancialMetric(
        id="metric-001",
        cik="320193",
        period_end="2023-09-30",
        form="10-K",
        revenues=383285000000.0,
        net_income=96995000000.0,
        gross_profit=169148000000.0,
        computed_at=datetime.utcnow(),
    )
    db.upsert_financial_metric(metric)

    metrics = db.get_metrics_for_company("320193")
    assert len(metrics) == 1
    assert metrics[0]["revenues"] == pytest.approx(383285000000.0)
    assert metrics[0]["net_income"] == pytest.approx(96995000000.0)


def test_upsert_behavior(db):
    """Upserting the same company twice should update, not duplicate."""
    c1 = Company(cik="320193", name="Apple", ticker="AAPL", created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.upsert_company(c1)
    c2 = Company(cik="320193", name="Apple Inc.", ticker="AAPL", created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.upsert_company(c2)

    count = db.conn.execute("SELECT count(*) FROM companies WHERE cik = '320193'").fetchone()[0]
    result = db.get_company("320193")
    assert count == 1
    assert result["name"] == "Apple Inc."


def test_ingestion_run_lifecycle(db):
    """Ingestion run should transition from 'running' to 'completed'."""
    run = IngestionRun(id="run-001", started_at=datetime.utcnow(), status="running")
    db.upsert_ingestion_run(run)

    row = db.conn.execute(
        "SELECT status FROM ingestion_runs WHERE id = 'run-001'"
    ).fetchone()
    assert row[0] == "running"

    run.status = "completed"
    run.completed_at = datetime.utcnow()
    run.companies_processed = 1
    db.upsert_ingestion_run(run)

    row = db.conn.execute(
        "SELECT status, companies_processed FROM ingestion_runs WHERE id = 'run-001'"
    ).fetchone()
    assert row[0] == "completed"
    assert row[1] == 1


def test_indexes_created(db):
    """Five indexes must be present in the DuckDB catalog."""
    rows = db.conn.execute(
        "SELECT index_name FROM duckdb_indexes()"
    ).fetchall()
    index_names = {r[0] for r in rows}
    expected = {
        "idx_filings_cik",
        "idx_facts_cik",
        "idx_facts_period",
        "idx_metrics_cik",
        "idx_metrics_period",
    }
    assert expected.issubset(index_names), f"Missing indexes: {expected - index_names}"
