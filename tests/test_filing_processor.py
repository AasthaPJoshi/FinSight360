"""Tests for the FilingProcessor orchestrator."""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from ingestion.filing_processor import FilingProcessor
from ingestion.models import Company, FinancialMetric
from storage.database import Database


MOCK_SUBMISSIONS = {
    "cik": "320193",
    "name": "Apple Inc.",
    "tickers": ["AAPL"],
    "sic": "3674",
    "sicDescription": "Semiconductors",
    "ein": "94-2404110",
    "stateOfIncorporation": "CA",
    "fiscalYearEnd": "0930",
    "exchanges": ["Nasdaq"],
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-23-000106"],
            "form": ["10-K"],
            "filingDate": ["2023-11-03"],
            "reportDate": ["2023-09-30"],
            "primaryDocument": ["aapl-20230930.htm"],
            "documentCount": [88],
        }
    },
}

MOCK_FACTS = {
    "cik": 320193,
    "entityName": "Apple Inc.",
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "label": "Revenue",
                "units": {
                    "USD": [
                        {
                            "accn": "0000320193-23-000106",
                            "start": "2022-10-01",
                            "end": "2023-09-30",
                            "val": 383285000000,
                            "form": "10-K",
                            "filed": "2023-11-03",
                        }
                    ]
                },
            },
            "NetIncomeLoss": {
                "label": "Net Income",
                "units": {
                    "USD": [
                        {
                            "accn": "0000320193-23-000106",
                            "start": "2022-10-01",
                            "end": "2023-09-30",
                            "val": 96995000000,
                            "form": "10-K",
                            "filed": "2023-11-03",
                        }
                    ]
                },
            },
        }
    },
}


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "test.duckdb")
    database = Database(db_path=path)
    yield database
    database.close()


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.resolve_cik.return_value = "320193"
    client.get_submissions.return_value = MOCK_SUBMISSIONS
    client.get_company_facts.return_value = MOCK_FACTS
    return client


def test_process_company(db, mock_client):
    """Processing a company should store it in the database."""
    processor = FilingProcessor(db=db, client=mock_client)
    with patch.object(db, "run_dbt_transformations", return_value={"returncode": 0, "stdout": ""}):
        result = processor.process_ticker("AAPL")

    assert result["status"] == "success"
    assert result["cik"] == "320193"

    company = db.get_company("320193")
    assert company is not None
    assert company["name"] == "Apple Inc."
    assert company["ticker"] == "AAPL"


def test_compute_metrics(db, mock_client):
    """After processing, financial metrics should be stored for the company."""
    processor = FilingProcessor(db=db, client=mock_client)
    with patch.object(db, "run_dbt_transformations", return_value={"returncode": 0, "stdout": ""}):
        processor.process_ticker("AAPL")

    metrics = db.get_metrics_for_company("320193")
    assert len(metrics) >= 1
    annual = [m for m in metrics if m["form"] == "10-K"]
    assert len(annual) >= 1
    assert annual[0]["revenues"] == pytest.approx(383285000000.0)
