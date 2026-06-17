"""Tests for the SEC EDGAR HTTP client."""
from unittest.mock import MagicMock, patch

import pytest

from ingestion.edgar_client import EdgarClient


@pytest.fixture
def client():
    return EdgarClient(delay=0.0)


def test_resolve_cik(client):
    """resolve_cik should return the CIK string for a known ticker."""
    mock_tickers = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"},
    }
    with patch.object(client, "_get", return_value=mock_tickers):
        cik = client.resolve_cik("AAPL")
    assert cik == "320193"


def test_resolve_cik_case_insensitive(client):
    """Ticker lookup must be case-insensitive."""
    mock_tickers = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    }
    with patch.object(client, "_get", return_value=mock_tickers):
        assert client.resolve_cik("aapl") == "320193"
        assert client.resolve_cik("Aapl") == "320193"


def test_resolve_cik_unknown_ticker(client):
    """resolve_cik should raise ValueError for an unknown ticker."""
    mock_tickers = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    }
    with patch.object(client, "_get", return_value=mock_tickers):
        with pytest.raises(ValueError, match="UNKNOWN"):
            client.resolve_cik("UNKNOWN")


def test_get_submissions(client):
    """get_submissions should return the mocked submissions JSON."""
    mock_data = {
        "cik": "320193",
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "filings": {"recent": {}},
    }
    with patch.object(client, "_get", return_value=mock_data) as mock_get:
        result = client.get_submissions("320193")
        assert result["name"] == "Apple Inc."
        mock_get.assert_called_once()
        assert "CIK0000320193" in mock_get.call_args[0][0]
