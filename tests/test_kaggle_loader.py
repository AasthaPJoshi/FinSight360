"""Tests for ingestion/kaggle_loader.py."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ingestion.kaggle_loader import KaggleDataLoader, SECTOR_SIC, _metric_id, _accession


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_companies_csv() -> pd.DataFrame:
    """Minimal sp500_companies.csv with two tickers."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "security": ["Apple Inc.", "Microsoft Corporation"],
            "gics sector": ["Information Technology", "Information Technology"],
            "cik": [320193, 789019],
        }
    )


def _make_stocks_csv(tickers: list[str] | None = None) -> pd.DataFrame:
    """Synthetic daily OHLCV for 2 tickers × 2 years (≈ 500 rows)."""
    if tickers is None:
        tickers = ["AAPL", "MSFT"]
    dates = pd.date_range("2022-01-03", "2023-12-29", freq="B")
    rows = []
    rng = np.random.default_rng(0)
    for ticker in tickers:
        price = 150.0
        for d in dates:
            price = max(10.0, price * (1 + rng.normal(0, 0.01)))
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": round(price * 0.99, 2),
                    "high": round(price * 1.01, 2),
                    "low": round(price * 0.98, 2),
                    "close": round(price, 2),
                    "volume": int(rng.integers(10_000_000, 80_000_000)),
                    "dividends": 0.0,
                    "stock splits": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _mock_db() -> MagicMock:
    """Return a Database mock whose get_connection() accepts arbitrary SQL."""
    conn = MagicMock()
    conn.execute.return_value = conn
    db = MagicMock()
    db.get_connection.return_value = conn
    return db


# ── test_check_files_missing ───────────────────────────────────────────────────

def test_check_files_missing():
    """check_files_exist() returns False when CSV files are absent."""
    loader = KaggleDataLoader(_mock_db())
    with (
        patch("ingestion.kaggle_loader.COMPANIES_FILE", Path("/nonexistent/companies.csv")),
        patch("ingestion.kaggle_loader.STOCKS_FILE", Path("/nonexistent/stocks.csv")),
    ):
        ok, msg = loader.check_files_exist()
    assert ok is False
    assert "Missing" in msg


# ── test_load_companies_standard_cols ─────────────────────────────────────────

def test_load_companies_standard_cols(tmp_path):
    """load_companies() maps symbol/security/gics-sector to ticker/name/sector."""
    csv_path = tmp_path / "sp500_companies.csv"
    _make_companies_csv().to_csv(csv_path, index=False)

    loader = KaggleDataLoader(_mock_db())
    with patch("ingestion.kaggle_loader.COMPANIES_FILE", csv_path):
        df = loader.load_companies()

    assert "ticker" in df.columns
    assert "name" in df.columns
    assert "sector" in df.columns
    assert "cik" in df.columns
    assert "sic_code" in df.columns

    assert set(df["ticker"]) == {"AAPL", "MSFT"}
    # CIKs should be zero-padded to 10 chars
    assert all(len(c) == 10 for c in df["cik"])
    # SIC codes must be present (from SECTOR_SIC or default)
    assert df["sic_code"].notna().all()


# ── test_load_companies_no_cik_column ─────────────────────────────────────────

def test_load_companies_no_cik_column(tmp_path):
    """load_companies() generates stable surrogate CIKs when CIK column is absent."""
    companies = _make_companies_csv().drop(columns=["cik"])
    csv_path = tmp_path / "sp500_companies.csv"
    companies.to_csv(csv_path, index=False)

    loader = KaggleDataLoader(_mock_db())
    with patch("ingestion.kaggle_loader.COMPANIES_FILE", csv_path):
        df = loader.load_companies()

    assert df["cik"].notna().all()
    assert all(len(c) == 10 for c in df["cik"])
    # Deterministic: same ticker → same CIK on re-run
    with patch("ingestion.kaggle_loader.COMPANIES_FILE", csv_path):
        df2 = loader.load_companies()
    assert list(df["cik"]) == list(df2["cik"])


# ── test_build_annual_metrics_shape ───────────────────────────────────────────

def test_build_annual_metrics_shape(tmp_path):
    """build_annual_metrics() returns non-empty DataFrame with required columns."""
    companies_df = _make_companies_csv().copy()
    companies_df.columns = companies_df.columns.str.lower().str.strip()
    companies_df = companies_df.rename(columns={"symbol": "ticker", "security": "name", "gics sector": "sector"})
    companies_df["cik"] = companies_df["cik"].astype(str).str.zfill(10)
    companies_df["sic_code"] = "7372"

    stocks_df = _make_stocks_csv(["AAPL", "MSFT"])

    loader = KaggleDataLoader(_mock_db())
    result = loader.build_annual_metrics(stocks_df, companies_df)

    assert not result.empty
    required = [
        "cik", "ticker", "period_end", "form",
        "revenues", "gross_profit", "operating_income", "net_income",
        "total_assets", "current_assets", "total_liabilities",
        "current_liabilities", "stockholders_equity",
        "cash", "accounts_receivable", "inventory",
        "long_term_debt", "rd_expense", "sga_expense",
        "operating_cashflow", "capex",
    ]
    for col in required:
        assert col in result.columns, f"Missing column: {col}"

    # 2 tickers × 2 years = 4 rows
    assert len(result) == 4
    assert set(result["form"]) == {"10-K"}


# ── test_metric_flags_binary ───────────────────────────────────────────────────

def test_metric_flags_binary(tmp_path):
    """All revenue and profit values should be positive."""
    companies_df = _make_companies_csv().copy()
    companies_df.columns = companies_df.columns.str.lower().str.strip()
    companies_df = companies_df.rename(columns={"symbol": "ticker", "security": "name", "gics sector": "sector"})
    companies_df["cik"] = companies_df["cik"].astype(str).str.zfill(10)
    companies_df["sic_code"] = "7372"

    stocks_df = _make_stocks_csv(["AAPL", "MSFT"])

    loader = KaggleDataLoader(_mock_db())
    result = loader.build_annual_metrics(stocks_df, companies_df)

    # Revenue and gross_profit must be positive (prices and volume are positive)
    assert (result["revenues"] > 0).all(), "revenues must be > 0"
    assert (result["gross_profit"] > 0).all(), "gross_profit must be > 0"
    # Total assets must exceed current assets
    assert (result["total_assets"] >= result["current_assets"]).all()


# ── test_run_returns_dict_when_files_missing ───────────────────────────────────

def test_run_returns_dict_when_files_missing():
    """run() returns a dict with status='failed' when CSV files are absent."""
    loader = KaggleDataLoader(_mock_db())
    with (
        patch("ingestion.kaggle_loader.COMPANIES_FILE", Path("/no/companies.csv")),
        patch("ingestion.kaggle_loader.STOCKS_FILE", Path("/no/stocks.csv")),
    ):
        result = loader.run(max_companies=10)

    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] == "failed"
    assert "reason" in result


# ── test_metric_id_deterministic ──────────────────────────────────────────────

def test_metric_id_deterministic():
    """_metric_id() returns the same hash for the same inputs."""
    mid1 = _metric_id("0000320193", "2023-12-31", "10-K")
    mid2 = _metric_id("0000320193", "2023-12-31", "10-K")
    assert mid1 == mid2
    assert len(mid1) == 32  # MD5 hex


# ── test_accession_unique_per_year ────────────────────────────────────────────

def test_accession_unique_per_year():
    """_accession() produces different values for different years."""
    a2022 = _accession("0000320193", 2022)
    a2023 = _accession("0000320193", 2023)
    assert a2022 != a2023
    assert "2022" in a2022
    assert "2023" in a2023


# ── test_sector_sic_coverage ──────────────────────────────────────────────────

def test_sector_sic_coverage():
    """SECTOR_SIC maps all major GICS sectors to a non-empty SIC code."""
    expected_sectors = [
        "Information Technology", "Health Care", "Financials",
        "Consumer Discretionary", "Consumer Staples", "Industrials",
        "Communication Services", "Energy", "Materials", "Utilities",
    ]
    for sector in expected_sectors:
        assert sector in SECTOR_SIC, f"Missing sector: {sector}"
        assert SECTOR_SIC[sector], f"Empty SIC for: {sector}"
