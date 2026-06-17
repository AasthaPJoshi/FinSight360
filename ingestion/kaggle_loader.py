"""
Kaggle CSV data loader for FinSight360.

Loads S&P 500 data from the Kaggle dataset 'andrewmvd/sp-500-stocks'
into the same DuckDB tables used by the SEC EDGAR ingestion pipeline.
The downstream dbt models, ML pipeline, and dashboard are unchanged.

Required files (data/kaggle/):
  sp500_companies.csv  — symbol, security, GICS sector, CIK, ...
  sp500_stocks.csv     — date, symbol, open, high, low, close, volume, ...

DuckDB tables populated:
  companies          ← sp500_companies.csv
  filings            ← synthetic annual filing records
  financial_metrics  ← annual aggregates computed from daily price/volume
  ingestion_runs     ← pipeline run metadata

Design note: Kaggle stock data contains daily OHLCV, not income statements.
Annual financial_metrics rows are constructed from yearly aggregations of
price and volume data (revenue proxy = avg_close × annual_volume / 1e9),
with sector-calibrated synthetic ratios for the non-revenue columns. This
gives the dbt models enough signal to compute risk flags and enables
downstream ML scoring without requiring SEC EDGAR access.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

KAGGLE_DIR = Path("data/kaggle")
COMPANIES_FILE = KAGGLE_DIR / "sp500_companies.csv"
STOCKS_FILE = KAGGLE_DIR / "sp500_stocks.csv"

# Map GICS sector → SEC SIC code (representative)
SECTOR_SIC = {
    "Information Technology": "7372",
    "Technology": "7372",
    "Health Care": "8099",
    "Healthcare": "8099",
    "Financials": "6020",
    "Consumer Discretionary": "5900",
    "Consumer Staples": "5400",
    "Industrials": "3559",
    "Communication Services": "4813",
    "Energy": "1311",
    "Materials": "2819",
    "Real Estate": "6552",
    "Utilities": "4911",
}

# Sector-level financial ratio medians (from S&P 500 historical averages)
# Used to generate realistic synthetic income-statement proxies.
SECTOR_PROFILES: dict[str, dict[str, float]] = {
    "Information Technology": dict(gm=0.55, om=0.25, nm=0.20, capex_r=0.04, rnd_r=0.12),
    "Technology":             dict(gm=0.55, om=0.25, nm=0.20, capex_r=0.04, rnd_r=0.12),
    "Health Care":            dict(gm=0.62, om=0.18, nm=0.14, capex_r=0.05, rnd_r=0.15),
    "Healthcare":             dict(gm=0.62, om=0.18, nm=0.14, capex_r=0.05, rnd_r=0.15),
    "Financials":             dict(gm=0.58, om=0.30, nm=0.22, capex_r=0.02, rnd_r=0.00),
    "Consumer Discretionary": dict(gm=0.35, om=0.10, nm=0.07, capex_r=0.05, rnd_r=0.01),
    "Consumer Staples":       dict(gm=0.38, om=0.12, nm=0.08, capex_r=0.04, rnd_r=0.01),
    "Industrials":            dict(gm=0.33, om=0.12, nm=0.08, capex_r=0.05, rnd_r=0.02),
    "Communication Services": dict(gm=0.52, om=0.20, nm=0.14, capex_r=0.10, rnd_r=0.03),
    "Energy":                 dict(gm=0.28, om=0.12, nm=0.08, capex_r=0.15, rnd_r=0.01),
    "Materials":              dict(gm=0.28, om=0.10, nm=0.07, capex_r=0.08, rnd_r=0.01),
    "Real Estate":            dict(gm=0.58, om=0.32, nm=0.20, capex_r=0.20, rnd_r=0.00),
    "Utilities":              dict(gm=0.32, om=0.18, nm=0.11, capex_r=0.20, rnd_r=0.00),
}
_DEFAULT_PROFILE = dict(gm=0.40, om=0.15, nm=0.10, capex_r=0.06, rnd_r=0.03)


def _metric_id(cik: str, period_end: str, form: str) -> str:
    return hashlib.md5(f"{cik}|{period_end}|{form}".encode()).hexdigest()


def _accession(cik: str, year: int) -> str:
    digest = abs(hash(f"{cik}{year}")) % 100_000
    return f"{cik[:10].zfill(10)}-{year}-10K{digest:05d}"


def _jitter(base: float, spread: float = 0.08, rng: np.random.Generator | None = None) -> float:
    """Return base ± spread, always positive."""
    r = rng if rng is not None else np.random.default_rng()
    return float(base * max(0.01, 1.0 + r.uniform(-spread, spread)))


class KaggleDataLoader:
    """
    Loads Kaggle S&P 500 CSV data into FinSight360 DuckDB tables.

    Usage::

        from storage.database import Database
        from ingestion.kaggle_loader import KaggleDataLoader

        db = Database()
        loader = KaggleDataLoader(db)
        summary = loader.run(max_companies=100)
        db.close()
    """

    def __init__(self, db) -> None:
        self.db = db

    # ── public ────────────────────────────────────────────────────────────

    def check_files_exist(self) -> tuple[bool, str]:
        """Return (True, 'OK') if both CSVs are present, else (False, reason)."""
        if not COMPANIES_FILE.exists():
            return False, (
                f"Missing: {COMPANIES_FILE}\n"
                "Run: python scripts/download_kaggle_data.py\n"
                "Or download manually from: "
                "kaggle.com/datasets/andrewmvd/sp-500-stocks"
            )
        if not STOCKS_FILE.exists():
            return False, f"Missing: {STOCKS_FILE}"
        return True, "OK"

    def load_companies(self) -> pd.DataFrame:
        """Read sp500_companies.csv and normalise to internal schema."""
        df = pd.read_csv(COMPANIES_FILE)
        df.columns = df.columns.str.lower().str.strip()

        # Column aliases across dataset versions
        aliases = {
            "symbol": "ticker",
            "security": "name",
            "gics sector": "sector",
            "gics sub-industry": "sub_industry",
            "headquarters location": "hq",
        }
        df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

        for col in ("ticker", "name", "sector"):
            if col not in df.columns:
                df[col] = "Unknown"

        # Use CIK from CSV when present; otherwise derive a stable surrogate
        if "cik" in df.columns and df["cik"].notna().any():
            df["cik"] = (
                df["cik"]
                .fillna(0)
                .astype(float)
                .astype(int)
                .astype(str)
                .str.zfill(10)
            )
        else:
            df["cik"] = df["ticker"].apply(
                lambda t: str(
                    int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % 9_000_000 + 1_000_000
                ).zfill(10)
            )

        df["sic_code"] = df["sector"].map(SECTOR_SIC).fillna("9999")

        log.info("companies_loaded", count=len(df))
        return df

    def load_stocks(self) -> pd.DataFrame:
        """Read sp500_stocks.csv, parse dates, return sorted DataFrame."""
        df = pd.read_csv(STOCKS_FILE)
        df.columns = df.columns.str.lower().str.strip()

        if "symbol" in df.columns:
            df = df.rename(columns={"symbol": "ticker"})

        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        log.info(
            "stocks_loaded",
            rows=len(df),
            tickers=df["ticker"].nunique(),
            from_=str(df["date"].min().date()),
            to=str(df["date"].max().date()),
        )
        return df

    def build_annual_metrics(
        self,
        stocks_df: pd.DataFrame,
        companies_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate daily OHLCV into annual financial_metrics rows.

        Revenue proxy = (avg_close × annual_volume) / 1e9
        Balance-sheet and income-statement items are derived from
        sector-calibrated ratios applied to the revenue proxy.
        """
        stocks_df = stocks_df.copy()
        stocks_df["year"] = stocks_df["date"].dt.year

        annual = (
            stocks_df.groupby(["ticker", "year"])
            .agg(
                period_end_raw=("date", "max"),
                avg_close=("close", "mean"),
                total_volume=("volume", "sum"),
                open_price=("open", "first"),
                close_price=("close", "last"),
                trading_days=("close", "count"),
            )
            .reset_index()
        )

        # Revenue proxy (billions → raw dollars for consistency with EDGAR scale)
        annual["revenues"] = annual["avg_close"] * annual["total_volume"] / 1e3

        # YoY growth
        annual = annual.sort_values(["ticker", "year"])
        annual["revenue_yoy_growth"] = annual.groupby("ticker")["revenues"].pct_change()

        # Build CIK + sector maps
        cik_map = dict(zip(companies_df["ticker"], companies_df["cik"].astype(str)))
        sector_map = dict(zip(companies_df["ticker"], companies_df["sector"].fillna("Unknown")))

        annual["cik"] = annual["ticker"].map(cik_map)
        annual["sector"] = annual["ticker"].map(sector_map).fillna("Unknown")
        annual = annual.dropna(subset=["cik"])

        # Synthetic income / balance-sheet columns using sector medians
        rng = np.random.default_rng(42)

        def _col(base_col: str, ratio_key: str, spread: float = 0.08) -> pd.Series:
            profiles = annual["sector"].map(SECTOR_PROFILES).apply(
                lambda p: (p or _DEFAULT_PROFILE)[ratio_key]
            )
            noise = pd.Series(
                rng.uniform(1 - spread, 1 + spread, len(annual)),
                index=annual.index,
            )
            return annual[base_col] * profiles * noise

        rev = annual["revenues"]
        annual["gross_profit"] = _col("revenues", "gm")
        annual["operating_income"] = _col("revenues", "om")
        annual["net_income"] = _col("revenues", "nm")

        # Balance sheet (multiples of revenue)
        annual["total_assets"] = rev * pd.Series(rng.uniform(1.8, 3.2, len(annual)), index=annual.index)
        annual["current_assets"] = rev * pd.Series(rng.uniform(0.5, 1.2, len(annual)), index=annual.index)
        annual["total_liabilities"] = rev * pd.Series(rng.uniform(1.0, 2.2, len(annual)), index=annual.index)
        annual["current_liabilities"] = rev * pd.Series(rng.uniform(0.3, 0.9, len(annual)), index=annual.index)
        annual["stockholders_equity"] = annual["total_assets"] - annual["total_liabilities"]
        annual["cash"] = rev * pd.Series(rng.uniform(0.08, 0.30, len(annual)), index=annual.index)
        annual["accounts_receivable"] = rev * pd.Series(rng.uniform(0.08, 0.25, len(annual)), index=annual.index)
        annual["inventory"] = rev * pd.Series(rng.uniform(0.02, 0.12, len(annual)), index=annual.index)
        annual["long_term_debt"] = rev * pd.Series(rng.uniform(0.20, 0.80, len(annual)), index=annual.index)

        annual["rd_expense"] = _col("revenues", "rnd_r", spread=0.15)
        annual["sga_expense"] = rev * pd.Series(rng.uniform(0.05, 0.18, len(annual)), index=annual.index)
        annual["operating_cashflow"] = annual["net_income"] * pd.Series(
            rng.uniform(1.0, 1.4, len(annual)), index=annual.index
        )
        annual["capex"] = _col("revenues", "capex_r", spread=0.20)

        # period_end → last day of the fiscal year (Dec 31)
        annual["period_end"] = annual["year"].apply(lambda y: date(y, 12, 31))
        annual["form"] = "10-K"

        log.info(
            "annual_metrics_built",
            rows=len(annual),
            tickers=annual["ticker"].nunique(),
            years=sorted(annual["year"].unique().tolist()),
        )
        return annual

    def run(self, max_companies: int = 100) -> dict:
        """
        Main entry point: CSV files → DuckDB tables.

        Args:
            max_companies: cap on number of companies to load (0 = no cap).

        Returns:
            dict with keys: status, companies, metrics, filings.
        """
        ok, msg = self.check_files_exist()
        if not ok:
            log.error("kaggle_files_missing", reason=msg)
            return {"status": "failed", "reason": msg}

        conn = self.db.get_connection()
        run_id = str(uuid.uuid4())
        started = datetime.utcnow()

        conn.execute(
            """
            INSERT INTO ingestion_runs
                (id, started_at, status, companies_processed,
                 filings_processed, facts_processed)
            VALUES (?, ?, 'running', 0, 0, 0)
            """,
            [run_id, started],
        )

        try:
            # ── 1. Companies ──────────────────────────────────────────────
            log.info("step", n=1, name="load_companies")
            companies_df = self.load_companies()
            if max_companies and max_companies > 0:
                companies_df = companies_df.head(max_companies)

            now = datetime.utcnow()
            for _, row in companies_df.iterrows():
                conn.execute(
                    """
                    INSERT INTO companies
                        (cik, name, ticker, sic_code, sic_description,
                         state_of_incorporation, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'US', ?, ?)
                    ON CONFLICT (cik) DO UPDATE SET
                        name      = excluded.name,
                        ticker    = excluded.ticker,
                        sic_code  = excluded.sic_code,
                        updated_at = excluded.updated_at
                    """,
                    [
                        str(row["cik"]),
                        str(row.get("name", row.get("ticker", ""))),
                        str(row.get("ticker", "")),
                        str(row.get("sic_code", "9999")),
                        str(row.get("sector", "Unknown")),
                        now,
                        now,
                    ],
                )
            log.info("companies_inserted", count=len(companies_df))

            # ── 2. Stock data ─────────────────────────────────────────────
            log.info("step", n=2, name="load_stocks")
            stocks_df = self.load_stocks()
            tickers = set(companies_df["ticker"].tolist())
            stocks_df = stocks_df[stocks_df["ticker"].isin(tickers)]

            # ── 3. Annual metrics ─────────────────────────────────────────
            log.info("step", n=3, name="build_annual_metrics")
            annual_df = self.build_annual_metrics(stocks_df, companies_df)

            # ── 4. Insert into DuckDB ─────────────────────────────────────
            log.info("step", n=4, name="insert_duckdb")
            metrics_count = 0
            filing_count = 0

            for _, row in annual_df.iterrows():
                cik = str(row["cik"])
                period = str(row["period_end"])
                year = int(row["year"])

                # Synthetic filing record
                acc = _accession(cik, year)
                conn.execute(
                    """
                    INSERT INTO filings
                        (accession_number, cik, form_type,
                         filing_date, period_of_report,
                         document_count, ingested_at)
                    VALUES (?, ?, '10-K', ?, ?, 1, current_timestamp)
                    ON CONFLICT (accession_number) DO NOTHING
                    """,
                    [acc, cik, period, period],
                )
                filing_count += 1

                # Financial metrics row
                mid = _metric_id(cik, period, "10-K")

                def _safe(col: str) -> float | None:
                    val = row.get(col)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return None
                    return float(val)

                conn.execute(
                    """
                    INSERT INTO financial_metrics (
                        id, cik, period_end, form,
                        revenues, gross_profit, operating_income, net_income,
                        total_assets, current_assets, total_liabilities,
                        current_liabilities, stockholders_equity,
                        cash, accounts_receivable, inventory,
                        long_term_debt, rd_expense, sga_expense,
                        operating_cashflow, capex,
                        revenue_yoy_growth, computed_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,current_timestamp)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    [
                        mid, cik, period, "10-K",
                        _safe("revenues"),
                        _safe("gross_profit"),
                        _safe("operating_income"),
                        _safe("net_income"),
                        _safe("total_assets"),
                        _safe("current_assets"),
                        _safe("total_liabilities"),
                        _safe("current_liabilities"),
                        _safe("stockholders_equity"),
                        _safe("cash"),
                        _safe("accounts_receivable"),
                        _safe("inventory"),
                        _safe("long_term_debt"),
                        _safe("rd_expense"),
                        _safe("sga_expense"),
                        _safe("operating_cashflow"),
                        _safe("capex"),
                        _safe("revenue_yoy_growth"),
                    ],
                )
                metrics_count += 1

            # ── 5. Update run record ───────────────────────────────────────
            conn.execute(
                """
                UPDATE ingestion_runs SET
                    status             = 'completed',
                    completed_at       = current_timestamp,
                    companies_processed = ?,
                    filings_processed  = ?,
                    facts_processed    = ?
                WHERE id = ?
                """,
                [len(companies_df), filing_count, metrics_count, run_id],
            )

            summary = {
                "status": "completed",
                "companies": len(companies_df),
                "metrics": metrics_count,
                "filings": filing_count,
            }
            log.info("kaggle_load_complete", **summary)
            return summary

        except Exception as exc:
            conn.execute(
                "UPDATE ingestion_runs SET status = 'failed' WHERE id = ?",
                [run_id],
            )
            log.error("kaggle_load_failed", error=str(exc))
            raise
