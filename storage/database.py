import os
import subprocess
from datetime import datetime
from typing import Optional

import duckdb
import structlog

from ingestion.models import Company, Filing, FinancialFact, FinancialMetric, IngestionRun

log = structlog.get_logger(__name__)

DEFAULT_DB_PATH = os.environ.get("DUCKDB_PATH", "data/finsight360.duckdb")


class Database:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self._create_schema()

    def _create_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                cik VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                ticker VARCHAR,
                sic_code VARCHAR,
                sic_description VARCHAR,
                ein VARCHAR,
                state_of_incorporation VARCHAR,
                fiscal_year_end VARCHAR,
                exchanges VARCHAR,
                created_at TIMESTAMP DEFAULT current_timestamp,
                updated_at TIMESTAMP DEFAULT current_timestamp
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS filings (
                accession_number VARCHAR PRIMARY KEY,
                cik VARCHAR NOT NULL,
                form_type VARCHAR NOT NULL,
                filing_date DATE,
                period_of_report DATE,
                document_count INTEGER,
                primary_document VARCHAR,
                ingested_at TIMESTAMP DEFAULT current_timestamp
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_facts (
                id VARCHAR PRIMARY KEY,
                cik VARCHAR NOT NULL,
                accession_number VARCHAR,
                taxonomy VARCHAR,
                concept VARCHAR NOT NULL,
                label VARCHAR,
                value DOUBLE,
                unit VARCHAR,
                period_start DATE,
                period_end DATE NOT NULL,
                filed DATE,
                form VARCHAR,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_metrics (
                id VARCHAR PRIMARY KEY,
                cik VARCHAR NOT NULL,
                period_end DATE NOT NULL,
                form VARCHAR,
                revenues DOUBLE,
                gross_profit DOUBLE,
                operating_income DOUBLE,
                net_income DOUBLE,
                total_assets DOUBLE,
                current_assets DOUBLE,
                total_liabilities DOUBLE,
                current_liabilities DOUBLE,
                stockholders_equity DOUBLE,
                cash DOUBLE,
                accounts_receivable DOUBLE,
                inventory DOUBLE,
                long_term_debt DOUBLE,
                rd_expense DOUBLE,
                sga_expense DOUBLE,
                operating_cashflow DOUBLE,
                capex DOUBLE,
                revenue_yoy_growth DOUBLE,
                computed_at TIMESTAMP DEFAULT current_timestamp
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_runs (
                id VARCHAR PRIMARY KEY,
                started_at TIMESTAMP DEFAULT current_timestamp,
                completed_at TIMESTAMP,
                status VARCHAR DEFAULT 'running',
                companies_processed INTEGER DEFAULT 0,
                filings_processed INTEGER DEFAULT 0,
                facts_processed INTEGER DEFAULT 0,
                error_message VARCHAR
            )
        """)

        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_filings_cik ON filings(cik)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_cik ON financial_facts(cik)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_period ON financial_facts(period_end)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_cik ON financial_metrics(cik)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_period ON financial_metrics(period_end)")

        # ── ML anomaly scores ─────────────────────────────────────────────
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_anomaly_scores (
                cik VARCHAR PRIMARY KEY,
                ticker VARCHAR,
                company_name VARCHAR,
                industry_group VARCHAR,
                isolation_forest_score DOUBLE,
                anomaly_percentile DOUBLE,
                ml_risk_score DOUBLE,
                is_anomaly BOOLEAN,
                pre_ml_risk_score DOUBLE,
                risk_tier VARCHAR,
                computed_at TIMESTAMP DEFAULT current_timestamp
            )
        """)

        # ── Benford analysis results ───────────────────────────────────────
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS benford_results (
                cik VARCHAR PRIMARY KEY,
                company_name VARCHAR,
                ticker VARCHAR,
                n_observations BIGINT,
                mad_score DOUBLE,
                chi_square_statistic DOUBLE,
                chi_square_p_value DOUBLE,
                conformity_rating VARCHAR,
                benford_risk_score DOUBLE,
                suspicious_digits VARCHAR,
                interpretation VARCHAR,
                computed_at TIMESTAMP DEFAULT current_timestamp
            )
        """)

        # ── GenAI risk briefs ──────────────────────────────────────────────
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_briefs (
                cik VARCHAR PRIMARY KEY,
                company_name VARCHAR,
                ticker VARCHAR,
                final_risk_score DOUBLE,
                brief_text TEXT,
                generated_at TIMESTAMP DEFAULT current_timestamp
            )
        """)

        # ── SHAP explainability values ─────────────────────────────────────
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS shap_values (
                id VARCHAR PRIMARY KEY,
                cik VARCHAR NOT NULL,
                ticker VARCHAR,
                company_name VARCHAR,
                feature_name VARCHAR NOT NULL,
                friendly_name VARCHAR,
                shap_value DOUBLE,
                abs_shap_value DOUBLE,
                feature_value DOUBLE,
                direction VARCHAR,
                rank_in_company INTEGER,
                computed_at TIMESTAMP DEFAULT current_timestamp
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_shap_cik ON shap_values(cik)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_briefs_score ON risk_briefs(final_risk_score)"
        )

    def upsert_company(self, company: Company) -> None:
        self.conn.execute(
            """
            INSERT INTO companies
                (cik, name, ticker, sic_code, sic_description, ein,
                 state_of_incorporation, fiscal_year_end, exchanges,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (cik) DO UPDATE SET
                name = excluded.name,
                ticker = excluded.ticker,
                sic_code = excluded.sic_code,
                sic_description = excluded.sic_description,
                ein = excluded.ein,
                state_of_incorporation = excluded.state_of_incorporation,
                fiscal_year_end = excluded.fiscal_year_end,
                exchanges = excluded.exchanges,
                updated_at = excluded.updated_at
            """,
            [
                company.cik,
                company.name,
                company.ticker,
                company.sic_code,
                company.sic_description,
                company.ein,
                company.state_of_incorporation,
                company.fiscal_year_end,
                company.exchanges,
                company.created_at or datetime.utcnow(),
                company.updated_at or datetime.utcnow(),
            ],
        )

    def upsert_filing(self, filing: Filing) -> None:
        self.conn.execute(
            """
            INSERT INTO filings
                (accession_number, cik, form_type, filing_date,
                 period_of_report, document_count, primary_document, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (accession_number) DO UPDATE SET
                cik = excluded.cik,
                form_type = excluded.form_type,
                filing_date = excluded.filing_date,
                period_of_report = excluded.period_of_report,
                document_count = excluded.document_count,
                primary_document = excluded.primary_document
            """,
            [
                filing.accession_number,
                filing.cik,
                filing.form_type,
                filing.filing_date,
                filing.period_of_report,
                filing.document_count,
                filing.primary_document,
                filing.ingested_at or datetime.utcnow(),
            ],
        )

    def upsert_financial_fact(self, fact: FinancialFact) -> None:
        self.conn.execute(
            """
            INSERT INTO financial_facts
                (id, cik, accession_number, taxonomy, concept, label,
                 value, unit, period_start, period_end, filed, form, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                value = excluded.value,
                filed = excluded.filed
            """,
            [
                fact.id,
                fact.cik,
                fact.accession_number,
                fact.taxonomy,
                fact.concept,
                fact.label,
                fact.value,
                fact.unit,
                fact.period_start,
                fact.period_end,
                fact.filed,
                fact.form,
                fact.created_at or datetime.utcnow(),
            ],
        )

    def upsert_financial_metric(self, metric: FinancialMetric) -> None:
        self.conn.execute(
            """
            INSERT INTO financial_metrics
                (id, cik, period_end, form, revenues, gross_profit,
                 operating_income, net_income, total_assets, current_assets,
                 total_liabilities, current_liabilities, stockholders_equity,
                 cash, accounts_receivable, inventory, long_term_debt,
                 rd_expense, sga_expense, operating_cashflow, capex,
                 revenue_yoy_growth, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                revenues = excluded.revenues,
                gross_profit = excluded.gross_profit,
                operating_income = excluded.operating_income,
                net_income = excluded.net_income,
                total_assets = excluded.total_assets,
                current_assets = excluded.current_assets,
                total_liabilities = excluded.total_liabilities,
                current_liabilities = excluded.current_liabilities,
                stockholders_equity = excluded.stockholders_equity,
                cash = excluded.cash,
                accounts_receivable = excluded.accounts_receivable,
                inventory = excluded.inventory,
                long_term_debt = excluded.long_term_debt,
                rd_expense = excluded.rd_expense,
                sga_expense = excluded.sga_expense,
                operating_cashflow = excluded.operating_cashflow,
                capex = excluded.capex,
                revenue_yoy_growth = excluded.revenue_yoy_growth,
                computed_at = excluded.computed_at
            """,
            [
                metric.id,
                metric.cik,
                metric.period_end,
                metric.form,
                metric.revenues,
                metric.gross_profit,
                metric.operating_income,
                metric.net_income,
                metric.total_assets,
                metric.current_assets,
                metric.total_liabilities,
                metric.current_liabilities,
                metric.stockholders_equity,
                metric.cash,
                metric.accounts_receivable,
                metric.inventory,
                metric.long_term_debt,
                metric.rd_expense,
                metric.sga_expense,
                metric.operating_cashflow,
                metric.capex,
                metric.revenue_yoy_growth,
                metric.computed_at or datetime.utcnow(),
            ],
        )

    def upsert_ingestion_run(self, run: IngestionRun) -> None:
        self.conn.execute(
            """
            INSERT INTO ingestion_runs
                (id, started_at, completed_at, status, companies_processed,
                 filings_processed, facts_processed, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                completed_at = excluded.completed_at,
                status = excluded.status,
                companies_processed = excluded.companies_processed,
                filings_processed = excluded.filings_processed,
                facts_processed = excluded.facts_processed,
                error_message = excluded.error_message
            """,
            [
                run.id,
                run.started_at or datetime.utcnow(),
                run.completed_at,
                run.status,
                run.companies_processed,
                run.filings_processed,
                run.facts_processed,
                run.error_message,
            ],
        )

    def get_company(self, cik: str) -> Optional[dict]:
        result = self.conn.execute(
            "SELECT * FROM companies WHERE cik = ?", [cik]
        ).fetchone()
        if result is None:
            return None
        cols = [d[0] for d in self.conn.description]
        return dict(zip(cols, result))

    def get_metrics_for_company(self, cik: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM financial_metrics WHERE cik = ? ORDER BY period_end",
            [cik],
        ).fetchall()
        if not rows:
            return []
        cols = [d[0] for d in self.conn.description]
        return [dict(zip(cols, r)) for r in rows]

    def table_exists(self, name: str) -> bool:
        result = self.conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [name],
        ).fetchone()
        return result[0] > 0

    def run_dbt_transformations(self) -> dict:
        """Run dbt models after ingestion completes."""
        env = os.environ.copy()
        env["DUCKDB_PATH"] = os.path.abspath(self.db_path)
        result = subprocess.run(
            ["dbt", "run", "--profiles-dir", "dbt", "--project-dir", "dbt"],
            capture_output=True,
            text=True,
            cwd=".",
            env=env,
        )
        if result.returncode == 0:
            log.info("dbt_run_success", stdout=result.stdout[-500:])
        else:
            log.error("dbt_run_failed", stderr=result.stderr[-500:])
        return {"returncode": result.returncode, "stdout": result.stdout}

    def get_connection(self):
        """Return the raw DuckDB connection (used by ML layer)."""
        return self.conn

    def close(self):
        self.conn.close()
