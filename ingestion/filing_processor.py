import uuid
from datetime import datetime
from typing import Optional

import structlog

from ingestion.edgar_client import EdgarClient
from ingestion.models import Company, Filing, IngestionRun
from ingestion.xbrl_parser import (
    compute_yoy_growth,
    extract_filings,
    parse_company_facts,
    parse_submissions,
)
from storage.database import Database

log = structlog.get_logger(__name__)


class FilingProcessor:
    def __init__(self, db: Database, client: Optional[EdgarClient] = None):
        self.db = db
        self.client = client or EdgarClient()

    def process_ticker(self, ticker: str) -> dict:
        """Resolve ticker → CIK, then ingest all filings and facts for that company."""
        log.info("resolving_ticker", ticker=ticker)
        cik = self.client.resolve_cik(ticker)
        return self.process_cik(cik)

    def process_cik(self, cik: str) -> dict:
        """Ingest all SEC filings and financial facts for a given CIK."""
        run_id = str(uuid.uuid4())
        run = IngestionRun(id=run_id, started_at=datetime.utcnow(), status="running")
        self.db.upsert_ingestion_run(run)

        try:
            log.info("ingesting_company", cik=cik)

            # 1. Fetch and store company metadata
            submissions = self.client.get_submissions(cik)
            company_data = parse_submissions(cik, submissions)
            company = Company(
                cik=cik,
                name=company_data["name"],
                ticker=company_data["ticker"],
                sic_code=str(company_data["sic_code"]) if company_data["sic_code"] else None,
                sic_description=company_data["sic_description"],
                ein=company_data["ein"],
                state_of_incorporation=company_data["state_of_incorporation"],
                fiscal_year_end=company_data["fiscal_year_end"],
                exchanges=company_data["exchanges"],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            self.db.upsert_company(company)
            run.companies_processed = 1

            # 2. Extract and store filing metadata
            filing_dicts = extract_filings(cik, submissions)
            for fd in filing_dicts:
                filing = Filing(
                    accession_number=fd["accession_number"],
                    cik=fd["cik"],
                    form_type=fd["form_type"],
                    filing_date=fd["filing_date"],
                    period_of_report=fd["period_of_report"],
                    document_count=fd["document_count"],
                    primary_document=fd["primary_document"],
                    ingested_at=datetime.utcnow(),
                )
                self.db.upsert_filing(filing)
            run.filings_processed = len(filing_dicts)
            log.info("filings_stored", cik=cik, count=len(filing_dicts))

            # 3. Fetch XBRL facts and build metrics
            facts_json = self.client.get_company_facts(cik)
            facts, metrics = parse_company_facts(cik, facts_json.get("facts", {"us-gaap": {}}))
            # compute_yoy_growth needs full facts_json structure
            facts_raw, metrics_raw = parse_company_facts(cik, facts_json)
            compute_yoy_growth(metrics_raw)

            for fact in facts_raw:
                self.db.upsert_financial_fact(fact)
            run.facts_processed = len(facts_raw)

            for metric in metrics_raw:
                self.db.upsert_financial_metric(metric)

            log.info(
                "ingestion_complete",
                cik=cik,
                facts=len(facts_raw),
                metrics=len(metrics_raw),
            )

            # 4. Mark ingestion run as completed
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            self.db.upsert_ingestion_run(run)

            # 5. Run dbt transformations
            self.db.run_dbt_transformations()

            return {
                "status": "success",
                "cik": cik,
                "company": company.name,
                "filings": run.filings_processed,
                "facts": run.facts_processed,
                "metrics": len(metrics_raw),
            }

        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)
            run.completed_at = datetime.utcnow()
            self.db.upsert_ingestion_run(run)
            log.error("ingestion_failed", cik=cik, error=str(exc))
            raise

    def run_full_ingestion(self, tickers: list[str]) -> dict:
        """Ingest multiple tickers and run dbt at the end."""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.process_ticker(ticker)
            except Exception as exc:
                results[ticker] = {"status": "error", "error": str(exc)}
        self.db.run_dbt_transformations()
        return results
