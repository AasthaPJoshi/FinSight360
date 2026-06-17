from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


@dataclass
class Company:
    cik: str
    name: str
    ticker: Optional[str] = None
    sic_code: Optional[str] = None
    sic_description: Optional[str] = None
    ein: Optional[str] = None
    state_of_incorporation: Optional[str] = None
    fiscal_year_end: Optional[str] = None
    exchanges: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Filing:
    accession_number: str
    cik: str
    form_type: str
    filing_date: Optional[date] = None
    period_of_report: Optional[date] = None
    document_count: Optional[int] = None
    primary_document: Optional[str] = None
    ingested_at: Optional[datetime] = None


@dataclass
class FinancialFact:
    id: str
    cik: str
    concept: str
    period_end: str
    value: Optional[float] = None
    accession_number: Optional[str] = None
    taxonomy: Optional[str] = None
    label: Optional[str] = None
    unit: Optional[str] = None
    period_start: Optional[str] = None
    filed: Optional[str] = None
    form: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class FinancialMetric:
    id: str
    cik: str
    period_end: str
    form: Optional[str] = None
    revenues: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    current_liabilities: Optional[float] = None
    stockholders_equity: Optional[float] = None
    cash: Optional[float] = None
    accounts_receivable: Optional[float] = None
    inventory: Optional[float] = None
    long_term_debt: Optional[float] = None
    rd_expense: Optional[float] = None
    sga_expense: Optional[float] = None
    operating_cashflow: Optional[float] = None
    capex: Optional[float] = None
    revenue_yoy_growth: Optional[float] = None
    computed_at: Optional[datetime] = None


@dataclass
class IngestionRun:
    id: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "running"
    companies_processed: int = 0
    filings_processed: int = 0
    facts_processed: int = 0
    error_message: Optional[str] = None
