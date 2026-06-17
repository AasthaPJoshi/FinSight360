"""
Parse SEC EDGAR XBRL company facts into FinancialFact and FinancialMetric objects.
"""
import hashlib
from datetime import datetime
from typing import Optional

import structlog

from ingestion.models import FinancialFact, FinancialMetric

log = structlog.get_logger(__name__)

START_DATE = "2019-01-01"

# Maps XBRL concept names → canonical metric field names.
# Earlier entries in the list win (priority order).
CONCEPT_MAP: dict[str, str] = {
    # Revenue
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenues",
    "Revenues": "revenues",
    "SalesRevenueNet": "revenues",
    "SalesRevenueGoodsNet": "revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax": "revenues",
    # Net income
    "NetIncomeLoss": "net_income",
    "NetIncomeLossAvailableToCommonStockholdersBasic": "net_income",
    # Gross profit
    "GrossProfit": "gross_profit",
    # Operating income
    "OperatingIncomeLoss": "operating_income",
    # Assets
    "Assets": "total_assets",
    "AssetsCurrent": "current_assets",
    # Liabilities
    "Liabilities": "total_liabilities",
    "LiabilitiesCurrent": "current_liabilities",
    # Equity
    "StockholdersEquity": "stockholders_equity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "stockholders_equity",
    # Cash
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "CashCashEquivalentsAndShortTermInvestments": "cash",
    "Cash": "cash",
    # Accounts receivable
    "AccountsReceivableNetCurrent": "accounts_receivable",
    "ReceivablesNetCurrent": "accounts_receivable",
    # Inventory
    "InventoryNet": "inventory",
    # Long-term debt
    "LongTermDebt": "long_term_debt",
    "LongTermDebtNoncurrent": "long_term_debt",
    "LongTermDebtAndCapitalLeaseObligations": "long_term_debt",
    # R&D
    "ResearchAndDevelopmentExpense": "rd_expense",
    # SGA
    "SellingGeneralAndAdministrativeExpense": "sga_expense",
    # Operating cash flow
    "NetCashProvidedByUsedInOperatingActivities": "operating_cashflow",
    # Capex
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "PaymentsForCapitalImprovements": "capex",
}

ANNUAL_FORMS = {"10-K", "10-K/A"}
QUARTERLY_FORMS = {"10-Q", "10-Q/A"}
ACCEPTED_FORMS = ANNUAL_FORMS | QUARTERLY_FORMS


def _make_fact_id(cik: str, concept: str, period_end: str, accn: str) -> str:
    raw = f"{cik}|{concept}|{period_end}|{accn}"
    return hashlib.md5(raw.encode()).hexdigest()


def parse_company_facts(
    cik: str, facts_json: dict
) -> tuple[list[FinancialFact], list[FinancialMetric]]:
    """
    Parse the company facts JSON returned by EDGAR into facts and metrics.
    Returns (facts, metrics) where metrics are aggregated per (cik, period_end, form).
    """
    facts: list[FinancialFact] = []
    now = datetime.utcnow()

    for taxonomy, concepts in facts_json.get("facts", {}).items():
        for concept, concept_data in concepts.items():
            label = concept_data.get("label", concept)
            for unit, entries in concept_data.get("units", {}).items():
                if unit not in ("USD", "shares", "pure"):
                    continue
                for entry in entries:
                    period_end = entry.get("end")
                    form = entry.get("form", "")
                    if not period_end or period_end < START_DATE:
                        continue
                    if form not in ACCEPTED_FORMS:
                        continue
                    val = entry.get("val")
                    if val is None:
                        continue
                    accn = entry.get("accn", "")
                    fact = FinancialFact(
                        id=_make_fact_id(cik, concept, period_end, accn),
                        cik=cik,
                        accession_number=accn or None,
                        taxonomy=taxonomy,
                        concept=concept,
                        label=label,
                        value=float(val),
                        unit=unit,
                        period_start=entry.get("start"),
                        period_end=period_end,
                        filed=entry.get("filed"),
                        form=form,
                        created_at=now,
                    )
                    facts.append(fact)

    metrics = _build_metrics(cik, facts)
    log.info(
        "xbrl_parsed",
        cik=cik,
        facts=len(facts),
        metrics=len(metrics),
    )
    return facts, metrics


def _build_metrics(cik: str, facts: list[FinancialFact]) -> list[FinancialMetric]:
    """Aggregate facts into one FinancialMetric per (period_end, form)."""
    # Group: (period_end, form) → {field_name: value}
    # Only USD facts contribute to metrics; take the latest-filed value per concept.
    buckets: dict[tuple, dict] = {}
    filed_at: dict[tuple, dict[str, str]] = {}  # tracks filed date per field

    for fact in facts:
        if fact.unit != "USD":
            continue
        field = CONCEPT_MAP.get(fact.concept)
        if field is None:
            continue
        key = (fact.period_end, fact.form)
        if key not in buckets:
            buckets[key] = {}
            filed_at[key] = {}
        # Keep the value from the latest-filed accession for this field
        current_filed = filed_at[key].get(field, "")
        fact_filed = fact.filed or ""
        if fact_filed >= current_filed:
            buckets[key][field] = fact.value
            filed_at[key][field] = fact_filed

    now = datetime.utcnow()
    metrics: list[FinancialMetric] = []
    for (period_end, form), fields in buckets.items():
        metric_id = hashlib.md5(f"{cik}|{period_end}|{form}".encode()).hexdigest()
        m = FinancialMetric(
            id=metric_id,
            cik=cik,
            period_end=period_end,
            form=form,
            revenues=fields.get("revenues"),
            gross_profit=fields.get("gross_profit"),
            operating_income=fields.get("operating_income"),
            net_income=fields.get("net_income"),
            total_assets=fields.get("total_assets"),
            current_assets=fields.get("current_assets"),
            total_liabilities=fields.get("total_liabilities"),
            current_liabilities=fields.get("current_liabilities"),
            stockholders_equity=fields.get("stockholders_equity"),
            cash=fields.get("cash"),
            accounts_receivable=fields.get("accounts_receivable"),
            inventory=fields.get("inventory"),
            long_term_debt=fields.get("long_term_debt"),
            rd_expense=fields.get("rd_expense"),
            sga_expense=fields.get("sga_expense"),
            operating_cashflow=fields.get("operating_cashflow"),
            capex=fields.get("capex"),
            computed_at=now,
        )
        metrics.append(m)

    return metrics


def compute_yoy_growth(metrics: list[FinancialMetric]) -> list[FinancialMetric]:
    """
    Compute revenue YoY growth for annual (10-K) metrics.
    Mutates metric objects in-place and returns the list.
    """
    annual = {
        m.period_end: m
        for m in metrics
        if m.form in ANNUAL_FORMS and m.revenues is not None
    }
    for m in metrics:
        if m.form not in ANNUAL_FORMS or m.revenues is None:
            continue
        # Same month/day but one year earlier
        try:
            year = int(m.period_end[:4]) - 1
            prior_end = f"{year}{m.period_end[4:]}"
        except (ValueError, IndexError):
            continue
        prior = annual.get(prior_end)
        if prior and prior.revenues and prior.revenues != 0:
            m.revenue_yoy_growth = (m.revenues - prior.revenues) / abs(prior.revenues)
    return metrics


def parse_submissions(cik: str, submissions_json: dict) -> dict:
    """Extract company metadata from submissions JSON."""
    return {
        "cik": cik,
        "name": submissions_json.get("name", ""),
        "ticker": (submissions_json.get("tickers") or [""])[0] or None,
        "sic_code": submissions_json.get("sic"),
        "sic_description": submissions_json.get("sicDescription"),
        "ein": submissions_json.get("ein"),
        "state_of_incorporation": submissions_json.get("stateOfIncorporation"),
        "fiscal_year_end": submissions_json.get("fiscalYearEnd"),
        "exchanges": ",".join(submissions_json.get("exchanges") or []) or None,
    }


def extract_filings(cik: str, submissions_json: dict) -> list[dict]:
    """Extract 10-K and 10-Q filing records from the submissions JSON."""
    recent = submissions_json.get("filings", {}).get("recent", {})
    if not recent:
        return []

    accns = recent.get("accessionNumber", [])
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    periods = recent.get("reportDate", [])
    docs = recent.get("primaryDocument", [])
    doc_counts = recent.get("documentCount", [])

    results = []
    for i, form in enumerate(forms):
        if form not in ACCEPTED_FORMS:
            continue
        period = periods[i] if i < len(periods) else None
        if not period or period < START_DATE:
            continue
        results.append(
            {
                "accession_number": accns[i].replace("-", "") if i < len(accns) else "",
                "cik": cik,
                "form_type": form,
                "filing_date": dates[i] if i < len(dates) else None,
                "period_of_report": period,
                "document_count": doc_counts[i] if i < len(doc_counts) else None,
                "primary_document": docs[i] if i < len(docs) else None,
            }
        )
    return results
