"""
Seed representative AAPL financial data directly into DuckDB.
Used when the SEC EDGAR API is unavailable (e.g., in CI / sandbox environments).
"""
import hashlib
from datetime import datetime

from storage.database import Database
from ingestion.models import Company, Filing, FinancialFact, FinancialMetric


def _fact_id(cik, concept, period_end, accn):
    return hashlib.md5(f"{cik}|{concept}|{period_end}|{accn}".encode()).hexdigest()


def _metric_id(cik, period_end, form):
    return hashlib.md5(f"{cik}|{period_end}|{form}".encode()).hexdigest()


# ──────────────────────────────────────────────
# Company record
# ──────────────────────────────────────────────
AAPL_COMPANY = Company(
    cik="320193",
    name="Apple Inc.",
    ticker="AAPL",
    sic_code="3674",
    sic_description="Semiconductors and Related Devices",
    ein="94-2404110",
    state_of_incorporation="CA",
    fiscal_year_end="0930",
    exchanges="Nasdaq",
    created_at=datetime.utcnow(),
    updated_at=datetime.utcnow(),
)


# ──────────────────────────────────────────────
# Filing records  (10-K + recent 10-Qs)
# ──────────────────────────────────────────────
AAPL_FILINGS = [
    Filing("000032019323000106", "320193", "10-K", "2023-11-03", "2023-09-30", 88, "aapl-20230930.htm"),
    Filing("000032019322000108", "320193", "10-K", "2022-10-28", "2022-09-30", 85, "aapl-20220930.htm"),
    Filing("000032019321000105", "320193", "10-K", "2021-10-29", "2021-09-30", 82, "aapl-20210930.htm"),
    Filing("000032019320000096", "320193", "10-K", "2020-10-30", "2020-09-30", 80, "aapl-20200930.htm"),
    Filing("000032019319000119", "320193", "10-K", "2019-10-31", "2019-09-30", 79, "aapl-20190930.htm"),
    Filing("000032019323000077", "320193", "10-Q", "2023-08-04", "2023-07-01", 55, "aapl-20230701.htm"),
    Filing("000032019323000045", "320193", "10-Q", "2023-05-05", "2023-04-01", 52, "aapl-20230401.htm"),
    Filing("000032019323000006", "320193", "10-Q", "2023-02-03", "2022-12-31", 50, "aapl-20221231.htm"),
    Filing("000032019322000070", "320193", "10-Q", "2022-07-29", "2022-06-25", 53, "aapl-20220625.htm"),
    Filing("000032019322000041", "320193", "10-Q", "2022-04-29", "2022-03-26", 51, "aapl-20220326.htm"),
]


# ──────────────────────────────────────────────
# Annual metrics  (10-K)
# ──────────────────────────────────────────────
ANNUAL_DATA = [
    # (period_end, revenues, gross_profit, op_income, net_income, total_assets,
    #  current_assets, total_liabilities, current_liabilities, equity, cash,
    #  ar, inventory, ltd, rd, sga, opcf, capex, yoy_growth)
    ("2023-09-30", 383_285e6, 169_148e6, 114_301e6,  96_995e6, 352_583e6,
     143_566e6, 290_437e6, 145_308e6,  62_146e6, 29_965e6,
      29_508e6,   6_331e6,  95_281e6, 29_915e6, 24_932e6, 110_543e6, 10_959e6, -0.028),
    ("2022-09-30", 394_328e6, 170_782e6, 119_437e6,  99_803e6, 352_755e6,
     135_405e6, 302_083e6, 153_982e6,  50_672e6, 23_646e6,
      28_184e6,   4_946e6,  98_959e6, 26_251e6, 25_094e6, 122_151e6, 10_708e6,  0.077),
    ("2021-09-30", 365_817e6, 152_836e6, 108_949e6,  94_680e6, 351_002e6,
     134_836e6, 287_912e6, 125_481e6,  63_090e6, 34_940e6,
      26_278e6,   6_580e6, 109_106e6, 21_914e6, 21_973e6, 104_038e6, 11_085e6,  0.332),
    ("2020-09-30", 274_515e6, 104_956e6,  66_288e6,  57_411e6, 323_888e6,
     143_713e6, 258_578e6, 105_392e6,  65_339e6, 38_016e6,
      16_120e6,   4_061e6,  98_667e6, 18_752e6, 19_916e6,  80_674e6,  7_309e6,  0.055),
    ("2019-09-30", 260_174e6,  98_392e6,  63_930e6,  55_256e6, 338_516e6,
     162_819e6, 248_028e6, 105_718e6,  90_488e6, 48_844e6,
      22_926e6,   4_106e6,  91_807e6, 16_217e6, 18_245e6,  69_391e6,  6_226e6,  None),
]

QUARTERLY_DATA = [
    # (period_end, form, revenues, gross_profit, op_income, net_income, total_assets,
    #  current_assets, total_liabilities, current_liabilities, equity, cash,
    #  ar, inventory, ltd, rd, sga, opcf, capex)
    ("2023-07-01", "10-Q",  81_797e6, 36_413e6, 23_170e6, 19_881e6, 335_038e6,
      99_646e6, 275_751e6, 126_458e6,  59_287e6, 30_657e6,
      25_063e6,  7_351e6,  95_397e6,  7_442e6,  6_161e6,  26_407e6,  2_750e6),
    ("2023-04-01", "10-Q",  94_836e6, 41_976e6, 28_316e6, 24_160e6, 332_160e6,
     106_682e6, 274_763e6, 130_211e6,  57_397e6, 24_687e6,
      28_184e6,  7_312e6,  99_803e6,  7_457e6,  6_201e6,  28_408e6,  2_803e6),
    ("2022-12-31", "10-Q", 117_154e6, 52_848e6, 35_601e6, 29_998e6, 346_747e6,
     128_777e6, 287_022e6, 153_423e6,  59_725e6, 20_535e6,
      28_437e6,   4_946e6,  99_803e6,  7_709e6,  6_607e6,  34_005e6,  3_241e6),
    ("2022-06-25", "10-Q",  82_959e6, 35_371e6, 23_073e6, 19_442e6, 336_309e6,
     106_681e6, 280_991e6, 133_989e6,  55_318e6, 27_502e6,
      18_565e6,   3_903e6,  91_824e6,  6_797e6,  6_487e6,  28_977e6,  2_734e6),
    ("2022-03-26", "10-Q",  97_278e6, 42_559e6, 29_978e6, 25_010e6, 350_662e6,
     126_367e6, 287_685e6, 145_070e6,  62_977e6, 28_183e6,
      35_672e6,   5_900e6,  94_044e6,  6_387e6,  6_102e6,  28_502e6,  2_590e6),
]


def seed(db: Database) -> None:
    """Insert all AAPL seed data into the database."""
    now = datetime.utcnow()

    db.upsert_company(AAPL_COMPANY)

    for filing in AAPL_FILINGS:
        filing.ingested_at = now
        db.upsert_filing(filing)

    # Insert representative facts for each annual period
    concept_fields = [
        ("RevenueFromContractWithCustomerExcludingAssessedTax", "revenues"),
        ("GrossProfit", "gross_profit"),
        ("OperatingIncomeLoss", "op_income"),
        ("NetIncomeLoss", "net_income"),
    ]

    for row in ANNUAL_DATA:
        period_end = row[0]
        field_vals = {
            "revenues": row[1], "gross_profit": row[2],
            "op_income": row[3], "net_income": row[4],
        }
        accn = f"annual-{period_end}"
        for concept, field in concept_fields:
            val = field_vals.get(field)
            if val is None:
                continue
            fact = FinancialFact(
                id=_fact_id("320193", concept, period_end, accn),
                cik="320193",
                accession_number=accn,
                taxonomy="us-gaap",
                concept=concept,
                label=concept,
                value=val,
                unit="USD",
                period_end=period_end,
                filed=period_end,
                form="10-K",
                created_at=now,
            )
            db.upsert_financial_fact(fact)

        yoy = row[18]
        metric = FinancialMetric(
            id=_metric_id("320193", period_end, "10-K"),
            cik="320193",
            period_end=period_end,
            form="10-K",
            revenues=row[1],
            gross_profit=row[2],
            operating_income=row[3],
            net_income=row[4],
            total_assets=row[5],
            current_assets=row[6],
            total_liabilities=row[7],
            current_liabilities=row[8],
            stockholders_equity=row[9],
            cash=row[10],
            accounts_receivable=row[11],
            inventory=row[12],
            long_term_debt=row[13],
            rd_expense=row[14],
            sga_expense=row[15],
            operating_cashflow=row[16],
            capex=row[17],
            revenue_yoy_growth=yoy,
            computed_at=now,
        )
        db.upsert_financial_metric(metric)

    for row in QUARTERLY_DATA:
        period_end, form = row[0], row[1]
        accn = f"qtr-{period_end}"
        # Insert a revenue fact for each quarter
        fact = FinancialFact(
            id=_fact_id("320193", "RevenueFromContractWithCustomerExcludingAssessedTax", period_end, accn),
            cik="320193",
            accession_number=accn,
            taxonomy="us-gaap",
            concept="RevenueFromContractWithCustomerExcludingAssessedTax",
            label="Revenue",
            value=row[2],
            unit="USD",
            period_end=period_end,
            filed=period_end,
            form=form,
            created_at=now,
        )
        db.upsert_financial_fact(fact)

        metric = FinancialMetric(
            id=_metric_id("320193", period_end, form),
            cik="320193",
            period_end=period_end,
            form=form,
            revenues=row[2],
            gross_profit=row[3],
            operating_income=row[4],
            net_income=row[5],
            total_assets=row[6],
            current_assets=row[7],
            total_liabilities=row[8],
            current_liabilities=row[9],
            stockholders_equity=row[10],
            cash=row[11],
            accounts_receivable=row[12],
            inventory=row[13],
            long_term_debt=row[14],
            rd_expense=row[15],
            sga_expense=row[16],
            operating_cashflow=row[17],
            capex=row[18],
            computed_at=now,
        )
        db.upsert_financial_metric(metric)
