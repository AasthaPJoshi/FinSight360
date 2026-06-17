"""Tests for XBRL fact parsing and metric aggregation."""
import pytest
from ingestion.xbrl_parser import (
    parse_company_facts,
    compute_yoy_growth,
    extract_filings,
    parse_submissions,
)


SAMPLE_FACTS = {
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
                        },
                        {
                            "accn": "0000320193-22-000108",
                            "start": "2021-10-01",
                            "end": "2022-09-30",
                            "val": 394328000000,
                            "form": "10-K",
                            "filed": "2022-11-04",
                        },
                        {
                            "accn": "0000320193-23-000077",
                            "start": "2023-04-02",
                            "end": "2023-07-01",
                            "val": 81797000000,
                            "form": "10-Q",
                            "filed": "2023-08-04",
                        },
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
                        },
                    ]
                },
            },
        }
    }
}


def test_parse_facts_count():
    """All USD facts within date range should be parsed."""
    facts, metrics = parse_company_facts("320193", SAMPLE_FACTS)
    assert len(facts) > 0
    assert len(metrics) > 0


def test_extract_revenues():
    """Revenue concept should map to 'revenues' field in metrics."""
    facts, metrics = parse_company_facts("320193", SAMPLE_FACTS)
    annual = [m for m in metrics if m.form == "10-K" and m.period_end == "2023-09-30"]
    assert len(annual) == 1
    assert annual[0].revenues == pytest.approx(383285000000.0)
    assert annual[0].net_income == pytest.approx(96995000000.0)


def test_handle_missing_concepts():
    """When facts don't include a concept, the metric field should be None."""
    sparse_facts = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "label": "Net Income",
                    "units": {
                        "USD": [
                            {
                                "accn": "test-001",
                                "end": "2023-09-30",
                                "val": 50000000000,
                                "form": "10-K",
                                "filed": "2023-11-01",
                            }
                        ]
                    },
                }
            }
        }
    }
    facts, metrics = parse_company_facts("999999", sparse_facts)
    assert len(metrics) == 1
    assert metrics[0].revenues is None
    assert metrics[0].net_income == pytest.approx(50000000000.0)


def test_deduplicate_facts():
    """When the same concept is reported in multiple filings, keep latest filed value."""
    duplicate_facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "label": "Revenue",
                    "units": {
                        "USD": [
                            {
                                "accn": "old-accn",
                                "end": "2023-09-30",
                                "val": 100000000000,
                                "form": "10-K",
                                "filed": "2023-10-01",
                            },
                            {
                                "accn": "new-accn",
                                "end": "2023-09-30",
                                "val": 383285000000,
                                "form": "10-K",
                                "filed": "2023-11-03",
                            },
                        ]
                    },
                }
            }
        }
    }
    facts, metrics = parse_company_facts("320193", duplicate_facts)
    annual = [m for m in metrics if m.form == "10-K" and m.period_end == "2023-09-30"]
    assert len(annual) == 1
    assert annual[0].revenues == pytest.approx(383285000000.0)


def test_compute_yoy_growth():
    """YoY growth should be computed correctly between consecutive 10-K periods."""
    facts, metrics = parse_company_facts("320193", SAMPLE_FACTS)
    metrics = compute_yoy_growth(metrics)
    # FY2023 revenue: 383285B, FY2022: 394328B → growth ≈ -2.8%
    fy23 = next(
        (m for m in metrics if m.form == "10-K" and m.period_end == "2023-09-30"), None
    )
    assert fy23 is not None
    assert fy23.revenue_yoy_growth is not None
    expected = (383285e9 - 394328e9) / abs(394328e9)
    assert fy23.revenue_yoy_growth == pytest.approx(expected, rel=1e-4)


def test_extract_filings():
    """extract_filings should return only 10-K and 10-Q filings since start_date."""
    submissions = {
        "filings": {
            "recent": {
                "accessionNumber": ["0001234567-23-001", "0001234567-22-001", "0001234567-18-001"],
                "form": ["10-K", "10-Q", "10-K"],
                "filingDate": ["2023-11-03", "2022-08-05", "2018-11-05"],
                "reportDate": ["2023-09-30", "2022-07-02", "2018-09-30"],
                "primaryDocument": ["form.htm", "form.htm", "form.htm"],
                "documentCount": [88, 55, 77],
            }
        }
    }
    filings = extract_filings("320193", submissions)
    # 2018 filing is before START_DATE (2019-01-01), should be excluded
    assert len(filings) == 2
    forms = {f["form_type"] for f in filings}
    assert forms == {"10-K", "10-Q"}
