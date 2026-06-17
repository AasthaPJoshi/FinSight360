"""
FinSight360 — Demo Data Seeder
Seeds 10 realistic companies into DuckDB so all dashboard pages
show meaningful data without running the full SEC EDGAR pipeline.
"""
import os
import sys
import uuid
import random
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("DUCKDB_PATH", "data/finsight360.duckdb")

from storage.database import Database

COMPANIES = [
    ("0000320193", "AAPL", "Apple Inc.",              "Technology",              "CA"),
    ("0000789019", "MSFT", "Microsoft Corporation",   "Technology",              "WA"),
    ("0001318605", "TSLA", "Tesla Inc.",               "Consumer Discretionary",  "TX"),
    ("0001018724", "AMZN", "Amazon.com Inc.",          "Consumer Discretionary",  "WA"),
    ("0001326801", "META", "Meta Platforms Inc.",      "Communication Services",  "CA"),
    ("0001045810", "NVDA", "NVIDIA Corporation",       "Technology",              "CA"),
    ("0000019617", "JPM",  "JPMorgan Chase & Co.",     "Financials",              "NY"),
    ("0000886982", "GS",   "Goldman Sachs Group Inc.", "Financials",              "NY"),
    ("0000200406", "JNJ",  "Johnson & Johnson",        "Healthcare",              "NJ"),
    ("0000034088", "XOM",  "Exxon Mobil Corporation",  "Energy",                  "TX"),
]

# Per-company financial profile: gross_margin, op_margin, net_margin,
# revenue_growth, late_filings, revenue_base (billions)
PROFILES = {
    "AAPL": (0.44, 0.30, 0.25,  0.08, 0, 383e9),
    "MSFT": (0.69, 0.43, 0.36,  0.16, 0, 212e9),
    "TSLA": (0.18, 0.09, 0.08,  0.51, 1,  97e9),
    "AMZN": (0.47, 0.06, 0.04,  0.12, 0, 575e9),
    "META": (0.81, 0.35, 0.29,  0.23, 0, 135e9),
    "NVDA": (0.74, 0.55, 0.49,  1.22, 0,  61e9),
    "JPM":  (0.62, 0.32, 0.25,  0.14, 0, 162e9),
    "GS":   (0.58, 0.28, 0.22,  0.05, 2,  47e9),
    "JNJ":  (0.68, 0.22, 0.16,  0.06, 0,  93e9),
    "XOM":  (0.12, 0.10, 0.08, -0.03, 1, 400e9),
}

ML_SCORES = {
    "AAPL": 18.0, "MSFT": 12.0, "TSLA": 74.0, "AMZN": 35.0,
    "META": 22.0, "NVDA": 28.0, "JPM":  45.0, "GS":   78.0,
    "JNJ":  20.0, "XOM":  55.0,
}

BENFORD = {
    "AAPL": (0.004, 0.82, "Conforming"),
    "MSFT": (0.005, 0.76, "Conforming"),
    "TSLA": (0.018, 0.03, "Non-conforming"),
    "AMZN": (0.007, 0.51, "Acceptable"),
    "META": (0.005, 0.79, "Conforming"),
    "NVDA": (0.006, 0.64, "Conforming"),
    "JPM":  (0.013, 0.08, "Marginal"),
    "GS":   (0.021, 0.01, "Non-conforming"),
    "JNJ":  (0.004, 0.88, "Conforming"),
    "XOM":  (0.011, 0.14, "Marginal"),
}

NET_SCORES = {
    "AAPL": 12.0, "MSFT": 10.0, "TSLA": 25.0, "AMZN": 18.0,
    "META": 14.0, "NVDA": 16.0, "JPM":  55.0, "GS":   72.0,
    "JNJ":  20.0, "XOM":  38.0,
}

INTERP = {
    "Conforming":     "First-digit distribution consistent with Benford's Law.",
    "Acceptable":     "Minor deviations within acceptable range.",
    "Marginal":       "Notable deviations; warrants closer review.",
    "Non-conforming": "Significant deviation — elevated fraud risk.",
}


def _final(ticker: str) -> tuple[float, str, bool]:
    ml = ML_SCORES[ticker]
    bscore = (1 - BENFORD[ticker][1]) * 100
    net = NET_SCORES[ticker]
    f = 0.50 * ml + 0.30 * bscore + 0.20 * net
    tier = ("CRITICAL" if f >= 85 else "HIGH" if f >= 70 else
            "MEDIUM" if f >= 50 else "LOW" if f >= 30 else "MINIMAL")
    return round(f, 1), tier, f >= 65


def seed(db: Database) -> None:
    conn = db.get_connection()
    now = datetime.now()

    print("Seeding demo data into DuckDB...")

    # ── 1. Companies ──────────────────────────────────────────────────────
    for cik, ticker, name, industry, state in COMPANIES:
        conn.execute("""
            INSERT INTO companies (cik, name, ticker, sic_code,
                                   state_of_incorporation, created_at, updated_at)
            VALUES (?, ?, ?, '0000', ?, ?, ?)
            ON CONFLICT (cik) DO UPDATE SET updated_at = excluded.updated_at
        """, [cik, name, ticker, state, now, now])
    print(f"  ✓ {len(COMPANIES)} companies")

    # ── 2. Ingestion run ──────────────────────────────────────────────────
    conn.execute("""
        INSERT INTO ingestion_runs (id, started_at, completed_at, status,
            companies_processed, filings_processed, facts_processed)
        VALUES (?, ?, ?, 'completed', ?, ?, ?)
    """, [str(uuid.uuid4()), now - timedelta(minutes=5), now,
          len(COMPANIES), len(COMPANIES) * 4, len(COMPANIES) * 80])
    print("  ✓ 1 ingestion_run")

    # ── 3. Financial metrics (8 annual 10-K periods per company) ─────────
    ciks = [c[0] for c in COMPANIES]
    conn.execute(
        f"DELETE FROM financial_metrics WHERE cik IN ({','.join(['?']*len(ciks))})",
        ciks,
    )
    count = 0
    current_year = now.year
    for cik, ticker, name, industry, state in COMPANIES:
        gm, om, nm, rg, lf, rev_base = PROFILES[ticker]
        for yr in range(8):  # 8 annual periods, one per year
            random.seed(hash(ticker + str(yr)))
            n = lambda base, spread=0.08: base * (1 + random.uniform(-spread, spread))
            # Use fiscal year-end date (Dec 31) for each historical year
            year = current_year - yr
            from datetime import date
            period_end = date(year, 12, 31)
            revenues = round(n(rev_base), 0)  # annual

            conn.execute("""
                INSERT INTO financial_metrics (
                    id, cik, period_end, form,
                    revenues, gross_profit, operating_income, net_income,
                    total_assets, current_assets, total_liabilities,
                    current_liabilities, stockholders_equity,
                    cash, accounts_receivable, inventory,
                    long_term_debt, rd_expense, sga_expense,
                    operating_cashflow, capex, revenue_yoy_growth, computed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (id) DO NOTHING
            """, [
                str(uuid.uuid4()), cik, period_end, "10-K",
                revenues,
                round(revenues * n(gm), 0),
                round(revenues * n(om), 0),
                round(revenues * n(nm), 0),
                round(revenues * n(2.1), 0),       # total_assets
                round(revenues * n(0.6), 0),        # current_assets
                round(revenues * n(1.4), 0),        # total_liabilities
                round(revenues * n(0.4), 0),        # current_liabilities
                round(revenues * n(0.7), 0),        # stockholders_equity
                round(revenues * n(0.15), 0),       # cash
                round(revenues * n(0.12), 0),       # accounts_receivable
                round(revenues * n(0.05), 0),       # inventory
                round(revenues * n(0.30), 0),       # long_term_debt
                round(revenues * n(0.05), 0),       # rd_expense
                round(revenues * n(0.08), 0),       # sga_expense
                round(revenues * n(nm + 0.05), 0),  # operating_cashflow
                round(revenues * n(0.04), 0),       # capex
                round(n(rg, 0.20), 4),              # revenue_yoy_growth
                now,
            ])
            count += 1
    print(f"  ✓ {count} financial_metrics rows")

    # ── 4. ML anomaly scores ──────────────────────────────────────────────
    conn.execute(
        f"DELETE FROM ml_anomaly_scores WHERE cik IN ({','.join(['?']*len(ciks))})",
        ciks,
    )
    for cik, ticker, name, industry, state in COMPANIES:
        ml = ML_SCORES[ticker]
        final, tier, anomaly = _final(ticker)
        conn.execute("""
            INSERT INTO ml_anomaly_scores (
                cik, ticker, company_name, industry_group,
                isolation_forest_score, anomaly_percentile,
                ml_risk_score, is_anomaly, pre_ml_risk_score,
                risk_tier, computed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, [cik, ticker, name, industry,
              round(-(ml / 100 - 0.5), 4),
              round(ml, 1),
              ml, anomaly,
              round(ml * 0.9, 1),
              tier, now])
    print(f"  ✓ {len(COMPANIES)} ml_anomaly_scores")

    # ── 5. Benford results ────────────────────────────────────────────────
    conn.execute(
        f"DELETE FROM benford_results WHERE cik IN ({','.join(['?']*len(ciks))})",
        ciks,
    )
    for cik, ticker, name, industry, state in COMPANIES:
        mad, pval, conformity = BENFORD[ticker]
        bscore = round((1 - pval) * 100, 1)
        chi_stat = round(-2 * __import__('math').log(max(pval, 1e-10)) * 8, 2)
        conn.execute("""
            INSERT INTO benford_results (
                cik, company_name, ticker, n_observations,
                mad_score, chi_square_statistic, chi_square_p_value,
                conformity_rating, benford_risk_score,
                suspicious_digits, interpretation, computed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, [cik, name, ticker, 240, mad, chi_stat, pval,
              conformity, bscore,
              "1,7" if conformity == "Non-conforming" else "",
              INTERP[conformity], now])
    print(f"  ✓ {len(COMPANIES)} benford_results")

    # ── 6. Final risk scores ──────────────────────────────────────────────
    try:
        conn.execute("DROP TABLE IF EXISTS final_risk_scores")
    except Exception:
        pass
    conn.execute("""
        CREATE TABLE final_risk_scores (
            cik                VARCHAR PRIMARY KEY,
            ticker             VARCHAR,
            company_name       VARCHAR,
            industry_group     VARCHAR,
            ml_risk_score      DOUBLE,
            benford_risk_score DOUBLE,
            network_risk_score DOUBLE,
            final_risk_score   DOUBLE,
            final_risk_tier    VARCHAR,
            is_anomaly         BOOLEAN,
            community_id       INTEGER DEFAULT -1,
            late_filings_count INTEGER DEFAULT 0,
            total_signal_count INTEGER DEFAULT 0,
            computed_at        TIMESTAMP DEFAULT current_timestamp
        )
    """)
    for cik, ticker, name, industry, state in COMPANIES:
        final, tier, anomaly = _final(ticker)
        mad, pval, _ = BENFORD[ticker]
        bscore = round((1 - pval) * 100, 1)
        gm, om, nm, rg, lf, _ = PROFILES[ticker]
        signals = int(ML_SCORES[ticker] > 60) + int(pval < 0.05) + int(NET_SCORES[ticker] > 50)
        conn.execute("""
            INSERT INTO final_risk_scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, [cik, ticker, name, industry,
              ML_SCORES[ticker], bscore, NET_SCORES[ticker],
              final, tier, anomaly,
              abs(hash(ticker)) % 5,
              lf, signals, now])
    print(f"  ✓ {len(COMPANIES)} final_risk_scores")

    # ── 6b. Create dbt-equivalent mart views from financial_metrics ───────
    # These mirror what dbt would produce. Allows dashboard to work without dbt.
    conn.execute("CREATE SCHEMA IF NOT EXISTS main_marts")
    conn.execute("CREATE SCHEMA IF NOT EXISTS main_staging")

    # mart_financial_summary — time-series financial data per company
    conn.execute("DROP VIEW IF EXISTS main_marts.mart_financial_summary")
    conn.execute("""
        CREATE VIEW main_marts.mart_financial_summary AS
        SELECT
            m.cik,
            c.ticker,
            c.name AS company_name,
            c.sic_description AS industry_group,
            m.period_end,
            EXTRACT(year FROM m.period_end)::INTEGER AS fiscal_year,
            EXTRACT(quarter FROM m.period_end)::INTEGER AS fiscal_quarter,
            COALESCE(m.form, '10-K') AS form,
            m.revenues,
            CASE WHEN m.revenues > 0 THEN m.gross_profit / m.revenues ELSE NULL END AS gross_margin,
            CASE WHEN m.revenues > 0 THEN m.operating_income / m.revenues ELSE NULL END AS operating_margin,
            CASE WHEN m.revenues > 0 THEN m.net_income / m.revenues ELSE NULL END AS net_margin,
            CASE WHEN m.revenues > 0 THEN m.accounts_receivable / m.revenues ELSE NULL END AS ar_to_revenue,
            CASE WHEN m.revenues > 0 AND m.net_income IS NOT NULL AND m.operating_cashflow IS NOT NULL
                 THEN (m.operating_cashflow - m.net_income) / m.revenues ELSE NULL END AS cash_earnings_gap,
            CASE WHEN m.current_liabilities > 0 THEN m.current_assets / m.current_liabilities ELSE NULL END AS current_ratio,
            CASE WHEN m.stockholders_equity > 0 THEN m.long_term_debt / m.stockholders_equity ELSE NULL END AS debt_to_equity,
            m.revenue_yoy_growth,
            0 AS total_signal_count,
            0 AS flag_ar_acceleration,
            0 AS flag_cashflow_divergence,
            0 AS flag_margin_compression,
            0 AS flag_aggressive_recognition,
            0 AS flag_high_leverage,
            0 AS flag_liquidity_stress,
            current_timestamp AS mart_computed_at
        FROM financial_metrics m
        LEFT JOIN companies c ON m.cik = c.cik
        ORDER BY m.cik, m.period_end
    """)

    # mart_risk_candidates — latest period per company with risk signals
    conn.execute("DROP VIEW IF EXISTS main_marts.mart_risk_candidates")
    conn.execute("""
        CREATE VIEW main_marts.mart_risk_candidates AS
        WITH latest AS (
            SELECT m.*,
                   ROW_NUMBER() OVER (PARTITION BY m.cik ORDER BY m.period_end DESC) AS rn
            FROM financial_metrics m
        ),
        latest1 AS (SELECT * FROM latest WHERE rn = 1),
        filings_agg AS (
            SELECT cik,
                   COUNT(*) FILTER (WHERE filing_date IS NOT NULL) AS late_filings_count,
                   MAX(DATEDIFF('day', period_of_report, filing_date)) AS max_reporting_lag_days
            FROM filings GROUP BY cik
        )
        SELECT
            l.cik,
            c.name AS company_name,
            c.ticker,
            c.sic_description AS industry_group,
            l.period_end AS latest_period,
            l.revenues,
            CASE WHEN l.revenues > 0 THEN l.gross_profit / l.revenues ELSE NULL END AS gross_margin,
            CASE WHEN l.revenues > 0 THEN l.operating_income / l.revenues ELSE NULL END AS operating_margin,
            CASE WHEN l.revenues > 0 THEN l.net_income / l.revenues ELSE NULL END AS net_margin,
            CASE WHEN l.current_liabilities > 0 THEN l.current_assets / l.current_liabilities ELSE 1.5 END AS current_ratio,
            CASE WHEN l.stockholders_equity > 0 THEN l.long_term_debt / l.stockholders_equity ELSE 0.5 END AS debt_to_equity,
            CASE WHEN l.revenues > 0 THEN COALESCE(l.accounts_receivable, 0) / l.revenues ELSE 0.1 END AS ar_to_revenue,
            l.revenue_yoy_growth,
            CASE WHEN l.revenues > 0 AND l.operating_cashflow IS NOT NULL AND l.net_income IS NOT NULL
                 THEN (l.operating_cashflow - l.net_income) / l.revenues ELSE 0 END AS cash_earnings_gap,
            0 AS flag_ar_acceleration,
            0 AS flag_cashflow_divergence,
            0 AS flag_margin_compression,
            0 AS flag_aggressive_recognition,
            CASE WHEN l.stockholders_equity > 0 AND l.long_term_debt / l.stockholders_equity > 2 THEN 1 ELSE 0 END AS flag_high_leverage,
            CASE WHEN l.current_liabilities > 0 AND l.current_assets / l.current_liabilities < 1 THEN 1 ELSE 0 END AS flag_liquidity_stress,
            0 AS total_signal_count,
            COALESCE(fa.late_filings_count, 0) > 0 AS has_late_filings,
            COALESCE(fa.late_filings_count, 0) AS late_filings_count,
            COALESCE(fa.max_reporting_lag_days, 0) AS max_reporting_lag_days,
            50.0 AS pre_ml_risk_score,
            'MEDIUM' AS risk_tier,
            'None' AS dominant_signal,
            current_timestamp AS mart_computed_at,
            ROW_NUMBER() OVER (ORDER BY l.revenues DESC) AS risk_rank,
            ROW_NUMBER() OVER (PARTITION BY c.sic_description ORDER BY l.revenues DESC) AS risk_rank_in_industry
        FROM latest1 l
        LEFT JOIN companies c ON l.cik = c.cik
        LEFT JOIN filings_agg fa ON l.cik = fa.cik
    """)

    # mart_company_dashboard — enriched dashboard view
    conn.execute("DROP VIEW IF EXISTS main_marts.mart_company_dashboard")
    conn.execute("""
        CREATE VIEW main_marts.mart_company_dashboard AS
        SELECT
            r.cik, r.ticker, r.company_name, r.industry_group,
            r.risk_tier, r.pre_ml_risk_score, r.risk_rank, r.risk_rank_in_industry,
            r.dominant_signal, r.total_signal_count,
            r.latest_period, r.revenues,
            r.gross_margin, r.operating_margin, r.net_margin,
            r.current_ratio, r.debt_to_equity, r.ar_to_revenue, r.revenue_yoy_growth,
            r.has_late_filings, r.late_filings_count,
            r.gross_margin AS t4q_gross_margin,
            r.operating_margin AS t4q_operating_margin,
            r.net_margin AS t4q_net_margin,
            r.revenue_yoy_growth AS t4q_revenue_growth,
            r.revenues AS t4q_total_revenues,
            0.0 AS gross_margin_vs_peers,
            0.0 AS leverage_vs_peers,
            AVG(r.gross_margin) OVER (PARTITION BY r.industry_group) AS industry_med_gross_margin,
            AVG(r.operating_margin) OVER (PARTITION BY r.industry_group) AS industry_med_operating_margin,
            current_timestamp AS dashboard_computed_at
        FROM main_marts.mart_risk_candidates r
    """)

    # Staging views
    conn.execute("DROP VIEW IF EXISTS main_staging.stg_companies")
    conn.execute("""
        CREATE VIEW main_staging.stg_companies AS
        SELECT cik, name AS company_name, ticker, sic_code, sic_description AS industry_group,
               state_of_incorporation, fiscal_year_end, created_at
        FROM companies
    """)
    print("  ✓ dbt mart views created (main_marts schema)")

    # ── 7. Audit trail ────────────────────────────────────────────────────
    from governance.audit_trail import AuditTrailManager, AuditEntry
    mgr = AuditTrailManager(db)
    for cik, ticker, name, industry, state in COMPANIES:
        final, tier, anomaly = _final(ticker)
        mad, pval, _ = BENFORD[ticker]
        entry = AuditEntry(
            event_type="risk_score_generated",
            cik=cik, ticker=ticker, company_name=name,
            model_name="finsight360_v1", model_version="1.0.0",
            ml_risk_score=ML_SCORES[ticker],
            benford_risk_score=round((1 - pval) * 100, 1),
            network_risk=NET_SCORES[ticker],
            final_risk_score=final, final_risk_tier=tier, is_anomaly=anomaly,
            total_signals=int(ML_SCORES[ticker] > 60) + int(pval < 0.05),
        )
        mgr.log_scoring_event(entry)
    print(f"  ✓ {len(COMPANIES)} audit_trail events")

    # ── 8. Risk briefs (Gemini-style placeholder text) ────────────────────
    conn.execute("CREATE TABLE IF NOT EXISTS risk_briefs (cik VARCHAR PRIMARY KEY, company_name VARCHAR, ticker VARCHAR, final_risk_score DOUBLE, brief_text TEXT, generated_at TIMESTAMP DEFAULT current_timestamp)")
    BRIEF_TEMPLATES = {
        "AAPL": "Apple Inc. demonstrates strong financial controls with conforming Benford's Law distribution (MAD=0.004). Revenue growth of 8% YoY supported by services diversification. Low anomaly risk. Recommend: MONITOR.",
        "MSFT": "Microsoft Corp shows exceptional margin quality (69% gross margin) with clean Benford signal. Cloud segment Azure continues compounding. Risk tier: LOW. No material concerns identified.",
        "TSLA": "Tesla Inc. flagged as HIGH RISK. Non-conforming Benford score (p=0.03) combined with Isolation Forest anomaly (score: 74/100). AR acceleration and late filing noted. Recommend: INVESTIGATE revenue recognition.",
        "AMZN": "Amazon.com shows acceptable Benford conformity with strong revenue scale ($575B). Operating margin remains thin at 6%. Network centrality elevated. Risk tier: MEDIUM — monitor AWS margin divergence.",
        "META": "Meta Platforms exhibits strong margin profile (81% gross margin) with conforming forensic signals. AI capex acceleration warrants monitoring but fundamentals remain sound. Risk: LOW.",
        "NVDA": "NVIDIA Corp. flagged for rapid revenue scaling (+122% YoY). While Benford conforming, growth velocity warrants ongoing monitoring. Network risk moderate. Overall: MEDIUM — growth-driven anomaly, not fraud signal.",
        "JPM": "JPMorgan Chase shows marginal Benford conformity (p=0.08) — common for large diversified financials. Network centrality very high (systemic institution). Risk tier: MEDIUM — systemic, not operational.",
        "GS": "Goldman Sachs flagged CRITICAL. Non-conforming Benford (p=0.01, MAD=0.021) combined with ML anomaly score 78/100 and 2 late filings. Recommend: FULL REVIEW of trading revenue recognition.",
        "JNJ": "Johnson & Johnson demonstrates clean financials with conforming Benford signal and stable margins. Litigation reserves adequately disclosed. Risk tier: LOW. Long-term hold fundamentals.",
        "XOM": "Exxon Mobil shows elevated network risk (55/100) linked to commodity price sensitivity. Revenue decline (-3% YoY). One late filing noted. Non-conforming Benford score warrants forensic review of asset valuations.",
    }
    for cik, ticker, name, industry, state in COMPANIES:
        final, tier, _ = _final(ticker)
        conn.execute("""
            INSERT INTO risk_briefs (cik, company_name, ticker, final_risk_score, brief_text, generated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (cik) DO UPDATE SET
                brief_text = excluded.brief_text,
                final_risk_score = excluded.final_risk_score,
                generated_at = excluded.generated_at
        """, [cik, name, ticker, final, BRIEF_TEMPLATES.get(ticker, "Risk brief pending AI analysis."), now])
    print(f"  ✓ {len(COMPANIES)} risk_briefs")

    # ── 9. SHAP values (synthetic feature importance) ─────────────────────
    conn.execute("CREATE TABLE IF NOT EXISTS shap_values (id VARCHAR PRIMARY KEY, cik VARCHAR NOT NULL, ticker VARCHAR, company_name VARCHAR, feature_name VARCHAR NOT NULL, friendly_name VARCHAR, shap_value DOUBLE, abs_shap_value DOUBLE, feature_value DOUBLE, direction VARCHAR, rank_in_company INTEGER, computed_at TIMESTAMP DEFAULT current_timestamp)")
    FEATURES = [
        ("gross_margin", "Gross Margin"),
        ("operating_margin", "Operating Margin"),
        ("net_margin", "Net Margin"),
        ("revenue_yoy_growth", "Revenue YoY Growth"),
        ("ar_to_revenue", "AR / Revenue Ratio"),
        ("current_ratio", "Current Ratio"),
        ("debt_to_equity", "Debt-to-Equity"),
        ("total_signal_count", "Total ML Signals"),
    ]
    SHAP_WEIGHTS = {
        "AAPL": [0.12, 0.18, 0.09, 0.07, -0.04, 0.06, 0.03, 0.02],
        "MSFT": [0.22, 0.25, 0.18, 0.11, -0.03, 0.07, 0.02, 0.01],
        "TSLA": [-0.08, -0.15, -0.12, 0.31, 0.22, -0.18, 0.25, 0.28],
        "AMZN": [0.15, -0.08, -0.06, 0.14, 0.09, 0.05, 0.07, 0.06],
        "META": [0.28, 0.22, 0.19, 0.16, -0.05, 0.08, -0.04, 0.03],
        "NVDA": [0.24, 0.31, 0.27, 0.35, -0.06, 0.09, -0.05, 0.08],
        "JPM":  [0.18, 0.14, 0.11, 0.09, 0.16, -0.12, 0.21, 0.15],
        "GS":   [-0.12, -0.18, -0.14, -0.06, 0.29, -0.22, 0.35, 0.32],
        "JNJ":  [0.21, 0.16, 0.13, 0.07, -0.04, 0.11, 0.04, 0.02],
        "XOM":  [-0.06, -0.04, -0.03, -0.11, 0.18, -0.08, 0.22, 0.19],
    }
    shap_rows = 0
    for cik, ticker, name, industry, state in COMPANIES:
        weights = SHAP_WEIGHTS.get(ticker, [0.1] * len(FEATURES))
        gm, om, nm, rg, lf, rev = PROFILES[ticker]
        feat_vals = [gm, om, nm, rg, accounts_receivable_proxy := rg * 0.15, 1.8 - rg, lf * 2.0, int(ML_SCORES[ticker] > 60)]
        for rank, ((fname, friendly), sv, fv) in enumerate(zip(FEATURES, weights, feat_vals), 1):
            row_id = f"{cik}_{fname}"
            conn.execute("""
                INSERT INTO shap_values VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (id) DO NOTHING
            """, [row_id, cik, ticker, name, fname, friendly,
                  sv, abs(sv), fv,
                  "increases_risk" if sv > 0 else "decreases_risk",
                  rank, now])
            shap_rows += 1
    print(f"  ✓ {shap_rows} shap_values")

    # ── Summary ───────────────────────────────────────────────────────────
    high = sum(1 for _, t, *_ in COMPANIES if _final(t)[1] in ("HIGH", "CRITICAL"))
    nonconf = sum(1 for _, t, *_ in COMPANIES if BENFORD[t][2] == "Non-conforming")
    print(f"\nDemo data seeded successfully.")
    print(f"  Companies:          {len(COMPANIES)}")
    print(f"  HIGH/CRITICAL:      {high}")
    print(f"  Non-conforming:     {nonconf}")
    print(f"\nRun: streamlit run dashboard/app.py")


if __name__ == "__main__":
    db = Database()
    try:
        seed(db)
    finally:
        db.close()
