"""Page 3: Full per-company analysis — all signals + charts + brief."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import streamlit as st

from dashboard.components.charts import (
    financial_trend_chart,
    risk_score_gauge,
    signal_bar_chart,
)
from dashboard.components.metrics_row import render_metric_row, render_risk_badge
from dashboard.data_loader import (
    load_company_brief,
    load_final_risk_scores,
    load_financial_summary,
)
from dashboard.styles.theme import inject_css, page_header

st.set_page_config(
    page_title="Company Deep Dive", page_icon="🔍", layout="wide"
)

inject_css()
page_header(
    "🔍",
    "Company Deep Dive",
    "Full per-company analysis — all signals, trends, and AI brief",
)

df = load_final_risk_scores()
ts_df = load_financial_summary()

if df.empty:
    st.warning("No data available.")
    st.stop()

# ── Company Selector ──────────────────────────────────────────────────────────
labels = [
    f"{row['ticker']} — {row['company_name']} "
    f"({row.get('final_risk_tier', '?')} | {row.get('final_risk_score', 0):.0f}/100)"
    for _, row in df.sort_values("final_risk_score", ascending=False).iterrows()
    if pd.notna(row.get("ticker"))
]

if not labels:
    st.warning("No companies found.")
    st.stop()

selected_label = st.selectbox("Select Company", labels)
selected_ticker = selected_label.split(" — ")[0].strip()

company = df[df["ticker"] == selected_ticker]
if company.empty:
    st.warning(f"No data for {selected_ticker}")
    st.stop()

row = company.iloc[0]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"## {row.get('company_name', '')} ({selected_ticker})")
st.markdown(
    f"**Industry:** {row.get('industry_group', 'N/A')} | "
    f"**Period:** {row.get('latest_period', 'N/A')} | "
    f"**Risk:** {render_risk_badge(str(row.get('final_risk_tier', '?')))}"
)

st.divider()

# ── KPI Row ───────────────────────────────────────────────────────────────────
render_metric_row([
    {"label": "Final Risk Score", "value": f"{row.get('final_risk_score', 0):.1f}/100"},
    {"label": "ML Anomaly Score", "value": f"{row.get('ml_risk_score', 0):.1f}"},
    {"label": "Benford Score", "value": f"{row.get('benford_risk_score', 0):.1f}"},
    {"label": "Network Risk", "value": f"{row.get('network_risk', 0):.1f}"},
    {
        "label": "Signal Count",
        "value": int(row.get("total_signal_count", 0)),
        "help": "Number of concurrent risk signals triggered",
    },
])

st.divider()

# ── Gauges + Signal Breakdown ─────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    fig_gauge = risk_score_gauge(float(row.get("final_risk_score", 0)))
    st.plotly_chart(fig_gauge, width="stretch")

with col2:
    fig_bar = signal_bar_chart(row)
    st.plotly_chart(fig_bar, width="stretch")

st.divider()

# ── Risk Flags ────────────────────────────────────────────────────────────────
st.markdown("### 🚦 Risk Signal Flags")
flag_cols = {
    "flag_ar_acceleration": "AR Acceleration",
    "flag_cashflow_divergence": "Cash Flow Divergence",
    "flag_margin_compression": "Margin Compression",
    "flag_aggressive_recognition": "Aggressive Revenue Recognition",
    "flag_high_leverage": "High Leverage",
    "flag_liquidity_stress": "Liquidity Stress",
}

flag_row_cols = st.columns(3)
for i, (flag, label) in enumerate(flag_cols.items()):
    with flag_row_cols[i % 3]:
        triggered = bool(row.get(flag, False))
        icon = "🔴" if triggered else "✅"
        st.markdown(f"{icon} **{label}**")

st.divider()

# ── Financial Trends ──────────────────────────────────────────────────────────
if not ts_df.empty and "cik" in ts_df.columns:
    st.markdown("### 📈 Financial Trends")
    cik = str(row.get("cik", ""))

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        if "gross_margin" in ts_df.columns:
            st.plotly_chart(
                financial_trend_chart(ts_df, cik, "gross_margin", "Gross Margin"),
                width="stretch",
            )
        if "ar_to_revenue" in ts_df.columns:
            st.plotly_chart(
                financial_trend_chart(ts_df, cik, "ar_to_revenue", "AR / Revenue"),
                width="stretch",
            )
    with t_col2:
        if "operating_margin" in ts_df.columns:
            st.plotly_chart(
                financial_trend_chart(
                    ts_df, cik, "operating_margin", "Operating Margin"
                ),
                width="stretch",
            )
        if "revenue_yoy_growth" in ts_df.columns:
            st.plotly_chart(
                financial_trend_chart(
                    ts_df, cik, "revenue_yoy_growth", "Revenue YoY Growth"
                ),
                width="stretch",
            )

# ── AI Risk Brief ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 🤖 AI Risk Brief")
cik = str(row.get("cik", ""))
brief = load_company_brief(cik)

if brief:
    st.markdown(brief)
else:
    st.info(
        f"No AI brief generated yet. Run: "
        f"`python -m finsight360.cli brief --ticker {selected_ticker}`"
    )
