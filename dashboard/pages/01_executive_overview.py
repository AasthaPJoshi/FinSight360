"""Page 1: Executive Overview — top-level KPIs and risk summary."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.charts import industry_risk_heatmap, scatter_risk_landscape
from dashboard.components.metrics_row import render_metric_row
from dashboard.components.risk_table import render_risk_table
from dashboard.data_loader import load_final_risk_scores
from dashboard.styles.theme import inject_css, page_header, TIER_COLORS

st.set_page_config(
    page_title="Executive Overview", page_icon="📋", layout="wide"
)

inject_css()
page_header(
    "📋",
    "Executive Overview",
    "Real-time risk intelligence across all monitored companies",
)

df = load_final_risk_scores()

if df.empty:
    st.warning("No data. Run: `python -m finsight360.cli ml-run`")
    st.stop()

# ── KPI Row ───────────────────────────────────────────────────────────────────
with st.spinner("Loading risk intelligence..."):
    total = len(df)
    critical = int((df["final_risk_tier"] == "CRITICAL").sum()) if "final_risk_tier" in df.columns else 0
    high = int((df["final_risk_tier"] == "HIGH").sum()) if "final_risk_tier" in df.columns else 0
    anomalies = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
    avg_score = float(df["final_risk_score"].mean()) if "final_risk_score" in df.columns else 0.0

    render_metric_row([
        {
            "label": "Companies Monitored",
            "value": total,
            "help": "Total companies in FinSight360 database",
        },
        {
            "label": "CRITICAL Risk",
            "value": critical,
            "delta": f"{critical / total * 100:.0f}% of total" if total else None,
            "delta_color": "inverse" if critical > 0 else "off",
            "help": "Final risk score > 70",
        },
        {
            "label": "HIGH Risk",
            "value": high,
            "delta": f"{high / total * 100:.0f}%" if total else None,
            "delta_color": "inverse" if high > 0 else "off",
        },
        {
            "label": "ML Anomalies",
            "value": anomalies,
            "help": "Isolation Forest anomaly classification",
        },
        {
            "label": "Avg Risk Score",
            "value": f"{avg_score:.1f}/100",
            "help": "Average final combined risk score",
        },
    ])

st.divider()

_DARK = dict(
    font=dict(family="Inter, sans-serif", size=12, color="#B0BEC5"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    xaxis=dict(gridcolor="rgba(41,121,255,0.1)", linecolor="rgba(41,121,255,0.2)", tickfont=dict(color="#B0BEC5")),
    yaxis=dict(gridcolor="rgba(41,121,255,0.1)", linecolor="rgba(41,121,255,0.2)", tickfont=dict(color="#B0BEC5")),
    legend=dict(bgcolor="rgba(13,27,42,0.8)", bordercolor="rgba(41,121,255,0.2)", font=dict(color="#B0BEC5")),
)

# ── Risk Distribution + Landscape ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Risk Tier Distribution")
    if "final_risk_tier" in df.columns:
        tier_counts = df["final_risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Count"]
        tier_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL", "CLEAN"]
        tier_counts["Risk Tier"] = pd.Categorical(
            tier_counts["Risk Tier"],
            categories=[t for t in tier_order if t in tier_counts["Risk Tier"].values],
            ordered=True,
        )
        tier_counts = tier_counts.sort_values("Risk Tier")

        fig = px.bar(
            tier_counts,
            x="Risk Tier",
            y="Count",
            color="Risk Tier",
            color_discrete_map=TIER_COLORS,
            title="Companies by Risk Tier",
        )
        fig.update_layout(
            showlegend=False,
            height=320,
            margin=dict(t=50, b=20, l=20, r=20),
            **_DARK,
        )
        st.plotly_chart(fig, width="stretch")

with col2:
    st.markdown("### Risk Landscape")
    fig2 = scatter_risk_landscape(df)
    fig2.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20), **_DARK)
    st.plotly_chart(fig2, width="stretch")

st.divider()

# ── Industry Heatmap ──────────────────────────────────────────────────────────
st.markdown("### Industry Risk Heatmap")
fig3 = industry_risk_heatmap(df)
if fig3.data:
    fig3.update_layout(margin=dict(t=40, b=20, l=20, r=20), **_DARK)
    st.plotly_chart(fig3, width="stretch")
else:
    st.info("Industry data not available — run graph-build to populate industry_group.")

st.divider()

# ── Top Risk Companies Table ──────────────────────────────────────────────────
st.markdown("### 🚨 Top Risk Companies")
render_risk_table(df, n_rows=15)
