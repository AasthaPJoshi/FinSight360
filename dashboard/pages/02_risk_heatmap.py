"""Page 2: Interactive risk heatmap with filtering."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import plotly.express as px
import streamlit as st

from dashboard.components.risk_table import render_risk_table
from dashboard.data_loader import load_final_risk_scores
from dashboard.styles.theme import inject_css, page_header, TIER_COLORS

st.set_page_config(page_title="Risk Heatmap", page_icon="🌡️", layout="wide")

inject_css()
page_header(
    "🌡️",
    "Risk Heatmap",
    "Filter and explore company risk across sectors",
)

df = load_final_risk_scores()

if df.empty:
    st.warning("No data available.")
    st.stop()

_DARK = dict(
    font=dict(family="Inter, sans-serif", size=12, color="#B0BEC5"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    xaxis=dict(gridcolor="rgba(41,121,255,0.1)", linecolor="rgba(41,121,255,0.2)", tickfont=dict(color="#B0BEC5")),
    yaxis=dict(gridcolor="rgba(41,121,255,0.1)", linecolor="rgba(41,121,255,0.2)", tickfont=dict(color="#B0BEC5")),
    legend=dict(bgcolor="rgba(13,27,42,0.8)", bordercolor="rgba(41,121,255,0.2)", font=dict(color="#B0BEC5")),
)

# ── Filters ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    industries = ["All"] + sorted(
        df["industry_group"].dropna().unique().tolist()
    ) if "industry_group" in df.columns else ["All"]
    selected_industry = st.selectbox("Industry", industries)

with col2:
    tiers = ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL", "CLEAN"]
    selected_tier = st.selectbox("Risk Tier", tiers)

with col3:
    min_score = st.slider("Minimum Risk Score", 0, 100, 0)

# Apply filters
filtered = df.copy()
if selected_industry != "All" and "industry_group" in filtered.columns:
    filtered = filtered[filtered["industry_group"] == selected_industry]
if selected_tier != "All" and "final_risk_tier" in filtered.columns:
    filtered = filtered[filtered["final_risk_tier"] == selected_tier]
if "final_risk_score" in filtered.columns:
    filtered = filtered[filtered["final_risk_score"] >= min_score]

st.markdown(f"**{len(filtered)} companies** match filters")
st.divider()

# ── Treemap ───────────────────────────────────────────────────────────────────
st.markdown("### Risk Treemap")
if not filtered.empty and "industry_group" in filtered.columns:
    treemap_df = filtered.dropna(subset=["industry_group", "final_risk_score"])
    if not treemap_df.empty:
        fig = px.treemap(
            treemap_df,
            path=[px.Constant("All Companies"), "industry_group", "ticker"],
            values="final_risk_score",
            color="final_risk_score",
            color_continuous_scale=["#0D47A1", "#C62828"],
            hover_data=["company_name", "final_risk_tier"],
            title="Company Risk by Industry (size = risk score)",
        )
        fig.update_layout(
            height=500,
            font=dict(family="Inter, sans-serif", size=12, color="#B0BEC5"),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No data with industry_group — run graph-build to populate.")
else:
    if not filtered.empty:
        fig = px.bar(
            filtered.head(30),
            x="ticker",
            y="final_risk_score",
            color="final_risk_tier",
            color_discrete_map=TIER_COLORS,
            title="Risk Scores by Company",
        )
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=30, l=20, r=20),
            **_DARK,
        )
        st.plotly_chart(fig, width="stretch")

st.divider()
st.markdown("### Filtered Company List")
render_risk_table(filtered)
