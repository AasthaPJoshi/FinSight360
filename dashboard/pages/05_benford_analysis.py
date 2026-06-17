"""Page 5: Benford's Law forensic analysis."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import plotly.express as px
import streamlit as st

from dashboard.components.charts import benford_deviation_chart
from dashboard.data_loader import load_benford_results
from dashboard.styles.theme import inject_css, page_header
from ml.benford import BENFORD_DISTRIBUTION

st.set_page_config(
    page_title="Benford Analysis", page_icon="📐", layout="wide"
)

inject_css()
page_header(
    "📐",
    "Benford's Law Analysis",
    "Forensic digit frequency analysis — Nigrini (2012)",
)

benford_df = load_benford_results()

if benford_df.empty:
    st.warning("No Benford results. Run: `python -m finsight360.cli ml-run`")
    st.stop()

# ── Summary KPIs ──────────────────────────────────────────────────────────────
conformity_counts = (
    benford_df["conformity_rating"].value_counts()
    if "conformity_rating" in benford_df.columns
    else {}
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Companies Analyzed", len(benford_df))
with col2:
    st.metric("Non-Conforming 🔴", conformity_counts.get("Non-conforming", 0))
with col3:
    st.metric("Marginal 🟡", conformity_counts.get("Marginal", 0))
with col4:
    st.metric("Conforming 🟢", conformity_counts.get("Conforming", 0))

st.divider()

# ── Conformity Distribution + MAD histogram ───────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    if not conformity_counts.empty:
        fig = px.pie(
            values=conformity_counts.values.tolist(),
            names=list(conformity_counts.index),
            color=list(conformity_counts.index),
            color_discrete_map={
                "Conforming": "#2E7D32",
                "Acceptable": "#1565C0",
                "Marginal": "#F9A825",
                "Non-conforming": "#C62828",
                "Insufficient Data": "#9E9E9E",
            },
            title="Benford Conformity Distribution",
        )
        fig.update_layout(
            height=320,
            font=dict(family="Inter, sans-serif", size=12, color="#B0BEC5"),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, width="stretch")

with col2:
    if "mad_score" in benford_df.columns:
        plot_data = benford_df.dropna(subset=["mad_score"])
        if not plot_data.empty:
            fig2 = px.histogram(
                plot_data,
                x="mad_score",
                nbins=20,
                title="MAD Score Distribution",
                labels={"mad_score": "Mean Absolute Deviation"},
                color_discrete_sequence=["#1565C0"],
            )
            fig2.add_vline(
                x=0.006,
                line_dash="dash",
                line_color="#F9A825",
                annotation_text="Marginal threshold",
            )
            fig2.add_vline(
                x=0.012,
                line_dash="dash",
                line_color="#C62828",
                annotation_text="Non-conforming threshold",
            )
            fig2.update_layout(
                height=320,
                font=dict(family="Inter, sans-serif", size=12, color="#B0BEC5"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                xaxis=dict(gridcolor="rgba(41,121,255,0.1)", linecolor="rgba(41,121,255,0.2)"),
                yaxis=dict(gridcolor="rgba(41,121,255,0.1)", linecolor="rgba(41,121,255,0.2)"),
                margin=dict(t=50, b=20, l=20, r=20),
            )
            st.plotly_chart(fig2, width="stretch")

st.divider()

# ── Company Selector ──────────────────────────────────────────────────────────
st.markdown("### Company Benford Analysis")

ticker_col = "ticker" if "ticker" in benford_df.columns else None
if ticker_col:
    selected_ticker = st.selectbox(
        "Select Company", benford_df[ticker_col].dropna().unique()
    )
    company_row = benford_df[benford_df[ticker_col] == selected_ticker]

    if not company_row.empty:
        r = company_row.iloc[0]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("MAD Score", f"{float(r.get('mad_score', 0)):.6f}")
        with c2:
            st.metric("Conformity Rating", str(r.get("conformity_rating", "N/A")))
        with c3:
            st.metric(
                "Benford Risk Score",
                f"{float(r.get('benford_risk_score', 0)):.1f}/100",
            )

        if r.get("interpretation"):
            st.info(str(r["interpretation"]))

        # Use uniform as placeholder observed (real data needs DB query)
        obs = {i: 1 / 9 for i in range(1, 10)}
        fig = benford_deviation_chart(
            obs,
            BENFORD_DISTRIBUTION,
            str(r.get("company_name", selected_ticker)),
        )
        st.plotly_chart(fig, width="stretch")

st.divider()

# ── Full Table ────────────────────────────────────────────────────────────────
st.markdown("### All Companies — Benford Results")
display_cols = [
    c
    for c in [
        "ticker",
        "company_name",
        "n_observations",
        "mad_score",
        "conformity_rating",
        "benford_risk_score",
    ]
    if c in benford_df.columns
]
if display_cols:
    st.dataframe(benford_df[display_cols], use_container_width=True)
