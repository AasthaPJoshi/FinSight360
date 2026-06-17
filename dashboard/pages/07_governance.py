"""
Page 07 — Responsible AI Governance
Tabs: Audit Trail | SHAP Explainability | Bias Analysis | Model Card & KQL
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import streamlit as st

from storage.database import Database
from dashboard.styles.theme import inject_css, page_header

try:
    from governance.kql_queries import KQL_QUERIES, KQL_TABLE_SCHEMA
    from governance.audit_trail import AuditTrailManager
    from governance.bias_analyzer import BiasAnalyzer
    from governance.model_card import generate_model_card
    GOVERNANCE_AVAILABLE = True
except ImportError:
    GOVERNANCE_AVAILABLE = False
    KQL_QUERIES = {}
    KQL_TABLE_SCHEMA = ""

os.environ.setdefault("DUCKDB_PATH", "data/finsight360.duckdb")

st.set_page_config(
    page_title="FinSight360 — Governance",
    page_icon="⚖️",
    layout="wide",
)

inject_css()
page_header(
    "⚖️",
    "Responsible AI Governance",
    "Audit trail · SHAP · Bias monitoring · Model card · SR 11-7 · EU AI Act",
)


@st.cache_resource
def get_db():
    return Database()


@st.cache_data(ttl=300)
def load_audit_trail(days_back: int = 30) -> pd.DataFrame:
    if not GOVERNANCE_AVAILABLE:
        return pd.DataFrame()
    db = get_db()
    mgr = AuditTrailManager(db)
    return mgr.get_audit_log(days_back=days_back)


@st.cache_data(ttl=300)
def load_audit_stats() -> dict:
    if not GOVERNANCE_AVAILABLE:
        return {}
    db = get_db()
    mgr = AuditTrailManager(db)
    return mgr.get_summary_stats()


@st.cache_data(ttl=300)
def load_final_scores() -> pd.DataFrame:
    db = get_db()
    conn = db.get_connection()
    try:
        return conn.execute("SELECT * FROM final_risk_scores").df()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_shap_global() -> pd.DataFrame:
    db = get_db()
    conn = db.get_connection()
    try:
        return conn.execute("""
            SELECT friendly_name, avg(abs_shap_value) AS mean_abs_shap,
                   count(DISTINCT cik) AS companies_affected,
                   countif(direction='Risk ↑') AS risk_increasing
            FROM shap_values
            GROUP BY friendly_name
            ORDER BY mean_abs_shap DESC
            LIMIT 15
        """).df()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_shap_company(cik: str) -> pd.DataFrame:
    db = get_db()
    conn = db.get_connection()
    try:
        return conn.execute(
            """
            SELECT feature_name, friendly_name, shap_value,
                   feature_value, abs_shap_value, direction, rank_in_company
            FROM shap_values
            WHERE cik = ?
            ORDER BY rank_in_company
            LIMIT 12
            """,
            [cik],
        ).df()
    except Exception:
        return pd.DataFrame()


tab_audit, tab_shap, tab_bias, tab_card = st.tabs([
    "📋 Audit Trail",
    "🔍 SHAP Explainability",
    "⚖️ Bias Analysis",
    "📄 Model Card & KQL",
])


# ---------------------------------------------------------------------------
# TAB 1: Audit Trail
# ---------------------------------------------------------------------------
with tab_audit:
    st.subheader("Audit Trail")
    st.caption("Append-only log of every risk scoring event (SR 11-7 compliant)")

    if not GOVERNANCE_AVAILABLE:
        st.info(
            "📋 Audit Trail requires the governance module. "
            "This feature is available when running locally with full dependencies."
        )
    else:
        stats = load_audit_stats()
        if stats:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Events", int(stats.get("total_events", 0)))
            c2.metric("Companies Logged", int(stats.get("companies_logged", 0)))
            c3.metric("Scoring Events", int(stats.get("scoring_events", 0)))
            c4.metric("Human Reviews", int(stats.get("human_reviews", 0)))
            c5.metric("High/Critical Flags", int(stats.get("high_risk_flags", 0)))

        st.divider()

        days = st.slider("Days back", min_value=7, max_value=365, value=30, step=7)
        audit_df = load_audit_trail(days_back=days)

        if audit_df.empty:
            st.info("No audit trail events found. Run `ml-run` or `governance-report` first.")
        else:
            display_cols = [
                c for c in [
                    "generated_at", "event_type", "ticker", "company_name",
                    "final_risk_score", "final_risk_tier", "is_anomaly",
                    "human_reviewed", "review_action",
                ]
                if c in audit_df.columns
            ]
            st.dataframe(
                audit_df[display_cols].sort_values("generated_at", ascending=False),
                use_container_width=True,
                height=400,
            )
            st.caption(f"{len(audit_df):,} events in last {days} days")


# ---------------------------------------------------------------------------
# TAB 2: SHAP Explainability
# ---------------------------------------------------------------------------
with tab_shap:
    st.subheader("SHAP Feature Importance")
    st.caption("SHapley Additive exPlanations — why did each company receive its risk score?")

    col_global, col_company = st.columns([1, 1])

    with col_global:
        st.markdown("#### Global Feature Importance")
        shap_global = load_shap_global()
        if shap_global.empty:
            st.info("No SHAP values found. Run `shap-run` first.")
        else:
            import plotly.express as px
            fig = px.bar(
                shap_global.sort_values("mean_abs_shap"),
                x="mean_abs_shap",
                y="friendly_name",
                orientation="h",
                color="mean_abs_shap",
                color_continuous_scale="Blues",
                labels={
                    "mean_abs_shap": "Mean |SHAP Value|",
                    "friendly_name": "Feature",
                },
                title="Average Feature Impact Across All Companies",
            )
            fig.update_layout(
                height=450,
                showlegend=False,
                coloraxis_showscale=False,
                yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                font=dict(family="Inter, sans-serif", size=12, color="#B0BEC5"),
                xaxis=dict(gridcolor="rgba(41,121,255,0.1)", tickfont=dict(color="#B0BEC5")),
                yaxis=dict(gridcolor="rgba(41,121,255,0.1)", tickfont=dict(color="#B0BEC5")),
            )
            st.plotly_chart(fig, width="stretch")

    with col_company:
        st.markdown("#### Per-Company Waterfall")
        scores_df = load_final_scores()

        if scores_df.empty:
            st.info("No scored companies found. Run `ml-run` first.")
        else:
            company_options = {
                f"{row.get('ticker','?')} — {row.get('company_name','Unknown')}": str(row["cik"])
                for _, row in scores_df.iterrows()
                if "cik" in row
            }
            selected_label = st.selectbox(
                "Select company", options=list(company_options.keys())
            )
            if selected_label:
                cik = company_options[selected_label]
                shap_df = load_shap_company(cik)
                if shap_df.empty:
                    st.info("No SHAP values for this company. Run `shap-run` first.")
                else:
                    import plotly.graph_objects as go
                    colors = [
                        "#C62828" if v > 0 else "#2E7D32"
                        for v in shap_df["shap_value"]
                    ]
                    fig2 = go.Figure(
                        go.Bar(
                            x=shap_df["shap_value"].tolist()[::-1],
                            y=shap_df["friendly_name"].tolist()[::-1],
                            orientation="h",
                            marker_color=colors[::-1],
                            text=[f"{v:+.4f}" for v in shap_df["shap_value"].tolist()[::-1]],
                            textposition="outside",
                        )
                    )
                    fig2.add_vline(x=0, line_width=1, line_color="#90CAF9")
                    fig2.update_layout(
                        title=f"Risk Drivers: {selected_label.split(' — ')[0]}",
                        xaxis_title="SHAP Value",
                        height=400,
                        plot_bgcolor="rgba(15,23,42,0.6)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", size=12, color="#B0BEC5"),
                        xaxis=dict(gridcolor="rgba(41,121,255,0.1)", tickfont=dict(color="#B0BEC5")),
                        yaxis=dict(gridcolor="rgba(41,121,255,0.1)", tickfont=dict(color="#B0BEC5")),
                    )
                    st.plotly_chart(fig2, width="stretch")
                    st.caption("Red = increases anomaly risk | Green = reduces risk")


# ---------------------------------------------------------------------------
# TAB 3: Bias Analysis
# ---------------------------------------------------------------------------
with tab_bias:
    st.subheader("Bias Analysis")
    st.caption("EU AI Act Article 10 — systematic bias monitoring across industry groups")

    scores_df = load_final_scores()

    if scores_df.empty:
        st.info("No scored companies. Run the ML pipeline first.")
    elif not GOVERNANCE_AVAILABLE:
        st.info(
            "⚖️ Bias Analysis requires the governance module. "
            "This feature is available when running locally with full dependencies."
        )
    else:
        analyzer = BiasAnalyzer()
        reports = analyzer.run_full_bias_audit(scores_df)

        if not reports:
            st.warning("Could not run bias analysis — required columns may be missing.")
        else:
            summary_df = analyzer.generate_bias_summary_table(reports)
            st.markdown("#### Bias Audit Summary")
            severity_colors = {
                "NONE": "green",
                "LOW": "orange",
                "MEDIUM": "orange",
                "HIGH": "red",
            }
            for _, row in summary_df.iterrows():
                sev = row["Bias Severity"]
                color = severity_colors.get(sev, "gray")
                flagged = row["Groups Flagged"]
                st.markdown(
                    f"**{row['Dimension']}** — :{color}[{sev}]"
                    f" | {row['Groups Analyzed']} groups | {flagged} flagged"
                    + (f" ({row['Flagged Groups']})" if flagged > 0 else "")
                )

            st.divider()

            selected_dim = st.selectbox(
                "Drill into dimension",
                options=list(reports.keys()),
                format_func=lambda x: x.replace("_", " ").title(),
            )
            if selected_dim and selected_dim in reports:
                report = reports[selected_dim]
                st.info(report.interpretation)

                grp = report.group_stats.copy()
                if not grp.empty:
                    import plotly.express as px
                    color_map = {
                        "Over-scored": "#C62828",
                        "Normal": "#1565C0",
                        "Under-scored": "#2E7D32",
                    }
                    fig3 = px.bar(
                        grp.sort_values("mean_score", ascending=True),
                        x="mean_score",
                        y=selected_dim,
                        orientation="h",
                        color="bias_direction",
                        color_discrete_map=color_map,
                        error_x="std_score",
                        title=f"Mean Risk Score by {selected_dim.replace('_', ' ').title()}",
                        labels={
                            "mean_score": "Mean Risk Score",
                            selected_dim: "",
                            "bias_direction": "Bias Status",
                        },
                    )
                    fig3.add_vline(
                        x=report.overall_mean,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text=f"Overall mean ({report.overall_mean:.1f})",
                    )
                    fig3.update_layout(height=max(300, len(grp) * 30 + 100))
                    st.plotly_chart(fig3, width="stretch")

                    st.dataframe(
                        grp[[
                            selected_dim, "company_count", "mean_score",
                            "std_score", "z_score", "bias_direction", "high_risk_rate",
                        ]].rename(columns={
                            selected_dim: "Group",
                            "company_count": "Companies",
                            "mean_score": "Mean Score",
                            "std_score": "Std Dev",
                            "z_score": "Z-Score",
                            "bias_direction": "Status",
                            "high_risk_rate": "High-Risk Rate",
                        }),
                        use_container_width=True,
                    )


# ---------------------------------------------------------------------------
# TAB 4: Model Card & KQL
# ---------------------------------------------------------------------------
with tab_card:
    st.subheader("Model Card & KQL Query Library")

    col_mc, col_kql = st.columns([1, 1])

    with col_mc:
        st.markdown("#### Model Card")
        from pathlib import Path
        mc_path = Path("docs/model_card.md")
        if mc_path.exists():
            content = mc_path.read_text()
            with st.expander("View full model card", expanded=False):
                st.markdown(content)

            json_path = Path("docs/model_card.json")
            if json_path.exists():
                import json
                st.download_button(
                    label="Download model_card.json",
                    data=json_path.read_text(),
                    file_name="model_card.json",
                    mime="application/json",
                )
        else:
            st.info("Model card not generated yet. Run `python -m finsight360.cli model-card`")

        if GOVERNANCE_AVAILABLE and st.button("Regenerate Model Card"):
            db = get_db()
            with st.spinner("Generating..."):
                result = generate_model_card(db)
            st.success(f"Generated: {result['md_path']}")
            load_final_scores.clear()

    with col_kql:
        st.markdown("#### KQL Query Library")
        st.caption("Ready-to-paste into Azure Data Explorer, Azure Monitor, or Microsoft Sentinel")

        if not GOVERNANCE_AVAILABLE:
            st.info(
                "📊 KQL Query Library requires the governance module. "
                "This feature is available when running locally with full dependencies."
            )
        else:
            query_name = st.selectbox(
                "Select query",
                options=list(KQL_QUERIES.keys()),
                format_func=lambda k: k.replace("_", " ").title(),
            )
            if query_name:
                st.code(KQL_QUERIES[query_name].strip(), language="sql")

            with st.expander("Table Schemas (ADX)"):
                st.code(KQL_TABLE_SCHEMA.strip(), language="sql")

            kql_path = Path("docs/kql_queries.kql")
            if kql_path.exists():
                st.download_button(
                    label="Download kql_queries.kql",
                    data=kql_path.read_text(),
                    file_name="kql_queries.kql",
                    mime="text/plain",
                )
            else:
                if st.button("Export KQL File"):
                    from governance.kql_queries import export_kql_file
                    path = export_kql_file()
                    st.success(f"Exported to {path}")
