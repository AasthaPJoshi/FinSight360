"""Page 4: Corporate relationship network graph (pyvis)."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
from pathlib import Path

import streamlit as st

from dashboard.data_loader import load_final_risk_scores
from dashboard.styles.theme import inject_css, page_header

st.set_page_config(
    page_title="Network Graph", page_icon="🕸️", layout="wide"
)

inject_css()
page_header(
    "🕸️",
    "Corporate Network",
    "Company relationships: auditors, subsidiaries, partnerships",
)

df = load_final_risk_scores()

if df.empty:
    st.warning("No data available.")
    st.stop()

RISK_NODE_COLORS = {
    "CRITICAL": "#C62828",
    "HIGH": "#E65100",
    "MEDIUM": "#F9A825",
    "LOW": "#2E7D32",
    "MINIMAL": "#1565C0",
    "CLEAN": "#1565C0",
    "UNKNOWN": "#9E9E9E",
}

try:
    from pyvis.network import Network

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="#1A1A2E",
    )
    net.set_options("""
    {
      "physics": {"stabilization": {"iterations": 100}},
      "nodes": {"font": {"size": 12}},
      "edges": {"smooth": {"type": "continuous"}}
    }
    """)

    for _, row in df.iterrows():
        tier = str(row.get("final_risk_tier", "UNKNOWN"))
        score = float(row.get("final_risk_score", 0))
        ticker = str(row.get("ticker", row.get("cik", "")))
        name = str(row.get("company_name", ticker))
        cik = str(row.get("cik", ""))

        net.add_node(
            cik,
            label=ticker,
            title=(
                f"{name}\nRisk: {score:.1f}/100\n"
                f"Tier: {tier}\n"
                f"Signals: {row.get('total_signal_count', 0)}"
            ),
            color=RISK_NODE_COLORS.get(tier, "#9E9E9E"),
            size=max(10, min(40, score / 3)),
        )

    rel_path = Path("data/reference/sp500_relationships.json")
    if rel_path.exists():
        with open(rel_path) as f:
            relationships = json.load(f)

        edge_colors = {
            "SUBSIDIARY": "#5C6BC0",
            "PARTNERS_WITH": "#26A69A",
            "CUSTOMER_OF": "#8D6E63",
            "JOINT_VENTURE_WITH": "#AB47BC",
        }

        existing_ids = {n["id"] for n in net.nodes}
        for rel in relationships:
            parent = str(rel.get("parent_cik", ""))
            child = str(rel.get("subsidiary_cik", ""))
            rel_type = str(rel.get("relationship_type", "RELATED"))

            if parent not in existing_ids:
                continue
            if child not in existing_ids:
                net.add_node(
                    child,
                    label=str(rel.get("subsidiary_name", child))[:15],
                    color="#BDBDBD",
                    size=8,
                )
                existing_ids.add(child)

            net.add_edge(
                parent,
                child,
                title=rel_type,
                color=edge_colors.get(rel_type, "#9E9E9E"),
                width=1.5,
            )

    Path("data").mkdir(exist_ok=True)
    html_path = "data/network_graph.html"
    net.save_graph(html_path)

    with open(html_path) as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=620, scrolling=False)

except ImportError:
    st.error("pyvis not installed. Run: `pip install pyvis`")

    import numpy as np
    import plotly.graph_objects as go

    n = len(df)
    if n > 0:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)

        fig = go.Figure()
        for i, (_, row) in enumerate(df.iterrows()):
            color = RISK_NODE_COLORS.get(
                str(row.get("final_risk_tier", "")), "#9E9E9E"
            )
            fig.add_trace(
                go.Scatter(
                    x=[x[i]],
                    y=[y[i]],
                    mode="markers+text",
                    text=[str(row.get("ticker", ""))],
                    textposition="top center",
                    marker=dict(
                        size=max(8, float(row.get("final_risk_score", 0)) / 5),
                        color=color,
                    ),
                    showlegend=False,
                )
            )
        fig.update_layout(
            height=500,
            title="Company Risk Network (simplified)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(family="Inter, sans-serif", color="#B0BEC5"),
        )
        st.plotly_chart(fig, width="stretch")

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🔴 **CRITICAL** risk")
    st.markdown("🟠 **HIGH** risk")
with col2:
    st.markdown("🟡 **MEDIUM** risk")
    st.markdown("🟢 **LOW** risk")
with col3:
    st.markdown("🔵 **CLEAN** / monitored")
    st.markdown("⚫ Subsidiary / partner")
