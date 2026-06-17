"""Reusable KPI metric card row."""
import streamlit as st


def render_metric_row(metrics: list[dict]) -> None:
    """
    Render a row of KPI cards.
    Each dict: {label, value, delta, delta_color, help}
    delta_color: "normal" | "inverse" | "off"
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=m.get("label", ""),
                value=m.get("value", "—"),
                delta=m.get("delta"),
                delta_color=m.get("delta_color", "normal"),
                help=m.get("help", ""),
            )


def render_risk_badge(risk_tier: str) -> str:
    """Return colored emoji badge for risk tier."""
    badges = {
        "CRITICAL": "🔴 CRITICAL",
        "HIGH": "🟠 HIGH",
        "MEDIUM": "🟡 MEDIUM",
        "LOW": "🟢 LOW",
        "CLEAN": "✅ CLEAN",
    }
    return badges.get(str(risk_tier).upper(), "⚪ UNKNOWN")
