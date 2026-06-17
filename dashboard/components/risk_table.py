"""Styled risk ranking table component."""
import pandas as pd
import streamlit as st

TIER_COLORS = {
    "CRITICAL": "background-color: #FFCDD2; color: #B71C1C; font-weight: bold",
    "HIGH": "background-color: #FFE0B2; color: #E65100; font-weight: bold",
    "MEDIUM": "background-color: #FFF9C4; color: #F57F17",
    "LOW": "background-color: #E8F5E9; color: #1B5E20",
    "CLEAN": "background-color: #E3F2FD; color: #0D47A1",
}


def style_risk_tier(val: str) -> str:
    return TIER_COLORS.get(str(val).upper(), "")


def render_risk_table(
    df: pd.DataFrame,
    n_rows: int = 20,
    show_cols: list | None = None,
) -> None:
    """Render styled risk ranking table."""
    if df.empty:
        st.info("No risk data available. Run the pipeline first.")
        return

    default_cols = [
        "ticker",
        "company_name",
        "industry_group",
        "final_risk_score",
        "final_risk_tier",
        "ml_risk_score",
        "benford_risk_score",
        "network_risk",
        "total_signal_count",
        "is_anomaly",
    ]

    cols = show_cols or [c for c in default_cols if c in df.columns]
    display_df = df[cols].head(n_rows).copy()

    for col in ["final_risk_score", "ml_risk_score", "benford_risk_score", "network_risk"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(1)

    styled = display_df.style
    if "final_risk_tier" in display_df.columns:
        styled = styled.map(style_risk_tier, subset=["final_risk_tier"])

    st.dataframe(
        styled,
        use_container_width=True,
        height=min(600, 45 * n_rows + 38),
    )
