"""Reusable Plotly chart functions."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

RISK_COLORS = {
    "CRITICAL": "#C62828",
    "HIGH": "#E65100",
    "MEDIUM": "#F9A825",
    "LOW": "#2E7D32",
    "CLEAN": "#1565C0",
}


def risk_score_gauge(score: float, title: str = "Final Risk Score") -> go.Figure:
    """Gauge chart for a risk score 0-100."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 16}},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#1565C0"},
                "steps": [
                    {"range": [0, 30], "color": "#E8F5E9"},
                    {"range": [30, 50], "color": "#FFF9C4"},
                    {"range": [50, 70], "color": "#FFE0B2"},
                    {"range": [70, 100], "color": "#FFCDD2"},
                ],
                "threshold": {
                    "line": {"color": "#C62828", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
    return fig


def signal_bar_chart(row: pd.Series) -> go.Figure:
    """Horizontal bar chart showing signal contribution to final score."""
    signals = {
        "ML Anomaly (50%)": float(row.get("ml_risk_score", 0)) * 0.5,
        "Benford's Law (30%)": float(row.get("benford_risk_score", 0)) * 0.3,
        "Network Risk (20%)": float(row.get("network_risk", 0)) * 0.2,
    }

    fig = go.Figure(
        go.Bar(
            y=list(signals.keys()),
            x=list(signals.values()),
            orientation="h",
            marker_color=["#1565C0", "#C62828", "#F57C00"],
            text=[f"{v:.1f}" for v in signals.values()],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Risk Score Composition",
        xaxis_title="Contribution to Final Score",
        height=220,
        margin=dict(t=40, b=20, l=20, r=40),
        showlegend=False,
        xaxis=dict(range=[0, 60]),
    )
    return fig


def financial_trend_chart(
    df: pd.DataFrame, cik: str, metric: str, title: str
) -> go.Figure:
    """Line chart of a financial metric over time for one company."""
    company_df = df[df["cik"] == cik].sort_values("period_end")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=company_df["period_end"],
            y=company_df[metric],
            mode="lines+markers",
            name=title,
            line=dict(color="#1565C0", width=2),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title=metric.replace("_", " ").title(),
        height=280,
        margin=dict(t=40, b=20, l=20, r=20),
        hovermode="x unified",
    )
    return fig


def industry_risk_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: industry vs risk tier company count."""
    if df.empty or "industry_group" not in df.columns:
        return go.Figure()

    pivot = pd.crosstab(df["industry_group"], df["final_risk_tier"])
    tier_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "CLEAN"]
    pivot = pivot.reindex(columns=[c for c in tier_order if c in pivot.columns])

    fig = px.imshow(
        pivot,
        color_continuous_scale=["#E8F5E9", "#FFCDD2"],
        aspect="auto",
        title="Companies by Industry × Risk Tier",
    )
    fig.update_layout(
        height=420,
        margin=dict(t=50, b=20, l=120, r=20),
        coloraxis_colorbar=dict(title="Count"),
    )
    return fig


def benford_deviation_chart(
    observed: dict, expected: dict, company_name: str
) -> go.Figure:
    """Bar chart comparing observed vs expected Benford digit frequencies."""
    digits = list(range(1, 10))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=digits,
            y=[observed.get(d, 0) for d in digits],
            name="Observed",
            marker_color="#1565C0",
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=digits,
            y=[expected.get(d, 0) for d in digits],
            mode="lines+markers",
            name="Expected (Benford)",
            line=dict(color="#C62828", width=2, dash="dash"),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        title=f"Benford's Law Analysis: {company_name}",
        xaxis_title="Leading Digit",
        yaxis_title="Frequency",
        xaxis=dict(tickmode="array", tickvals=digits),
        height=320,
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(x=0.7, y=0.95),
        barmode="overlay",
    )
    return fig


def scatter_risk_landscape(df: pd.DataFrame) -> go.Figure:
    """Scatter: ML score vs Benford score, colored by risk tier."""
    if df.empty:
        return go.Figure()

    plot_df = df.dropna(subset=["ml_risk_score", "benford_risk_score"])
    if plot_df.empty:
        return go.Figure()

    fig = px.scatter(
        plot_df,
        x="ml_risk_score",
        y="benford_risk_score",
        color="final_risk_tier",
        color_discrete_map=RISK_COLORS,
        hover_data=["ticker", "company_name", "final_risk_score"],
        size="final_risk_score",
        size_max=20,
        title="Risk Landscape: ML Anomaly vs Benford Deviation",
        labels={
            "ml_risk_score": "ML Anomaly Score",
            "benford_risk_score": "Benford Risk Score",
        },
    )
    fig.update_layout(height=480, margin=dict(t=50, b=30, l=30, r=20))
    return fig
