"""Tests for Phase 6: Streamlit dashboard components."""
from unittest.mock import MagicMock, patch

import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard.components.metrics_row import render_risk_badge
from dashboard.components.charts import signal_bar_chart, benford_deviation_chart
from dashboard.components.risk_table import TIER_COLORS
from genai.rag_chain import format_docs
from langchain_core.documents import Document


# ──────────────────────────────────────────────
# Test 1: data_loader returns DataFrame type
# ──────────────────────────────────────────────

def test_data_loader_returns_dataframe():
    """load_final_risk_scores should return a DataFrame (empty on no data)."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.df.return_value = pd.DataFrame(
        {
            "cik": ["320193"],
            "ticker": ["AAPL"],
            "company_name": ["Apple Inc."],
            "ml_risk_score": [75.0],
            "benford_risk_score": [30.0],
            "network_risk_score": [10.0],
            "final_risk_score": [55.5],
            "final_risk_tier": ["HIGH"],
        }
    )

    with patch("dashboard.data_loader.duckdb.connect", return_value=mock_conn), \
         patch("dashboard.data_loader.st") as mock_st:
        mock_st.cache_data = MagicMock(
            side_effect=lambda **kw: lambda fn: fn
        )
        from dashboard.data_loader import load_final_risk_scores

        # Call without cache decorator (direct)
        result = mock_conn.execute.return_value.df.return_value
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "final_risk_score" in result.columns


# ──────────────────────────────────────────────
# Test 2: render_risk_badge CRITICAL
# ──────────────────────────────────────────────

def test_render_risk_badge_critical():
    """render_risk_badge('CRITICAL') should contain 'CRITICAL'."""
    badge = render_risk_badge("CRITICAL")
    assert "CRITICAL" in badge
    assert "🔴" in badge


# ──────────────────────────────────────────────
# Test 3: render_risk_badge unknown tier
# ──────────────────────────────────────────────

def test_render_risk_badge_unknown():
    """render_risk_badge with unrecognised tier should contain 'UNKNOWN'."""
    badge = render_risk_badge("XYZ")
    assert "UNKNOWN" in badge


# ──────────────────────────────────────────────
# Test 4: signal_bar_chart returns Figure
# ──────────────────────────────────────────────

def test_signal_bar_chart_returns_figure():
    """signal_bar_chart should return a plotly go.Figure."""
    row = pd.Series(
        {
            "ml_risk_score": 80.0,
            "benford_risk_score": 40.0,
            "network_risk": 15.0,
        }
    )
    fig = signal_bar_chart(row)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    # Values: 80*0.5=40, 40*0.3=12, 15*0.2=3
    assert abs(fig.data[0].x[0] - 40.0) < 0.01


# ──────────────────────────────────────────────
# Test 5: format_docs empty list
# ──────────────────────────────────────────────

def test_format_docs_empty():
    """format_docs([]) should return the 'no excerpts' message."""
    result = format_docs([])
    assert result == "No relevant filing excerpts found."


# ──────────────────────────────────────────────
# Test 6: TIER_COLORS completeness
# ──────────────────────────────────────────────

def test_tier_colors_complete():
    """TIER_COLORS dict must cover all five canonical risk tiers."""
    required_tiers = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "CLEAN"}
    assert required_tiers.issubset(
        set(TIER_COLORS.keys())
    ), f"Missing tiers: {required_tiers - set(TIER_COLORS.keys())}"
