"""Unit tests for Phase 3: Benford's Law, Isolation Forest, feature builder."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from ml.benford import BenfordAnalyzer, BENFORD_DISTRIBUTION
from ml.anomaly_detector import AnomalyDetector, AnomalyResult
from ml.feature_builder import FeatureBuilder, ISOLATION_FOREST_FEATURES


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_feature_df(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Synthetic feature DataFrame for Isolation Forest tests."""
    rng = np.random.default_rng(seed)
    data = {col: rng.standard_normal(n) for col in ISOLATION_FOREST_FEATURES}
    df = pd.DataFrame(data)
    # Flag columns should be 0/1
    flag_cols = [c for c in ISOLATION_FOREST_FEATURES if c.startswith("flag_")]
    df[flag_cols] = (df[flag_cols] > 0).astype(float)
    df["total_signal_count"] = df[flag_cols].sum(axis=1)
    df["late_filings_count"] = (df["late_filings_count"].abs() * 3).astype(int)
    df["max_reporting_lag_days"] = (df["max_reporting_lag_days"].abs() * 60).astype(int)

    df["cik"] = [f"CIK{i:06d}" for i in range(n)]
    df["ticker"] = [f"T{i}" for i in range(n)]
    df["company_name"] = [f"Company {i}" for i in range(n)]
    df["industry_group"] = ["Technology"] * n
    df["risk_tier"] = ["CLEAN"] * n
    df["pre_ml_risk_score"] = rng.uniform(0, 60, n)
    return df


# ──────────────────────────────────────────────
# Benford tests
# ──────────────────────────────────────────────

def test_benford_known_distribution():
    """Log-uniform data should conform to Benford's Law (MAD < 0.012)."""
    analyzer = BenfordAnalyzer()
    rng = np.random.default_rng(0)
    # Log-uniform on [1, 10^6] naturally follows Benford's Law
    values = pd.Series(10 ** rng.uniform(0, 6, 1000))
    result = analyzer.analyze_series(values, "TEST", "Test Co", "TST")

    assert result.n_observations == 1000
    assert result.mad_score < BenfordAnalyzer.MAD_MARGINAL, (
        f"Expected MAD < {BenfordAnalyzer.MAD_MARGINAL}, got {result.mad_score}"
    )
    assert result.conformity_rating in ("Acceptable", "Marginal")
    assert result.benford_risk_score < 60


def test_benford_uniform_distribution():
    """Uniformly distributed data should deviate significantly from Benford's Law."""
    analyzer = BenfordAnalyzer()
    rng = np.random.default_rng(7)
    # Integers 10-99: leading digits are roughly uniform, not Benford
    values = pd.Series(rng.integers(10, 100, 500).astype(float))
    result = analyzer.analyze_series(values, "TEST2", "Fake Co", "FAK")

    assert result.conformity_rating != "Acceptable", (
        f"Uniform data should not be Acceptable, got {result.conformity_rating}"
    )


def test_benford_insufficient_data():
    """Fewer than 50 observations should return 'Insufficient Data' rating."""
    analyzer = BenfordAnalyzer()
    values = pd.Series([100.0, 200.0, 300.0])
    result = analyzer.analyze_series(values, "X", "Tiny Co", "TIN")
    assert result.conformity_rating == "Insufficient Data"
    assert result.benford_risk_score == 0.0


# ──────────────────────────────────────────────
# Isolation Forest tests
# ──────────────────────────────────────────────

def test_isolation_forest_trains():
    """Isolation Forest should fit without error on synthetic feature matrix."""
    feature_df = _make_feature_df(n=30)
    detector = AnomalyDetector(contamination=0.10, n_estimators=10)
    detector.fit(feature_df, ISOLATION_FOREST_FEATURES)

    assert detector.feature_cols == ISOLATION_FOREST_FEATURES
    assert detector.pipeline is not None
    # Check scaler is fitted
    scaler = detector.pipeline.named_steps["scaler"]
    assert hasattr(scaler, "center_")


def test_isolation_forest_predicts():
    """After fit, predict should return AnomalyResult list of correct length."""
    feature_df = _make_feature_df(n=30)
    detector = AnomalyDetector(contamination=0.10, n_estimators=10)
    detector.fit(feature_df, ISOLATION_FOREST_FEATURES)
    results = detector.predict(feature_df)

    assert len(results) == 30
    assert all(isinstance(r, AnomalyResult) for r in results)
    assert all(0.0 <= r.ml_risk_score <= 100.0 for r in results)
    assert all(0.0 <= r.anomaly_percentile <= 100.0 for r in results)
    # contamination=0.10 with 30 samples → ~3 anomalies
    anomaly_count = sum(1 for r in results if r.is_anomaly)
    assert 1 <= anomaly_count <= 10  # plausible range


# ──────────────────────────────────────────────
# Feature builder tests
# ──────────────────────────────────────────────

def test_feature_builder_handles_nulls():
    """Null-filling logic should produce a clean feature matrix with no NaNs."""
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.get_connection.return_value = mock_conn

    # Build a DataFrame that has nulls in every type of column
    n = 10
    rng = np.random.default_rng(99)
    raw = {col: rng.standard_normal(n).tolist() for col in ISOLATION_FOREST_FEATURES}
    raw["cik"] = [f"C{i}" for i in range(n)]
    raw["ticker"] = [f"T{i}" for i in range(n)]
    raw["company_name"] = [f"Co {i}" for i in range(n)]
    raw["industry_group"] = ["Tech"] * n
    raw["risk_tier"] = ["CLEAN"] * n
    raw["pre_ml_risk_score"] = rng.uniform(0, 50, n).tolist()
    df_with_nulls = pd.DataFrame(raw)

    # Introduce NaNs across both flag and ratio columns
    for col in ISOLATION_FOREST_FEATURES:
        df_with_nulls.loc[0, col] = None
    df_with_nulls.loc[1, "gross_margin"] = None
    df_with_nulls.loc[2, "flag_ar_acceleration"] = None

    mock_conn.execute.return_value.df.return_value = df_with_nulls

    fb = FeatureBuilder(mock_db)
    result = fb.build_feature_matrix()

    available = [f for f in ISOLATION_FOREST_FEATURES if f in result.columns]
    null_count = result[available].isna().sum().sum()
    assert null_count == 0, f"Expected no nulls, found {null_count}"


# ──────────────────────────────────────────────
# Pipeline smoke test
# ──────────────────────────────────────────────

def test_ml_pipeline_smoke():
    """Pipeline should return status='failed' gracefully when no data is available."""
    from ml.pipeline import MLPipeline

    mock_db = MagicMock()
    mock_db.get_connection.return_value = MagicMock()

    with patch("ml.pipeline.FeatureBuilder") as MockFB, \
         patch("ml.pipeline.ModelRegistry"), \
         patch("ml.pipeline.BenfordAnalyzer"), \
         patch("ml.pipeline.AnomalyDetector"):

        MockFB.return_value.build_feature_matrix.return_value = pd.DataFrame()

        pipeline = MLPipeline(mock_db)
        result = pipeline.run()

    assert result["status"] == "failed"
    assert result["reason"] == "no_data"
