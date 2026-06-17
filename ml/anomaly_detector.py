"""
Isolation Forest anomaly detection for financial statement fraud signals.

Isolation Forest rationale: works well on high-dimensional financial data
without requiring labeled fraud examples (which are rare). It isolates
anomalies by randomly partitioning features — anomalous companies require
fewer splits to isolate, yielding a lower anomaly score.
"""
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from scipy.stats import rankdata
from utils.logger import get_logger

log = get_logger("anomaly_detector")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class AnomalyResult:
    cik: str
    ticker: str
    company_name: str
    industry_group: str
    isolation_forest_score: float    # raw IF score (more negative = more anomalous)
    anomaly_percentile: float        # percentile rank (100 = most anomalous)
    ml_risk_score: float             # final 0-100 combined score
    is_anomaly: bool                 # IF classification: True = anomalous
    pre_ml_risk_score: float         # rule-based score from dbt
    risk_tier: str
    feature_values: dict


class AnomalyDetector:
    """
    Isolation Forest pipeline with RobustScaler preprocessing.

    RobustScaler chosen over StandardScaler because financial data has heavy
    tails — RobustScaler uses median/IQR, making it resistant to the very
    anomalies we're trying to detect.
    """

    def __init__(self, contamination: float = 0.10, n_estimators: int = 200,
                 random_state: int = 42):
        """
        contamination=0.10: expect ~10% of companies to be anomalous.
        Conservative — actual fraud rate is lower but we prefer high recall
        so all candidates surface for human review.
        """
        self.contamination = contamination
        self.feature_cols: list = []

        self.pipeline = Pipeline([
            ("scaler", RobustScaler()),
            ("isolation_forest", IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=random_state,
                bootstrap=False,
                n_jobs=-1,
            ))
        ])

    def fit(self, feature_df: pd.DataFrame,
            feature_cols: list) -> "AnomalyDetector":
        """Train Isolation Forest on feature matrix."""
        self.feature_cols = feature_cols
        X = feature_df[feature_cols].values

        log.info("isolation_forest_training",
                 n_samples=len(X),
                 n_features=len(feature_cols),
                 contamination=self.contamination)

        self.pipeline.fit(X)
        log.info("isolation_forest_trained")
        return self

    def predict(self, feature_df: pd.DataFrame) -> list:
        """Run anomaly detection on feature matrix."""
        X = feature_df[self.feature_cols].values

        # IF returns: -1 = anomaly, +1 = normal
        predictions = self.pipeline.predict(X)
        # Raw decision scores: more negative = more anomalous
        scores = self.pipeline.decision_function(X)

        # Convert to 0-100 percentile (100 = most anomalous)
        percentiles = (1 - rankdata(scores) / len(scores)) * 100

        results = []
        for i, row in feature_df.reset_index(drop=True).iterrows():
            # Combined score: 60% ML anomaly percentile + 40% rule-based
            ml_component = percentiles[i] * 0.60
            rule_component = float(row.get("pre_ml_risk_score", 0)) * 0.40
            combined = round(min(100, ml_component + rule_component), 1)

            results.append(AnomalyResult(
                cik=str(row["cik"]),
                ticker=str(row.get("ticker", "")),
                company_name=str(row["company_name"]),
                industry_group=str(row.get("industry_group", "")),
                isolation_forest_score=round(float(scores[i]), 4),
                anomaly_percentile=round(float(percentiles[i]), 1),
                ml_risk_score=combined,
                is_anomaly=bool(predictions[i] == -1),
                pre_ml_risk_score=float(row.get("pre_ml_risk_score", 0)),
                risk_tier=str(row.get("risk_tier", "UNKNOWN")),
                feature_values={
                    col: float(row[col])
                    for col in self.feature_cols
                    if col in row.index
                }
            ))

        log.info("anomaly_detection_complete",
                 total=len(results),
                 anomalies=sum(1 for r in results if r.is_anomaly))
        return results

    def save(self, path: str = "models/isolation_forest.joblib"):
        joblib.dump({"pipeline": self.pipeline,
                     "feature_cols": self.feature_cols}, path)
        log.info("model_saved", path=path)

    @classmethod
    def load(cls, path: str = "models/isolation_forest.joblib") -> "AnomalyDetector":
        data = joblib.load(path)
        detector = cls()
        detector.pipeline = data["pipeline"]
        detector.feature_cols = data["feature_cols"]
        return detector
