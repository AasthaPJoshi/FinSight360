"""MLflow experiment tracking and model registry for FinSight360."""
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from utils.logger import get_logger

log = get_logger("model_registry")

EXPERIMENT_NAME = "finsight360-anomaly-detection"
TRACKING_URI = "mlruns"


class ModelRegistry:
    def __init__(self):
        Path(TRACKING_URI).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(TRACKING_URI)
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
        except Exception as exc:
            log.warning("mlflow_experiment_init_failed", error=str(exc))

    def log_training_run(self, detector, feature_df: pd.DataFrame,
                         results: list, benford_df: pd.DataFrame,
                         params: dict) -> str:
        """Log a complete training run to MLflow."""
        with mlflow.start_run(run_name="isolation_forest_v1") as run:
            mlflow.log_params({
                "contamination": params.get("contamination", 0.10),
                "n_estimators": params.get("n_estimators", 200),
                "n_companies": len(feature_df),
                "n_features": len(detector.feature_cols),
                "feature_list": ",".join(detector.feature_cols),
            })

            anomaly_count = sum(1 for r in results if r.is_anomaly)
            high_risk = sum(1 for r in results if r.ml_risk_score > 70)
            avg_risk = sum(r.ml_risk_score for r in results) / len(results) if results else 0

            benford_nc = (
                len(benford_df[benford_df["conformity_rating"] == "Non-conforming"])
                if not benford_df.empty else 0
            )

            mlflow.log_metrics({
                "total_companies": len(results),
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_count / len(results) if results else 0,
                "high_risk_count": high_risk,
                "avg_risk_score": avg_risk,
                "benford_non_conforming": benford_nc,
            })

            # Log model artifact
            try:
                mlflow.sklearn.log_model(
                    detector.pipeline,
                    "isolation_forest",
                    registered_model_name="finsight360-anomaly-detector",
                )
            except Exception:
                mlflow.sklearn.log_model(detector.pipeline, "isolation_forest")

            # Log feature summary
            feature_summary = pd.DataFrame([{
                "feature": col,
                "mean_value": feature_df[col].mean() if col in feature_df else 0,
                "std_value": feature_df[col].std() if col in feature_df else 0,
            } for col in detector.feature_cols])

            summary_path = Path(TRACKING_URI) / "feature_summary.csv"
            feature_summary.to_csv(summary_path, index=False)
            mlflow.log_artifact(str(summary_path))

            run_id = run.info.run_id
            log.info("mlflow_run_logged", run_id=run_id,
                     anomaly_count=anomaly_count)
            return run_id

    def get_latest_run(self) -> dict:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            return {}
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return {}
        return {
            "run_id": runs[0].info.run_id,
            "metrics": runs[0].data.metrics,
            "params": runs[0].data.params,
        }
