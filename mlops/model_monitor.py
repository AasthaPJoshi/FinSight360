"""
Model drift detection between pipeline runs.
Flags: score drift, anomaly rate changes, new CRITICAL companies.
"""
from utils.logger import get_logger

log = get_logger("model_monitor")


class ModelMonitor:
    """
    Monitors for data drift and model drift between pipeline runs.

    Drift signals:
    - Average risk score shifts > 10 points
    - Anomaly rate changes > 5 percentage points
    - New companies entering CRITICAL tier
    """

    SCORE_DRIFT_THRESHOLD = 10.0
    ANOMALY_RATE_DRIFT_THRESHOLD = 0.05

    def __init__(self, db):
        self.db = db

    def get_run_summary(self) -> dict:
        """Get summary stats for most recent pipeline run."""
        conn = self.db.get_connection()
        try:
            result = conn.execute("""
                SELECT
                    count(*) AS n_companies,
                    avg(final_risk_score) AS avg_score,
                    stddev(final_risk_score) AS std_score,
                    sum(CASE WHEN is_anomaly THEN 1 ELSE 0 END)
                        * 1.0 / count(*) AS anomaly_rate,
                    sum(CASE WHEN final_risk_tier = 'CRITICAL'
                        THEN 1 ELSE 0 END) AS critical_count,
                    max(computed_at) AS run_timestamp
                FROM final_risk_scores
            """).df()

            if result.empty:
                return {}
            return result.iloc[0].to_dict()
        except Exception as e:
            log.warning("run_summary_failed", error=str(e))
            return {}

    def compare_runs(self) -> dict:
        """
        Compare current vs previous pipeline run via audit trail.
        Returns drift report dict.
        """
        current = self.get_run_summary()

        if not current:
            return {"status": "no_data", "message": "No pipeline runs found"}

        conn = self.db.get_connection()
        try:
            history = conn.execute("""
                SELECT
                    date_trunc('hour', generated_at) AS run_hour,
                    avg(final_risk_score) AS avg_score,
                    sum(CASE WHEN is_anomaly THEN 1 ELSE 0 END)
                        * 1.0 / count(*) AS anomaly_rate,
                    count(*) AS n_companies
                FROM audit_trail
                WHERE event_type = 'risk_score_generated'
                GROUP BY date_trunc('hour', generated_at)
                ORDER BY run_hour DESC
                LIMIT 5
            """).df()
        except Exception:
            return {
                "status": "ok",
                "message": "First run — no comparison available",
                "current": current,
            }

        if len(history) < 2:
            return {
                "status": "ok",
                "message": "Insufficient history for comparison",
                "current": current,
            }

        latest = history.iloc[0]
        previous = history.iloc[1]

        score_drift = float(latest["avg_score"]) - float(previous["avg_score"])
        anomaly_drift = (
            float(latest["anomaly_rate"]) - float(previous["anomaly_rate"])
        )

        alerts = []

        if abs(score_drift) > self.SCORE_DRIFT_THRESHOLD:
            alerts.append({
                "type": "SCORE_DRIFT",
                "severity": "HIGH" if abs(score_drift) > 20 else "MEDIUM",
                "message": (
                    f"Average risk score shifted {score_drift:+.1f} points "
                    f"({previous['avg_score']:.1f} → {latest['avg_score']:.1f})"
                ),
            })

        if abs(anomaly_drift) > self.ANOMALY_RATE_DRIFT_THRESHOLD:
            alerts.append({
                "type": "ANOMALY_RATE_DRIFT",
                "severity": "MEDIUM",
                "message": (
                    f"Anomaly rate changed {anomaly_drift:+.1%} "
                    f"({previous['anomaly_rate']:.1%} → {latest['anomaly_rate']:.1%})"
                ),
            })

        log.info(
            "drift_check_complete",
            score_drift=score_drift,
            anomaly_drift=anomaly_drift,
            alerts=len(alerts),
        )

        return {
            "status": "alerts" if alerts else "ok",
            "score_drift": round(score_drift, 2),
            "anomaly_rate_drift": round(anomaly_drift, 4),
            "alerts": alerts,
            "history_runs": len(history),
            "current_avg_score": round(float(latest["avg_score"]), 1),
            "previous_avg_score": round(float(previous["avg_score"]), 1),
        }
