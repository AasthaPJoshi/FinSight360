"""
Audit trail system for FinSight360.
Every risk score generated must be logged with:
- What was scored (company, period)
- Which model version was used
- What score was produced
- What features drove it
- When it was generated
- Whether human review occurred

Compliance backbone for SR 11-7, EU AI Act, and internal audit.
"""
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import pandas as pd

from storage.database import Database
from utils.logger import get_logger

log = get_logger("audit_trail")


@dataclass
class AuditEntry:
    """Immutable audit record for one risk scoring event."""
    audit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Literal[
        "risk_score_generated",
        "anomaly_flagged",
        "brief_generated",
        "model_trained",
        "data_ingested",
        "human_review",
        "alert_escalated",
    ] = "risk_score_generated"

    # What was scored
    cik: str = ""
    ticker: str = ""
    company_name: str = ""
    period_of_report: str = ""

    # Which model
    model_name: str = "isolation_forest_v1"
    model_version: str = "1.0.0"
    mlflow_run_id: str = ""

    # Scores produced
    ml_risk_score: float = 0.0
    benford_risk_score: float = 0.0
    network_risk: float = 0.0
    final_risk_score: float = 0.0
    final_risk_tier: str = ""
    is_anomaly: bool = False

    # Feature context
    top_features: str = ""  # JSON string of top SHAP features
    total_signals: int = 0

    # Metadata
    generated_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    pipeline_version: str = "1.0.0"
    data_hash: str = ""
    human_reviewed: bool = False
    human_reviewer: str = ""
    review_action: str = ""
    review_notes: str = ""

    def compute_data_hash(self, feature_values: dict) -> str:
        serialized = json.dumps(feature_values, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class AuditTrailManager:
    """
    Manages the append-only audit trail table in DuckDB.

    Design: append-only, immutable records, indexed by cik/time/tier.
    """

    def __init__(self, db: Database):
        self.db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        conn = self.db.get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_trail (
                audit_id      VARCHAR PRIMARY KEY,
                event_type    VARCHAR NOT NULL,
                cik           VARCHAR,
                ticker        VARCHAR,
                company_name  VARCHAR,
                period_of_report VARCHAR,
                model_name    VARCHAR,
                model_version VARCHAR,
                mlflow_run_id VARCHAR,
                ml_risk_score    DOUBLE,
                benford_risk_score DOUBLE,
                network_risk     DOUBLE,
                final_risk_score DOUBLE,
                final_risk_tier  VARCHAR,
                is_anomaly       BOOLEAN,
                top_features     TEXT,
                total_signals    INTEGER,
                generated_at     TIMESTAMP NOT NULL,
                pipeline_version VARCHAR,
                data_hash        VARCHAR,
                human_reviewed   BOOLEAN DEFAULT false,
                human_reviewer   VARCHAR,
                review_action    VARCHAR,
                review_notes     TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_cik ON audit_trail(cik)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_generated ON audit_trail(generated_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_tier ON audit_trail(final_risk_tier)"
        )

    def log_scoring_event(self, entry: AuditEntry) -> str:
        """Log a risk scoring event (upsert by audit_id). Returns audit_id."""
        conn = self.db.get_connection()
        conn.execute(
            """
            INSERT INTO audit_trail VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            ON CONFLICT (audit_id) DO NOTHING
            """,
            [
                entry.audit_id, entry.event_type,
                entry.cik, entry.ticker, entry.company_name,
                entry.period_of_report, entry.model_name,
                entry.model_version, entry.mlflow_run_id,
                entry.ml_risk_score, entry.benford_risk_score,
                entry.network_risk, entry.final_risk_score,
                entry.final_risk_tier, entry.is_anomaly,
                entry.top_features, entry.total_signals,
                entry.generated_at, entry.pipeline_version,
                entry.data_hash, entry.human_reviewed,
                entry.human_reviewer, entry.review_action,
                entry.review_notes,
            ],
        )
        log.info(
            "audit_logged",
            audit_id=entry.audit_id[:8],
            cik=entry.cik,
            ev=entry.event_type,
            score=entry.final_risk_score,
        )
        return entry.audit_id

    def log_all_scores(
        self, final_scores_df: pd.DataFrame, mlflow_run_id: str = ""
    ) -> int:
        """Bulk-log all companies after a pipeline run."""
        count = 0
        for _, row in final_scores_df.iterrows():
            entry = AuditEntry(
                event_type="risk_score_generated",
                cik=str(row.get("cik", "")),
                ticker=str(row.get("ticker", "")),
                company_name=str(row.get("company_name", "")),
                model_name="finsight360_v1",
                model_version="1.0.0",
                mlflow_run_id=mlflow_run_id,
                ml_risk_score=float(row.get("ml_risk_score", 0)),
                benford_risk_score=float(row.get("benford_risk_score", 0)),
                network_risk=float(row.get("network_risk_score", row.get("network_risk", 0))),
                final_risk_score=float(row.get("final_risk_score", 0)),
                final_risk_tier=str(row.get("final_risk_tier", "")),
                is_anomaly=bool(row.get("is_anomaly", False)),
                total_signals=int(row.get("total_signal_count", 0)),
            )
            self.log_scoring_event(entry)
            count += 1
        log.info("bulk_audit_logged", count=count)
        return count

    def log_human_review(
        self, cik: str, reviewer: str, action: str, notes: str
    ) -> str:
        entry = AuditEntry(
            event_type="human_review",
            cik=cik,
            human_reviewed=True,
            human_reviewer=reviewer,
            review_action=action,
            review_notes=notes,
        )
        return self.log_scoring_event(entry)

    def get_audit_log(
        self,
        cik: str | None = None,
        event_type: str | None = None,
        days_back: int = 30,
    ) -> pd.DataFrame:
        conn = self.db.get_connection()
        where_clauses = [
            f"generated_at >= current_timestamp - INTERVAL '{days_back} days'"
        ]
        if cik:
            where_clauses.append(f"cik = '{cik}'")
        if event_type:
            where_clauses.append(f"event_type = '{event_type}'")
        where = " AND ".join(where_clauses)
        try:
            return conn.execute(
                f"SELECT * FROM audit_trail WHERE {where} ORDER BY generated_at DESC"
            ).df()
        except Exception:
            return pd.DataFrame()

    def get_summary_stats(self) -> dict:
        conn = self.db.get_connection()
        try:
            result = conn.execute("""
                SELECT
                    count(*) AS total_events,
                    count(DISTINCT cik) AS companies_logged,
                    sum(CASE WHEN event_type='risk_score_generated' THEN 1 ELSE 0 END) AS scoring_events,
                    sum(CASE WHEN human_reviewed THEN 1 ELSE 0 END) AS human_reviews,
                    sum(CASE WHEN final_risk_tier IN ('HIGH','CRITICAL') THEN 1 ELSE 0 END) AS high_risk_flags,
                    min(generated_at) AS first_event,
                    max(generated_at) AS last_event
                FROM audit_trail
            """).df()
            return result.iloc[0].to_dict() if not result.empty else {}
        except Exception:
            return {}
