"""Generates Excel/CSV risk report from ML anomaly scores and dbt marts."""
import pandas as pd
from datetime import datetime
from pathlib import Path
from utils.logger import get_logger

log = get_logger("risk_report")

REPORTS_DIR = Path("reports/output")


def generate_risk_report(db, output_path: str = None) -> str:
    """Generate Excel risk report. Falls back to CSV if openpyxl is missing."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(REPORTS_DIR / f"risk_report_{ts}.xlsx")

    conn = db.get_connection()

    def safe_query(sql: str) -> pd.DataFrame:
        try:
            return conn.execute(sql).df()
        except Exception:
            return pd.DataFrame()

    scores_df = safe_query(
        "SELECT * FROM ml_anomaly_scores ORDER BY ml_risk_score DESC"
    )
    benford_df = safe_query(
        "SELECT * FROM benford_results ORDER BY benford_risk_score DESC"
    )

    risk_df = pd.DataFrame()
    for schema in ("main_marts", "marts"):
        risk_df = safe_query(
            f"SELECT * FROM {schema}.mart_risk_candidates "
            f"ORDER BY pre_ml_risk_score DESC"
        )
        if not risk_df.empty:
            break

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            if not risk_df.empty:
                risk_df.to_excel(writer, sheet_name="Risk Candidates", index=False)
            if not scores_df.empty:
                scores_df.to_excel(writer, sheet_name="ML Scores", index=False)
            if not benford_df.empty:
                benford_df.to_excel(writer, sheet_name="Benford Analysis", index=False)
        log.info("excel_report_generated", path=output_path)
    except ImportError:
        output_path = output_path.replace(".xlsx", ".csv")
        combined = pd.concat(
            [df for df in [risk_df, scores_df] if not df.empty], ignore_index=True
        )
        combined.to_csv(output_path, index=False)
        log.info("csv_report_generated", path=output_path)

    return output_path
