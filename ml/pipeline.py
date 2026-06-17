"""
End-to-end ML pipeline orchestrator.
Runs: feature build → Benford analysis → Isolation Forest →
      SHAP explanations → MLflow logging → results to DuckDB.
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
from utils.logger import get_logger
from ml.feature_builder import FeatureBuilder, ISOLATION_FOREST_FEATURES
from ml.benford import BenfordAnalyzer
from ml.anomaly_detector import AnomalyDetector
from ml.model_registry import ModelRegistry
from ml.explainer import AnomalyExplainer
from rich.console import Console
from rich.table import Table

log = get_logger("ml_pipeline")
console = Console()


class MLPipeline:
    def __init__(self, db):
        self.db = db
        self.feature_builder = FeatureBuilder(db)
        self.benford = BenfordAnalyzer()
        self.detector = AnomalyDetector(contamination=0.10, n_estimators=200)
        self.registry = ModelRegistry()

    def run(self) -> dict:
        """Execute full ML pipeline. Returns summary dict."""
        log.info("ml_pipeline_started")

        # Step 1: Build feature matrix
        console.print("[bold blue]Step 1/5: Building feature matrix...[/]")
        feature_df = self.feature_builder.build_feature_matrix()

        if feature_df.empty:
            log.error("no_data_for_ml")
            console.print("[bold red]No data found. Run ingestion first.[/]")
            return {"status": "failed", "reason": "no_data"}

        console.print(
            f"  [green]✓[/] {len(feature_df)} companies, "
            f"{len(ISOLATION_FOREST_FEATURES)} features"
        )

        # Step 2: Benford analysis
        console.print("[bold blue]Step 2/5: Running Benford's Law analysis...[/]")
        try:
            benford_df = self.benford.analyze_all_companies(self.db)
        except Exception as exc:
            log.warning("benford_failed", error=str(exc))
            benford_df = pd.DataFrame()
        console.print(f"  [green]✓[/] {len(benford_df)} companies analyzed")
        if not benford_df.empty:
            nc = len(benford_df[benford_df["conformity_rating"] == "Non-conforming"])
            if nc:
                console.print(f"  [yellow]⚠[/] {nc} non-conforming companies")

        # Step 3: Train + predict Isolation Forest
        console.print("[bold blue]Step 3/5: Training Isolation Forest...[/]")
        available_features = [
            f for f in ISOLATION_FOREST_FEATURES if f in feature_df.columns
        ]

        self.detector.fit(feature_df, available_features)
        results = self.detector.predict(feature_df)
        self.detector.save()

        anomaly_count = sum(1 for r in results if r.is_anomaly)
        console.print(
            f"  [green]✓[/] {anomaly_count}/{len(results)} companies flagged as anomalous"
        )

        # Step 4: SHAP explanations for top 10 highest-risk companies
        console.print("[bold blue]Step 4/5: Generating SHAP explanations...[/]")
        explanations = {}
        try:
            explainer = AnomalyExplainer(self.detector)
            top_companies = sorted(
                results, key=lambda r: r.ml_risk_score, reverse=True
            )[:10]

            for company in top_companies:
                try:
                    exp = explainer.explain_company(feature_df, company.cik)
                    explanations[company.cik] = exp
                    narrative = explainer.generate_summary_explanation(
                        exp, company.company_name, company.ml_risk_score
                    )
                    log.info("shap_narrative", cik=company.cik,
                             narrative=narrative[:120])
                except Exception as e:
                    log.warning("shap_failed", cik=company.cik, error=str(e))

            console.print(
                f"  [green]✓[/] SHAP explanations for top {len(top_companies)} companies"
            )
        except Exception as exc:
            log.warning("shap_explainer_init_failed", error=str(exc))
            console.print(f"  [yellow]⚠[/] SHAP unavailable: {exc}")

        # Step 5: Log to MLflow
        console.print("[bold blue]Step 5/5: Logging to MLflow...[/]")
        try:
            run_id = self.registry.log_training_run(
                self.detector, feature_df, results, benford_df,
                params={"contamination": 0.10, "n_estimators": 200},
            )
            console.print(f"  [green]✓[/] MLflow run logged: {run_id[:8]}...")
        except Exception as exc:
            log.warning("mlflow_logging_failed", error=str(exc))
            run_id = "local-run"
            console.print(f"  [yellow]⚠[/] MLflow logging skipped: {exc}")

        # Write results to DuckDB
        self._write_results_to_db(results, benford_df)

        # Print rich summary table
        self._print_summary_table(results)

        benford_nc = (
            len(benford_df[benford_df["conformity_rating"] == "Non-conforming"])
            if not benford_df.empty else 0
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "companies_analyzed": len(results),
            "anomalies_detected": anomaly_count,
            "benford_non_conforming": benford_nc,
        }

    def _write_results_to_db(self, results: list, benford_df: pd.DataFrame):
        """Store ML results in DuckDB for dashboard consumption."""
        conn = self.db.get_connection()
        now = datetime.utcnow()

        # Build scores DataFrame and insert via DuckDB relation
        scores_data = [{
            "cik": r.cik,
            "ticker": r.ticker,
            "company_name": r.company_name,
            "industry_group": r.industry_group,
            "isolation_forest_score": r.isolation_forest_score,
            "anomaly_percentile": r.anomaly_percentile,
            "ml_risk_score": r.ml_risk_score,
            "is_anomaly": r.is_anomaly,
            "pre_ml_risk_score": r.pre_ml_risk_score,
            "risk_tier": r.risk_tier,
            "computed_at": now,
        } for r in results]

        scores_df = pd.DataFrame(scores_data)
        if not scores_df.empty:
            conn.execute("DROP TABLE IF EXISTS ml_anomaly_scores")
            conn.execute(
                "CREATE TABLE ml_anomaly_scores AS SELECT * FROM scores_df"
            )

        if not benford_df.empty:
            benford_out = benford_df.copy()
            benford_out["computed_at"] = now
            conn.execute("DROP TABLE IF EXISTS benford_results")
            conn.execute(
                "CREATE TABLE benford_results AS SELECT * FROM benford_out"
            )

        log.info("ml_results_written_to_db",
                 scores=len(results), benford=len(benford_df))

    def _print_summary_table(self, results: list):
        table = Table(title="FinSight360 — Top 10 Risk Companies")
        table.add_column("Rank", style="dim")
        table.add_column("Ticker")
        table.add_column("Company")
        table.add_column("ML Risk Score", justify="right")
        table.add_column("IF Score", justify="right")
        table.add_column("Anomaly?")
        table.add_column("Risk Tier")

        top = sorted(results, key=lambda r: r.ml_risk_score, reverse=True)[:10]
        for i, r in enumerate(top, 1):
            table.add_row(
                str(i),
                r.ticker or r.cik,
                r.company_name[:30],
                f"{r.ml_risk_score:.1f}",
                f"{r.isolation_forest_score:.4f}",
                "[red]YES[/]" if r.is_anomaly else "[green]No[/]",
                r.risk_tier,
            )

        console.print(table)
