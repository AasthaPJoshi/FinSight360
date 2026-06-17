"""
Pipeline health and data freshness monitoring.
Checks: DB connectivity, data recency, model staleness, ChromaDB status.
"""
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

from utils.logger import get_logger

log = get_logger("health_check")


@dataclass
class HealthStatus:
    component: str
    status: str       # "OK" | "WARN" | "ERROR"
    message: str
    value: str = ""


class PipelineHealthChecker:
    """
    Runs health checks across all FinSight360 components.
    Called by: CLI health command, dashboard sidebar,
    GitHub Actions pipeline verification step.
    """

    def __init__(self):
        self.db_path = os.environ.get("DUCKDB_PATH", "data/finsight360.duckdb")

    def check_all(self) -> list[HealthStatus]:
        return [
            self.check_database(),
            self.check_data_freshness(),
            self.check_model_exists(),
            self.check_chromadb(),
            self.check_reference_files(),
            self.check_mlflow_runs(),
        ]

    def check_database(self) -> HealthStatus:
        try:
            if not Path(self.db_path).exists():
                return HealthStatus(
                    "DuckDB", "ERROR",
                    f"Database not found: {self.db_path}",
                    "Run: python -m finsight360.cli db init",
                )

            conn = duckdb.connect(self.db_path, read_only=True)
            tables = conn.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'main'
            """).df()
            conn.close()

            expected = {
                "companies", "filings", "financial_facts",
                "financial_metrics", "ingestion_runs",
            }
            found = set(tables["table_name"].tolist())
            missing = expected - found

            if missing:
                return HealthStatus(
                    "DuckDB", "WARN",
                    f"Missing tables: {missing}",
                    str(len(found)),
                )

            size_kb = Path(self.db_path).stat().st_size // 1024
            return HealthStatus(
                "DuckDB", "OK",
                f"Connected. {len(found)} tables.",
                f"{size_kb} KB",
            )
        except Exception as e:
            return HealthStatus("DuckDB", "ERROR", str(e)[:80])

    def check_data_freshness(self) -> HealthStatus:
        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            result = conn.execute("""
                SELECT max(started_at) as last_run,
                       count(*) as total_companies
                FROM ingestion_runs
                WHERE status = 'completed'
            """).df()
            conn.close()

            if result.empty or result.iloc[0]["last_run"] is None:
                return HealthStatus(
                    "Data Freshness", "WARN",
                    "No completed ingestion runs found",
                    "Run: python -m finsight360.cli ingest --ticker AAPL",
                )

            last_run = result.iloc[0]["last_run"]
            if hasattr(last_run, "to_pydatetime"):
                last_run = last_run.to_pydatetime()

            age = (
                datetime.utcnow() - last_run.replace(tzinfo=None)
                if hasattr(last_run, "replace")
                else timedelta(days=999)
            )

            if age.days > 7:
                return HealthStatus(
                    "Data Freshness", "WARN",
                    f"Last ingestion {age.days} days ago",
                    str(last_run)[:19],
                )

            return HealthStatus(
                "Data Freshness", "OK",
                f"Last ingestion {age.days}d {age.seconds // 3600}h ago",
                str(last_run)[:19],
            )
        except Exception as e:
            return HealthStatus("Data Freshness", "WARN", str(e)[:80])

    def check_model_exists(self) -> HealthStatus:
        model_path = Path("models/isolation_forest.joblib")
        if not model_path.exists():
            return HealthStatus(
                "ML Model", "WARN",
                "Isolation Forest model not found",
                "Run: python -m finsight360.cli ml-run",
            )

        size_kb = model_path.stat().st_size // 1024
        age_hours = (
            datetime.utcnow()
            - datetime.utcfromtimestamp(model_path.stat().st_mtime)
        ).total_seconds() / 3600

        if age_hours > 168:
            return HealthStatus(
                "ML Model", "WARN",
                f"Model is {age_hours:.0f}h old — consider retraining",
                f"{size_kb} KB",
            )

        return HealthStatus(
            "ML Model", "OK",
            f"Model exists ({size_kb} KB, {age_hours:.0f}h old)",
            f"{size_kb} KB",
        )

    def check_chromadb(self) -> HealthStatus:
        chroma_dir = Path(
            os.environ.get("CHROMA_PERSIST_DIR", "data/chromadb")
        )
        if not chroma_dir.exists():
            return HealthStatus(
                "ChromaDB", "WARN",
                "ChromaDB directory not found",
                "Run: python -m finsight360.cli index --ticker AAPL",
            )

        files = list(chroma_dir.rglob("*"))
        if not files:
            return HealthStatus(
                "ChromaDB", "WARN",
                "ChromaDB empty — no filings indexed",
                "0 files",
            )

        return HealthStatus(
            "ChromaDB", "OK",
            f"ChromaDB active ({len(files)} files)",
            str(chroma_dir),
        )

    def check_reference_files(self) -> HealthStatus:
        required = [
            "data/reference/sp500_auditors.json",
            "data/reference/sp500_relationships.json",
        ]
        missing = [f for f in required if not Path(f).exists()]

        if missing:
            return HealthStatus(
                "Reference Files", "WARN",
                f"Missing: {missing}",
                "Graph layer may fail",
            )
        return HealthStatus(
            "Reference Files", "OK",
            "All reference files present",
            str(len(required)),
        )

    def check_mlflow_runs(self) -> HealthStatus:
        mlruns = Path("mlruns")
        if not mlruns.exists():
            return HealthStatus(
                "MLflow", "WARN",
                "No MLflow runs found",
                "Run: python -m finsight360.cli ml-run",
            )

        experiments = list(mlruns.glob("*/"))
        return HealthStatus(
            "MLflow", "OK",
            f"{len(experiments)} experiment(s) tracked",
            str(mlruns),
        )

    def print_health_report(self) -> tuple[int, int]:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        checks = self.check_all()

        table = Table(title="FinSight360 — Pipeline Health Check")
        table.add_column("Component", style="bold")
        table.add_column("Status")
        table.add_column("Message")
        table.add_column("Value", style="dim")

        status_icons = {
            "OK": "[green]✅ OK[/]",
            "WARN": "[yellow]⚠️  WARN[/]",
            "ERROR": "[red]❌ ERROR[/]",
        }

        for check in checks:
            table.add_row(
                check.component,
                status_icons.get(check.status, check.status),
                check.message,
                check.value,
            )

        console.print(table)

        errors = sum(1 for c in checks if c.status == "ERROR")
        warns = sum(1 for c in checks if c.status == "WARN")

        if errors:
            console.print(f"\n[red]❌ {errors} ERROR(s) — pipeline needs attention[/]")
        elif warns:
            console.print(f"\n[yellow]⚠️  {warns} WARNING(s) — pipeline degraded[/]")
        else:
            console.print("\n[green]✅ All systems healthy[/]")

        return errors, warns
