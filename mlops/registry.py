"""
MLflow model promotion workflow.
Manages: Staging → Production promotion, versioning, rollback.
"""
import mlflow
from mlflow.tracking import MlflowClient

from utils.logger import get_logger

log = get_logger("registry")

MODEL_NAME = "finsight360-anomaly-detector"


class ModelRegistryManager:
    """
    Manages model lifecycle in MLflow registry.

    Stages: None → Staging → Production → Archived
    """

    def __init__(self):
        mlflow.set_tracking_uri("mlruns")
        self.client = MlflowClient()

    def get_latest_versions(self) -> list[dict]:
        """List all registered model versions."""
        try:
            versions = self.client.search_model_versions(
                f"name='{MODEL_NAME}'"
            )
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id[:8] if v.run_id else "N/A",
                    "status": v.status,
                    "created": str(v.creation_timestamp),
                }
                for v in versions
            ]
        except Exception as e:
            log.warning("registry_list_failed", error=str(e))
            return []

    def promote_to_staging(self, version: str) -> bool:
        """Promote a model version to Staging."""
        try:
            self.client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version,
                stage="Staging",
                archive_existing_versions=False,
            )
            log.info("promoted_to_staging", version=version)
            return True
        except Exception as e:
            log.error("promotion_failed", error=str(e))
            return False

    def promote_to_production(self, version: str) -> bool:
        """Promote model to Production (archives current prod)."""
        try:
            self.client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
            log.info("promoted_to_production", version=version)
            return True
        except Exception as e:
            log.error("production_promotion_failed", error=str(e))
            return False

    def get_production_model_uri(self) -> str | None:
        """Get URI of current production model."""
        try:
            versions = self.client.get_latest_versions(
                MODEL_NAME, stages=["Production"]
            )
            if versions:
                return f"models:/{MODEL_NAME}/Production"
            return None
        except Exception:
            return None

    def print_registry_status(self):
        from rich.console import Console
        from rich.table import Table

        console = Console()
        versions = self.get_latest_versions()

        if not versions:
            console.print("[yellow]No registered models found.[/]")
            console.print("Run: [blue]python -m finsight360.cli ml-run[/]")
            return

        table = Table(title=f"MLflow Model Registry: {MODEL_NAME}")
        table.add_column("Version")
        table.add_column("Stage")
        table.add_column("Run ID")
        table.add_column("Status")

        stage_styles = {
            "Production": "[green]🟢 Production[/]",
            "Staging": "[yellow]🟡 Staging[/]",
            "Archived": "[dim]📦 Archived[/]",
            "None": "[blue]🔵 None[/]",
        }

        for v in versions:
            table.add_row(
                f"v{v['version']}",
                stage_styles.get(v["stage"], v["stage"]),
                v["run_id"],
                v["status"],
            )

        console.print(table)
