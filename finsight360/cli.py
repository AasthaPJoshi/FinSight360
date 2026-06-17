import os
import sys

import click
import structlog

log = structlog.get_logger(__name__)


@click.group()
def cli():
    """FinSight360 — SEC EDGAR financial anomaly detection platform."""
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(20),
        processors=[
            structlog.dev.ConsoleRenderer(),
        ],
    )


@cli.command()
@click.option("--ticker", required=True, help="Stock ticker symbol (e.g. AAPL)")
@click.option("--db", default="data/finsight360.duckdb", help="Path to DuckDB database")
@click.option("--no-dbt", is_flag=True, default=False, help="Skip dbt run after ingestion")
def ingest(ticker: str, db: str, no_dbt: bool):
    """Ingest SEC EDGAR filings and XBRL facts for a ticker."""
    from storage.database import Database
    from ingestion.edgar_client import EdgarClient
    from ingestion.filing_processor import FilingProcessor

    os.environ.setdefault("DUCKDB_PATH", os.path.abspath(db))

    click.echo(f"Ingesting {ticker.upper()} into {db} ...")
    database = Database(db_path=db)
    client = EdgarClient()
    processor = FilingProcessor(db=database, client=client)

    try:
        result = processor.process_ticker(ticker)
        if no_dbt:
            pass
        click.echo(f"Done: {result}")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        client.close()
        database.close()


@cli.command()
@click.option("--db", default="data/finsight360.duckdb", help="Path to DuckDB database")
def run_dbt(db: str):
    """Run dbt transformations on the ingested data."""
    from storage.database import Database

    os.environ.setdefault("DUCKDB_PATH", os.path.abspath(db))
    database = Database(db_path=db)
    result = database.run_dbt_transformations()
    if result["returncode"] == 0:
        click.echo("dbt run succeeded.")
    else:
        click.echo("dbt run failed.", err=True)
        sys.exit(1)
    database.close()


@cli.command()
@click.option("--db", default="data/finsight360.duckdb", help="Path to DuckDB database")
def ml_run(db: str):
    """Run the full ML anomaly detection pipeline (Benford + Isolation Forest + SHAP)."""
    from storage.database import Database
    from ml.pipeline import MLPipeline

    os.environ.setdefault("DUCKDB_PATH", os.path.abspath(db))
    database = Database(db_path=db)

    try:
        pipeline = MLPipeline(database)
        result = pipeline.run()
        click.echo(f"\nResult: {result}")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        database.close()


@cli.command()
def ml_status():
    """Show the latest MLflow run metrics."""
    from ml.model_registry import ModelRegistry

    registry = ModelRegistry()
    latest = registry.get_latest_run()

    if not latest:
        click.echo("No MLflow runs found. Run 'ml-run' first.")
        return

    click.echo(f"Latest run: {latest['run_id']}")
    click.echo("Metrics:")
    for k, v in sorted(latest.get("metrics", {}).items()):
        click.echo(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    click.echo("Params:")
    for k, v in sorted(latest.get("params", {}).items()):
        click.echo(f"  {k}: {v}")


@cli.command()
@click.option("--db", default="data/finsight360.duckdb", help="Path to DuckDB database")
def benford(db: str):
    """Run Benford's Law forensic analysis on all ingested companies."""
    from storage.database import Database
    from ml.benford import BenfordAnalyzer
    from rich.console import Console
    from rich.table import Table

    os.environ.setdefault("DUCKDB_PATH", os.path.abspath(db))
    database = Database(db_path=db)
    console = Console()

    try:
        analyzer = BenfordAnalyzer()
        results_df = analyzer.analyze_all_companies(database)

        if results_df.empty:
            click.echo("No data found. Run ingestion first.")
            return

        table = Table(title="Benford's Law Forensic Analysis")
        table.add_column("Ticker")
        table.add_column("N Obs", justify="right")
        table.add_column("MAD Score", justify="right")
        table.add_column("Chi² p-value", justify="right")
        table.add_column("Conformity")
        table.add_column("Risk Score", justify="right")

        for _, row in results_df.sort_values(
            "benford_risk_score", ascending=False
        ).iterrows():
            conformity = row["conformity_rating"]
            color = (
                "red" if conformity == "Non-conforming"
                else "yellow" if conformity == "Marginal"
                else "green"
            )
            table.add_row(
                str(row["ticker"]),
                str(row["n_observations"]),
                f"{row['mad_score']:.5f}",
                f"{row['chi_square_p_value']:.4f}",
                f"[{color}]{conformity}[/]",
                f"{row['benford_risk_score']:.1f}",
            )

        console.print(table)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        database.close()


@cli.command("graph-build")
@click.option("--db", default="data/finsight360.duckdb", help="Path to DuckDB database")
@click.option("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
@click.option("--neo4j-user", default="neo4j", help="Neo4j username")
@click.option("--neo4j-password", default="finsight360", help="Neo4j password")
@click.option("--skip-neo4j", is_flag=True, default=False, help="Skip Neo4j loading")
def graph_build(db: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, skip_neo4j: bool):
    """Build the corporate knowledge graph (NetworkX + optional Neo4j)."""
    from storage.database import Database
    from graph.graph_builder import CorporateGraphBuilder
    from graph.cluster_detector import ClusterDetector
    from graph.neo4j_loader import Neo4jLoader
    from graph.graph_queries import GraphQueryEngine
    from graph.graph_risk_scorer import GraphRiskScorer
    from rich.console import Console
    from rich.table import Table

    os.environ.setdefault("DUCKDB_PATH", os.path.abspath(db))
    console = Console()
    database = Database(db_path=db)

    try:
        console.print("\n[bold cyan]Phase 4: Corporate Knowledge Graph[/bold cyan]")

        console.print("[yellow]Building graph...[/yellow]")
        builder = CorporateGraphBuilder(database)
        G = builder.build()
        stats = builder.get_stats()

        stats_table = Table(title="Graph Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right", style="white")
        stats_table.add_row("Company Nodes", str(stats.n_companies))
        stats_table.add_row("Auditor Nodes", str(stats.n_auditors))
        stats_table.add_row("Subsidiary Edges", str(stats.n_subsidiaries))
        stats_table.add_row("Total Edges", str(stats.n_edges))
        stats_table.add_row("High-Risk Nodes", str(stats.high_risk_nodes))
        console.print(stats_table)

        console.print("[yellow]Detecting communities...[/yellow]")
        detector = ClusterDetector(G)
        partition = detector.detect_communities()
        centrality = detector.compute_centrality()
        n_communities = len(set(partition.values())) if partition else 0
        console.print(f"  Communities found: [bold]{n_communities}[/bold]")

        cluster_df = detector.identify_risk_clusters(partition)
        if not cluster_df.empty:
            cluster_table = Table(title="Top Risk Clusters")
            cluster_table.add_column("Community", justify="right")
            cluster_table.add_column("Companies", justify="right")
            cluster_table.add_column("Avg ML Score", justify="right")
            cluster_table.add_column("Anomalies", justify="right")
            cluster_table.add_column("Risk Level")
            for _, row in cluster_df.head(5).iterrows():
                color = "red" if row["cluster_risk_level"] == "HIGH" else "yellow" if row["cluster_risk_level"] == "MEDIUM" else "green"
                cluster_table.add_row(
                    str(int(row["community_id"])),
                    str(int(row["company_count"])),
                    f"{row['avg_ml_risk_score']:.1f}",
                    str(int(row["anomaly_count"])),
                    f"[{color}]{row['cluster_risk_level']}[/]",
                )
            console.print(cluster_table)

        connectors = detector.find_critical_connectors(top_n=5)
        if connectors:
            conn_table = Table(title="Critical Network Connectors")
            conn_table.add_column("Node")
            conn_table.add_column("Type")
            conn_table.add_column("Betweenness", justify="right")
            conn_table.add_column("PageRank", justify="right")
            for c in connectors:
                conn_table.add_row(
                    c["name"][:30],
                    c["node_type"],
                    f"{c['betweenness']:.4f}",
                    f"{c['pagerank']:.4f}",
                )
            console.print(conn_table)

        if not skip_neo4j:
            console.print("[yellow]Connecting to Neo4j...[/yellow]")
            loader = Neo4jLoader(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
            neo4j_ok = loader.connect()
            if neo4j_ok:
                loader.setup_schema()
                load_result = loader.load_graph(G)
                console.print(f"  Neo4j load: [green]{load_result}[/green]")
            else:
                console.print("  [dim]Neo4j not available — skipping[/dim]")
            engine = GraphQueryEngine(
                neo4j_loader=loader if neo4j_ok else None,
                networkx_graph=G,
            )
        else:
            loader = None
            engine = GraphQueryEngine(neo4j_loader=None, networkx_graph=G)

        high_risk = engine.query_high_risk_companies(threshold=70.0, limit=10)
        if high_risk:
            hr_table = Table(title="High-Risk Companies (Score ≥ 70)")
            hr_table.add_column("Ticker")
            hr_table.add_column("Company")
            hr_table.add_column("ML Score", justify="right")
            hr_table.add_column("Risk Tier")
            for r in high_risk:
                hr_table.add_row(
                    str(r.get("ticker", "")),
                    str(r.get("company_name", ""))[:35],
                    f"{r.get('ml_risk_score', 0):.1f}",
                    str(r.get("risk_tier", "")),
                )
            console.print(hr_table)

        console.print("[yellow]Computing final combined risk scores...[/yellow]")
        scorer = GraphRiskScorer(database)
        final_df = scorer.compute_final_scores(G)
        if not final_df.empty:
            console.print(f"  Final scores computed for [bold]{len(final_df)}[/bold] companies")
            top5 = final_df.nlargest(5, "final_risk_score")
            final_table = Table(title="Top 5 Final Risk Scores (50% ML + 30% Benford + 20% Network)")
            final_table.add_column("Ticker")
            final_table.add_column("ML Score", justify="right")
            final_table.add_column("Benford Score", justify="right")
            final_table.add_column("Network Score", justify="right")
            final_table.add_column("Final Score", justify="right")
            final_table.add_column("Tier")
            for _, row in top5.iterrows():
                tier = row["final_risk_tier"]
                color = "red" if tier == "HIGH" else "yellow" if tier in ("MEDIUM", "LOW") else "green"
                final_table.add_row(
                    str(row.get("ticker", "")),
                    f"{row['ml_risk_score']:.1f}",
                    f"{row['benford_risk_score']:.1f}",
                    f"{row['network_risk_score']:.1f}",
                    f"[bold]{row['final_risk_score']:.1f}[/bold]",
                    f"[{color}]{tier}[/]",
                )
            console.print(final_table)
        else:
            console.print("  [dim]No ML scores available — run ml-run first[/dim]")

        if loader is not None:
            loader.close()

        console.print("\n[bold green]Graph build complete.[/bold green]")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        database.close()


@cli.command("index")
@click.option("--ticker", required=True, help="Ticker to index (e.g. AAPL)")
@click.option("--db", default="data/finsight360.duckdb", help="Path to DuckDB database")
def index(ticker: str, db: str):
    """Index a company's SEC 10-K filings into ChromaDB for RAG."""
    from storage.database import Database
    from genai.qa_interface import FinSightQA

    os.environ.setdefault("DUCKDB_PATH", os.path.abspath(db))
    database = Database(db_path=db)

    try:
        qa = FinSightQA(database)
        click.echo(f"Indexing {ticker.upper()} filings into ChromaDB...")
        result = qa.index_company(ticker)
        if result["status"] == "error":
            click.echo(f"Error: {result['message']}", err=True)
            sys.exit(1)
        click.echo(
            f"Done: {result['docs_added']} chunks added "
            f"({result['total_in_store']} total in store)"
        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        database.close()


@cli.command("ask")
@click.argument("question")
@click.option("--ticker", default=None, help="Scope query to a specific ticker")
@click.option("--db", default="data/finsight360.duckdb", help="Path to DuckDB database")
def ask(question: str, ticker: str | None, db: str):
    """Ask a natural language question about SEC filings."""
    from storage.database import Database
    from genai.qa_interface import FinSightQA

    os.environ.setdefault("DUCKDB_PATH", os.path.abspath(db))
    database = Database(db_path=db)

    try:
        qa = FinSightQA(database)
        answer = qa.ask(question, ticker=ticker)
        click.echo(f"\n{answer}")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        database.close()


@cli.command("brief")
@click.option("--ticker", required=True, help="Ticker for risk brief (e.g. AAPL)")
@click.option("--db", default="data/finsight360.duckdb", help="Path to DuckDB database")
def brief(ticker: str, db: str):
    """Generate a full LLM risk brief for a company."""
    from storage.database import Database
    from genai.qa_interface import FinSightQA

    os.environ.setdefault("DUCKDB_PATH", os.path.abspath(db))
    database = Database(db_path=db)

    try:
        qa = FinSightQA(database)
        result = qa.generate_risk_brief(ticker)
        if result.get("status") == "error":
            click.echo(f"Error: {result['message']}", err=True)
            sys.exit(1)
        click.echo(f"\n{result.get('brief', 'No brief generated.')}")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        database.close()


@cli.command("genai-status")
@click.option("--db", default="data/finsight360.duckdb", help="Path to DuckDB database")
def genai_status(db: str):
    """Show ChromaDB vector store stats."""
    from storage.database import Database
    from genai.qa_interface import FinSightQA
    from rich.console import Console
    from rich.table import Table

    os.environ.setdefault("DUCKDB_PATH", os.path.abspath(db))
    database = Database(db_path=db)
    console = Console()

    try:
        qa = FinSightQA(database)
        stats = qa.get_store_stats()

        table = Table(title="ChromaDB Vector Store Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Collection", stats["collection_name"])
        table.add_row("Total Chunks", str(stats["total_chunks"]))
        table.add_row("Unique Sources", str(stats["unique_sources"]))
        table.add_row("Persist Dir", stats["persist_dir"])
        console.print(table)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        database.close()


@cli.command("shap-run")
@click.option("--db", default="data/finsight360.duckdb", help="Path to DuckDB database")
@click.option("--model", default="models/isolation_forest.joblib", help="Path to saved model")
def shap_run(db: str, model: str):
    """Compute SHAP explainability values for all scored companies."""
    from storage.database import Database
    from governance.shap_explainer import SHAPExplainer
    from rich.console import Console

    os.environ.setdefault("DUCKDB_PATH", os.path.abspath(db))
    database = Database(db_path=db)
    console = Console()

    try:
        conn = database.get_connection()
        try:
            feature_df = conn.execute(
                "SELECT * FROM ml_features"
            ).df()
        except Exception:
            click.echo("No ml_features table found. Run ml-run first.", err=True)
            sys.exit(1)

        if feature_df.empty:
            click.echo("No feature data found. Run ml-run first.", err=True)
            sys.exit(1)

        from governance.shap_explainer import FEATURE_FRIENDLY_NAMES
        feature_cols = [c for c in FEATURE_FRIENDLY_NAMES if c in feature_df.columns]
        if not feature_cols:
            click.echo("No recognized feature columns in ml_features.", err=True)
            sys.exit(1)

        explainer = SHAPExplainer(database)
        shap_df = explainer.explain_all_companies(feature_df, feature_cols, model_path=model)

        if shap_df.empty:
            console.print("[yellow]SHAP values could not be computed (model not found?).[/yellow]")
        else:
            console.print(
                f"[bold green]SHAP complete:[/bold green] {len(shap_df)} feature-company rows stored"
            )
            summary_path = explainer.plot_summary(shap_df)
            if summary_path:
                console.print(f"  Global importance chart: [cyan]{summary_path}[/cyan]")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        database.close()


@cli.command("governance-report")
@click.option("--db", default="data/finsight360.duckdb", help="Path to DuckDB database")
@click.option("--output-dir", default="docs", help="Output directory for report files")
def governance_report(db: str, output_dir: str):
    """Run full governance audit: bias analysis + audit trail summary + model card."""
    from storage.database import Database
    from governance.audit_trail import AuditTrailManager
    from governance.bias_analyzer import BiasAnalyzer
    from governance.model_card import generate_model_card
    from governance.kql_queries import export_kql_file
    from rich.console import Console
    from rich.table import Table

    os.environ.setdefault("DUCKDB_PATH", os.path.abspath(db))
    database = Database(db_path=db)
    console = Console()

    try:
        console.print("\n[bold cyan]FinSight360 — Governance Report[/bold cyan]")

        # Audit trail summary
        mgr = AuditTrailManager(database)
        stats = mgr.get_summary_stats()
        if stats:
            audit_table = Table(title="Audit Trail Summary")
            audit_table.add_column("Metric", style="cyan")
            audit_table.add_column("Value", justify="right")
            for k, v in stats.items():
                audit_table.add_row(str(k).replace("_", " ").title(), str(v))
            console.print(audit_table)
        else:
            console.print("[dim]No audit trail events found.[/dim]")

        # Bias analysis
        conn = database.get_connection()
        try:
            scores_df = conn.execute("SELECT * FROM final_risk_scores").df()
        except Exception:
            scores_df = None

        if scores_df is not None and not scores_df.empty:
            analyzer = BiasAnalyzer()
            reports = analyzer.run_full_bias_audit(scores_df)
            summary_df = analyzer.generate_bias_summary_table(reports)
            if not summary_df.empty:
                bias_table = Table(title="Bias Analysis Summary")
                for col in summary_df.columns:
                    bias_table.add_column(col)
                for _, row in summary_df.iterrows():
                    bias_table.add_row(*[str(v) for v in row.values])
                console.print(bias_table)
        else:
            console.print("[dim]No final_risk_scores found — skipping bias analysis.[/dim]")

        # Model card
        console.print("[yellow]Generating model card...[/yellow]")
        mc_result = generate_model_card(database, output_dir=output_dir)
        console.print(f"  JSON: [cyan]{mc_result['json_path']}[/cyan]")
        console.print(f"  Markdown: [cyan]{mc_result['md_path']}[/cyan]")

        # KQL export
        kql_path = export_kql_file(f"{output_dir}/kql_queries.kql")
        console.print(f"  KQL: [cyan]{kql_path}[/cyan]")

        console.print("\n[bold green]Governance report complete.[/bold green]")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        database.close()


@cli.command("model-card")
@click.option("--db", default="data/finsight360.duckdb", help="Path to DuckDB database")
@click.option("--output-dir", default="docs", help="Output directory for model card files")
def model_card(db: str, output_dir: str):
    """Generate model card (JSON + Markdown) from live DuckDB stats."""
    from storage.database import Database
    from governance.model_card import generate_model_card
    from rich.console import Console

    os.environ.setdefault("DUCKDB_PATH", os.path.abspath(db))
    database = Database(db_path=db)
    console = Console()

    try:
        result = generate_model_card(database, output_dir=output_dir)
        console.print("[bold green]Model card generated:[/bold green]")
        console.print(f"  JSON: [cyan]{result['json_path']}[/cyan]")
        console.print(f"  Markdown: [cyan]{result['md_path']}[/cyan]")
        data = result["model_card_data"]
        console.print(
            f"  Model: {data['model_details']['name']} v{data['model_details']['version']}"
        )
        console.print(
            f"  Companies scored: {data['performance']['total_companies_scored']}"
        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        database.close()


@cli.command("seed-demo")
def seed_demo():
    """Seed 10 realistic demo companies into DuckDB for dashboard preview."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "seed_demo_data",
        Path(__file__).parent.parent / "scripts" / "seed_demo_data.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from storage.database import Database
    database = Database()
    try:
        mod.seed(database)
    finally:
        database.close()


@cli.command("health")
def health():
    """Run pipeline health checks (DB, model, ChromaDB, reference files, MLflow)."""
    from mlops.health_check import PipelineHealthChecker

    checker = PipelineHealthChecker()
    errors, warns = checker.print_health_report()
    if errors:
        sys.exit(1)


@cli.command("drift-check")
@click.option("--db", default="data/finsight360.duckdb", help="Path to DuckDB database")
def drift_check(db: str):
    """Check for model score drift between pipeline runs."""
    from storage.database import Database
    from mlops.model_monitor import ModelMonitor
    from rich.console import Console
    from rich.table import Table

    os.environ.setdefault("DUCKDB_PATH", os.path.abspath(db))
    database = Database(db_path=db)
    console = Console()

    try:
        monitor = ModelMonitor(database)
        report = monitor.compare_runs()

        console.print(f"\n[bold cyan]Model Drift Report[/bold cyan]")
        console.print(f"  Status: [bold]{report.get('status', 'unknown')}[/bold]")
        console.print(f"  Message: {report.get('message', '')}")

        if "score_drift" in report:
            drift = report["score_drift"]
            color = "red" if abs(drift) > 10 else "yellow" if abs(drift) > 5 else "green"
            console.print(
                f"  Score drift: [{color}]{drift:+.1f}[/] points "
                f"({report.get('previous_avg_score', '?')} → {report.get('current_avg_score', '?')})"
            )

        alerts = report.get("alerts", [])
        if alerts:
            console.print(f"\n[red]⚠️  {len(alerts)} drift alert(s):[/]")
            for alert in alerts:
                console.print(
                    f"  [{alert['severity']}] {alert['type']}: {alert['message']}"
                )
        else:
            console.print("\n[green]✅ No drift alerts[/]")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        database.close()


@cli.command("registry-status")
def registry_status():
    """Show MLflow model registry status and all model versions."""
    from mlops.registry import ModelRegistryManager

    manager = ModelRegistryManager()
    manager.print_registry_status()


@cli.command("promote")
@click.option("--version", required=True, help="Model version to promote (e.g. 1)")
@click.option(
    "--stage",
    required=True,
    type=click.Choice(["staging", "production"], case_sensitive=False),
    help="Target stage",
)
def promote(version: str, stage: str):
    """Promote a model version to Staging or Production in MLflow registry."""
    from mlops.registry import ModelRegistryManager
    from rich.console import Console

    console = Console()
    manager = ModelRegistryManager()

    if stage.lower() == "staging":
        ok = manager.promote_to_staging(version)
    else:
        ok = manager.promote_to_production(version)

    if ok:
        console.print(
            f"[green]✅ Version {version} promoted to {stage.title()}[/]"
        )
    else:
        console.print(f"[red]❌ Promotion failed — check MLflow logs[/]")
        sys.exit(1)


@cli.command("ingest-kaggle")
@click.option("--db", default="data/finsight360.duckdb", help="Path to DuckDB database")
@click.option(
    "--max-companies",
    default=100,
    show_default=True,
    help="Max companies to load (0 = all)",
)
@click.option("--no-dbt", is_flag=True, default=False, help="Skip dbt run after loading")
def ingest_kaggle(db: str, max_companies: int, no_dbt: bool):
    """Load S&P 500 data from Kaggle CSVs into DuckDB (no SEC API required)."""
    from storage.database import Database
    from ingestion.kaggle_loader import KaggleDataLoader
    from rich.console import Console
    from rich.table import Table

    os.environ.setdefault("DUCKDB_PATH", os.path.abspath(db))
    database = Database(db_path=db)
    console = Console()

    try:
        loader = KaggleDataLoader(database)
        ok, msg = loader.check_files_exist()
        if not ok:
            console.print(f"[red]CSV files not found.[/]\n{msg}")
            sys.exit(1)

        console.print("\n[bold cyan]FinSight360 — Kaggle Data Loader[/bold cyan]")
        result = loader.run(max_companies=max_companies)

        table = Table(title="Kaggle Load Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="bold white")
        table.add_row("Status", result.get("status", "?"))
        table.add_row("Companies loaded", str(result.get("companies", 0)))
        table.add_row("Annual metric rows", str(result.get("metrics", 0)))
        table.add_row("Filing records", str(result.get("filings", 0)))
        console.print(table)

        if not no_dbt:
            console.print("\n[yellow]Running dbt transformations...[/yellow]")
            dbt_result = database.run_dbt_transformations()
            if dbt_result.get("returncode") == 0:
                console.print("[green]dbt run succeeded.[/green]")
            else:
                console.print("[yellow]dbt run returned non-zero — check dbt logs.[/yellow]")

        console.print(
            "\n[bold green]✅ Kaggle ingestion complete. "
            "Run: streamlit run dashboard/app.py[/bold green]"
        )

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        database.close()


@cli.command("download-kaggle")
def download_kaggle():
    """Download S&P 500 Kaggle dataset (requires ~/.kaggle/kaggle.json)."""
    import subprocess as _sp

    result = _sp.run(
        [sys.executable, "scripts/download_kaggle_data.py"],
        capture_output=False,
    )
    sys.exit(result.returncode)


def main():
    cli()


if __name__ == "__main__":
    main()
