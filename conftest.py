"""
Session-level fixture: seed AAPL data into the shared DuckDB file
before any dbt tests run.  Unit tests use tmp_path databases and
are unaffected.
"""
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring external services (Docker, Neo4j, etc.)",
    )


@pytest.fixture(scope="session", autouse=False)
def seed_aapl_data():
    """Seed AAPL financial data into data/finsight360.duckdb."""
    db_path = os.path.abspath("data/finsight360.duckdb")
    os.environ.setdefault("DUCKDB_PATH", db_path)

    from storage.database import Database
    from tests.fixtures.aapl_seed import seed

    db = Database(db_path=db_path)
    seed(db)
    db.close()
    return db_path
