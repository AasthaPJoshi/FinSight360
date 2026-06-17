"""Integration tests that verify dbt models compile and run successfully."""
import os
import subprocess

import pytest


def _dbt_env():
    env = os.environ.copy()
    env["DUCKDB_PATH"] = os.path.abspath("data/finsight360.duckdb")
    return env


@pytest.fixture(scope="module")
def seeded_db(seed_aapl_data):
    """Ensure AAPL data is seeded before dbt runs."""
    return seed_aapl_data


@pytest.fixture(scope="module")
def dbt_run_result(seeded_db):
    result = subprocess.run(
        [
            "dbt",
            "run",
            "--profiles-dir",
            "dbt",
            "--project-dir",
            "dbt",
            "--target",
            "dev",
        ],
        capture_output=True,
        text=True,
        env=_dbt_env(),
    )
    return result


def test_dbt_run_succeeds(dbt_run_result):
    assert dbt_run_result.returncode == 0, (
        f"dbt run failed:\n{dbt_run_result.stderr}\n{dbt_run_result.stdout}"
    )


def test_dbt_staging_models_exist(dbt_run_result):
    assert "stg_companies" in dbt_run_result.stdout
    assert "stg_financial_metrics" in dbt_run_result.stdout


def test_dbt_mart_models_exist(dbt_run_result):
    assert "mart_risk_candidates" in dbt_run_result.stdout
    assert "mart_company_dashboard" in dbt_run_result.stdout


def test_dbt_tests_pass(seeded_db):
    result = subprocess.run(
        [
            "dbt",
            "test",
            "--profiles-dir",
            "dbt",
            "--project-dir",
            "dbt",
        ],
        capture_output=True,
        text=True,
        env=_dbt_env(),
    )
    assert result.returncode == 0, (
        f"dbt tests failed:\n{result.stderr}\n{result.stdout}"
    )
