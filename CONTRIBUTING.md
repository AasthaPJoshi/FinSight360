# Contributing to FinSight360

Thank you for your interest in contributing to FinSight360. This document describes how to set up your development environment, follow project conventions, and submit a pull request that will be reviewed and merged efficiently.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Branch Naming Conventions](#branch-naming-conventions)
4. [Development Workflow](#development-workflow)
5. [Running Tests](#running-tests)
6. [Code Style](#code-style)
7. [Pull Request Template](#pull-request-template)
8. [Contact](#contact)

---

## Code of Conduct

This project follows a straightforward standard: be respectful, be constructive, and focus on the work. Pull requests that introduce working, tested, well-documented code are welcome regardless of the contributor's background. Issues that identify genuine bugs or gaps — with reproduction steps — are equally valued.

---

## Getting Started

### Fork and Clone

```bash
# 1. Fork the repository on GitHub (click Fork at the top-right)

# 2. Clone your fork
git clone https://github.com/<your-username>/FinSight360.git
cd FinSight360

# 3. Add the upstream remote
git remote add upstream https://github.com/AasthaPJoshi/FinSight360.git

# 4. Verify both remotes exist
git remote -v
# origin    https://github.com/<your-username>/FinSight360.git (fetch)
# upstream  https://github.com/AasthaPJoshi/FinSight360.git (fetch)
```

### Environment Setup

```bash
# Create and activate a virtual environment (Python 3.11 required)
python3.11 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate.bat     # Windows

# Install all dependencies
pip install -r requirements.txt

# Install dev extras (ruff, pytest)
pip install ruff pytest pytest-cov

# Copy environment template
cp .env.example .env
# Edit .env: set EDGAR_USER_AGENT to "Your Name your@email.com"
# Add OPENAI_API_KEY if working on GenAI features

# Initialize the database
python -m finsight360.cli db init

# Seed demo data to verify the setup works
python scripts/seed_demo_data.py

# Verify the dashboard launches
streamlit run dashboard/app.py &
# Should open http://localhost:8501 without errors
```

### Staying Up to Date

Before starting any work, sync your fork:

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

---

## Branch Naming Conventions

All work must happen on a dedicated branch. Never commit directly to `main`.

| Type | Pattern | Example |
|------|---------|---------|
| New feature | `feature/<short-description>` | `feature/add-z-score-signal` |
| Bug fix | `fix/<what-was-broken>` | `fix/benford-mad-calculation` |
| Documentation | `docs/<what-is-updated>` | `docs/update-cli-reference` |
| Refactor | `refactor/<what-was-changed>` | `refactor/extract-graph-scorer` |
| Test | `test/<what-is-being-tested>` | `test/add-audit-trail-coverage` |
| Chore | `chore/<what-was-done>` | `chore/pin-scipy-version` |

```bash
# Create your branch from an up-to-date main
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

---

## Development Workflow

### Making Changes

1. Make your changes in small, logically grouped commits.
2. Write tests for any new functionality (see [Running Tests](#running-tests)).
3. Update docstrings for any modified public functions.
4. If you are adding a new module, add it to the Project Structure section of `README.md`.
5. If you are changing CLI commands, update the CLI Reference table in `README.md`.

### Commit Message Format

Use the imperative present tense. Keep the first line under 72 characters. Reference issues where relevant.

```
Add Z-score network risk signal to GraphRiskScorer

Introduces a per-community Z-score calculation to detect companies
whose risk score deviates significantly from their cluster mean.
Weighted at 15% of network signal (replacing flat average).

Closes #42
```

**Avoid:**
- "Fixed some stuff"
- "WIP"
- "Updated code"
- Commit messages that describe what you did rather than why

---

## Running Tests

All pull requests must pass the full test suite. Tests are split into unit tests (fast, no external dependencies) and integration tests (require DuckDB with seeded data and optionally network access).

### Unit Tests (required before every PR)

```bash
# Run all unit tests (excludes integration-marked tests)
python -m pytest tests/ -v -m "not integration"

# Expected output: 69 passed, 0 failed
# This must be green before submitting any PR
```

### Full Test Suite (including integration)

```bash
# Run everything including dbt integration tests
python -m pytest tests/ -v

# Note: test_dbt_models.py requires dbt installed and
# DuckDB initialized with AAPL fixture data
```

### Coverage Report

```bash
python -m pytest tests/ -m "not integration" --cov=. --cov-report=term-missing
# Aim for ≥ 85% coverage on any new module you add
```

### dbt Tests

```bash
# After running dbt models
DUCKDB_PATH=data/finsight360.duckdb dbt test --profiles-dir dbt --project-dir dbt
# Expected: 21 tests, 0 failures
```

### What to Test

- **New ML signals**: test that the scorer returns a float in [0, 100], handles empty input, and produces deterministic output given the same seed
- **New dbt models**: add a `not_null` and `unique` test for the primary key in `sources.yml`
- **New CLI commands**: test that the command instantiates correctly and returns without exception on demo data
- **New governance features**: test that audit records are written and that bias scores are finite

---

## Code Style

### Python

FinSight360 uses **ruff** for linting and formatting. All code must pass `ruff check` with no errors before submission.

```bash
# Check for linting errors
ruff check .

# Auto-fix fixable issues
ruff check --fix .

# Format code
ruff format .
```

### Type Hints

All public function signatures must include type hints. Private helper functions (`_prefixed`) are encouraged but not required.

```python
# Good
def compute_mad_score(digits: list[int]) -> float:
    ...

# Bad
def compute_mad_score(digits):
    ...
```

### Docstrings

One-line docstrings for simple functions. Multi-line only for public APIs that benefit from parameter documentation.

```python
# Good — one line, imperative verb
def compute_benford_score(values: list[float]) -> float:
    """Return MAD-based Benford conformity score in [0, 100]."""
    ...

# Good — multi-line for complex public API
def score_company(cik: str, periods: int = 8) -> RiskScore:
    """
    Compute composite risk score for one company.

    Args:
        cik: SEC CIK identifier (padded or unpadded; normalized internally)
        periods: Number of historical periods to include in feature matrix

    Returns:
        RiskScore dataclass with ml_score, benford_score, network_score,
        final_score, and tier fields.
    """
    ...
```

### No Magic Numbers

Name constants at the module level. Use `config/settings.py` for values that may need to change across environments.

```python
# Good
CONTAMINATION_RATE = 0.10
N_ESTIMATORS = 200

# Bad
detector = IsolationForest(contamination=0.10, n_estimators=200)
```

### Structured Logging

Use `structlog` for all logging. Never use `print()` in production code. Use the bound logger from `utils/logger.py`.

```python
from utils.logger import get_logger
log = get_logger(__name__)

log.info("scoring_complete", cik=cik, final_score=score, tier=tier)
# NOT: log.info("scoring_complete", event=...)   ← 'event' is reserved by structlog
```

---

## Pull Request Template

When opening a pull request, fill in the following template completely. PRs without a completed template will be returned for revision.

```markdown
## Summary

<!-- One or two sentences: what does this PR do and why? -->

## Changes Made

<!-- Bullet list of specific changes -->
- 
- 
- 

## Tests Added / Modified

<!-- Which test file(s) did you add or modify? -->
- [ ] `tests/test_<module>.py` — added N tests for <feature>
- [ ] All 69 existing tests still pass (`pytest -m "not integration"`)
- [ ] dbt tests pass (21 passing)

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that changes existing behavior)
- [ ] Documentation update

## Checklist

- [ ] `ruff check .` passes with no errors
- [ ] All new public functions have type hints
- [ ] All new public functions have docstrings
- [ ] `README.md` updated if CLI commands or project structure changed
- [ ] No hardcoded secrets or API keys in any file
- [ ] `python -m pytest tests/ -m "not integration"` passes (69 tests)
```

---

## Contact

For questions, suggestions, or to discuss a significant change before implementing it, open a GitHub Issue or reach out via GitHub:

- **GitHub:** [github.com/AasthaPJoshi](https://github.com/AasthaPJoshi)
- **LinkedIn:** [linkedin.com/in/aasthajoshi14](https://www.linkedin.com/in/aasthajoshi14/)

For security vulnerabilities, please do not open a public issue. Contact directly via GitHub private message.

---

*Thank you for contributing to FinSight360.*
