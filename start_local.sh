#!/bin/bash
# FinSight360 — Local Mac Quickstart
# Run once: chmod +x start_local.sh && ./start_local.sh
set -e

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║  FinSight360 — Local Quickstart      ║"
echo "╚══════════════════════════════════════╝"
echo ""

# 1. Python version check
PY_VERSION=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo "→ Python $PY_VERSION detected"

# 2. Create virtualenv if not present
if [ ! -d "$VENV_DIR" ]; then
    echo "→ Creating virtual environment..."
    $PYTHON -m venv $VENV_DIR
fi

# 3. Activate
source $VENV_DIR/bin/activate
echo "→ Virtualenv activated"

# 4. Install dependencies
echo "→ Installing dependencies (may take 2-3 minutes first time)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "→ Dependencies installed"

# 5. Create .env if missing
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "→ Created .env from .env.example"
    echo "  ⚠️  Edit .env and add your GEMINI_API_KEY to enable AI Analyst"
fi

# 6. Create required directories
mkdir -p data/raw data/processed data/chromadb models mlruns \
         reports/shap_plots logs data/raw/filings

# 7. Seed demo data
echo "→ Seeding 10-company demo dataset..."
DUCKDB_PATH=data/finsight360.duckdb python scripts/seed_demo_data.py
echo "→ Demo data seeded"

# 8. Launch dashboard
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  Launching Dashboard...              ║"
echo "║  Open: http://localhost:8501         ║"
echo "╚══════════════════════════════════════╝"
echo ""

DUCKDB_PATH=data/finsight360.duckdb \
streamlit run dashboard/app.py \
  --server.port 8501 \
  --server.headless false
