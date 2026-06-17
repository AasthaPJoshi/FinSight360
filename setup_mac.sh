#!/bin/bash
# FinSight360 — Mac Setup Script (Intel + Apple Silicon)
set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  FinSight360 — Mac Environment Setup                ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Detect architecture
ARCH=$(uname -m)
echo "→ Architecture: $ARCH"

# Check Python 3.11
if ! command -v python3.11 &>/dev/null; then
    echo "⚠️  Python 3.11 not found. Install via Homebrew:"
    echo "   brew install python@3.11"
    exit 1
fi
echo "→ Python 3.11 found: $(python3.11 --version)"

# Check if inside the project directory
if [ ! -f "dashboard/app.py" ]; then
    echo "❌ Run this script from the FinSight360 project root."
    exit 1
fi

# Create virtualenv
echo "→ Creating virtual environment with Python 3.11..."
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q

# Install dependencies
echo "→ Installing Python dependencies..."
echo "   (First install takes ~5 minutes — includes ML libraries)"
pip install -r requirements.txt -q
echo "→ All dependencies installed"

# Setup .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  ACTION REQUIRED: Edit .env                         ║"
    echo "║                                                      ║"
    echo "║  Required for AI Analyst page:                      ║"
    echo "║  GEMINI_API_KEY=your_key_here                       ║"
    echo "║  → Get free key: aistudio.google.com/app/apikey    ║"
    echo "║                                                      ║"
    echo "║  Optional (for full SEC pipeline):                  ║"
    echo "║  OPENAI_API_KEY=sk-...                              ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""
fi

# Create directories
mkdir -p data/raw data/processed data/chromadb models mlruns \
         reports/shap_plots logs data/raw/filings

# Seed demo data
echo "→ Seeding demo dataset (10 S&P 500 companies)..."
DUCKDB_PATH=data/finsight360.duckdb python scripts/seed_demo_data.py
echo "→ Demo data ready"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Setup complete!                                     ║"
echo "║                                                      ║"
echo "║  Launch dashboard:                                   ║"
echo "║    source .venv/bin/activate                        ║"
echo "║    make run-no-seed                                 ║"
echo "║                                                      ║"
echo "║  Or one command:                                     ║"
echo "║    ./start_local.sh                                 ║"
echo "╚══════════════════════════════════════════════════════╝"
