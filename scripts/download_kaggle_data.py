"""
One-time script to download S&P 500 financial data from Kaggle.

Dataset: andrewmvd/sp-500-stocks
Contains:
  sp500_companies.csv  — ticker, security name, GICS sector, CIK
  sp500_stocks.csv     — daily OHLCV for each S&P 500 ticker

Run this once before using: python -m finsight360.cli ingest-kaggle
"""
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("data/kaggle")
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "andrewmvd/sp-500-stocks"


def check_kaggle_configured() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def download() -> bool:
    if not check_kaggle_configured():
        print("""
Kaggle API not configured. Two options:

OPTION A — Kaggle API (automatic):
  1. Go to: kaggle.com → Your Profile → Settings → API → Create New Token
  2. Save kaggle.json, then run:
       mkdir -p ~/.kaggle
       mv ~/Downloads/kaggle.json ~/.kaggle/
       chmod 600 ~/.kaggle/kaggle.json
  3. Run this script again:
       python scripts/download_kaggle_data.py

OPTION B — Manual download (no account needed):
  1. Go to: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks
  2. Click Download (free, requires Kaggle login)
  3. Unzip into data/kaggle/. You need:
       data/kaggle/sp500_companies.csv
       data/kaggle/sp500_stocks.csv
  4. Then run:
       python -m finsight360.cli ingest-kaggle
""")
        return False

    print(f"Downloading {DATASET} to {DATA_DIR}/ ...")
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET,
            "-p", str(DATA_DIR),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Download failed:\n{result.stderr}")
        return False

    print("Download complete.")
    csv_files = list(DATA_DIR.glob("*.csv"))
    print(f"Files available: {[f.name for f in csv_files]}")

    required = ["sp500_companies.csv", "sp500_stocks.csv"]
    missing = [r for r in required if not (DATA_DIR / r).exists()]
    if missing:
        print(f"\nWarning — expected files not found: {missing}")
        print("Files present:", [f.name for f in csv_files])
        return False

    return True


if __name__ == "__main__":
    success = download()
    if success:
        print("\nNext step: python -m finsight360.cli ingest-kaggle")
    sys.exit(0 if success else 1)
