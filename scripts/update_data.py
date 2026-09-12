import sys
from pathlib import Path

# Add repository root to Python path.
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import fetch_daily, fetch_hourly


def main():
    print("Starting Gold market data update...")

    fetch_daily()
    fetch_hourly()

    print("Gold data updated successfully.")


if __name__ == "__main__":
    main()
