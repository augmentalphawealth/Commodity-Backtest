import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]/'src'))
from data import fetch_daily, fetch_hourly
fetch_daily(); fetch_hourly()
print('Gold data updated.')
