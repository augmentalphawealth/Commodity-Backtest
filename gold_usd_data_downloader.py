from pathlib import Path
import json
import time
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

SYMBOL = "GC=F"
HOURLY_LOOKBACK_DAYS = 720
HOURLY_CHUNK_DAYS = 30
DAILY_START = "2018-01-01"
DAILY_CHUNK_DAYS = 365
SLEEP_SECONDS = 2.0
MAX_RETRIES = 5

DATA_DIR = Path("data_usd")
HOURLY_CACHE = DATA_DIR / "cache_hourly"
DAILY_CACHE = DATA_DIR / "cache_daily"
DATA_DIR.mkdir(exist_ok=True)
HOURLY_CACHE.mkdir(exist_ok=True)
DAILY_CACHE.mkdir(exist_ok=True)

HOURLY_FILE = DATA_DIR / "gold_usd_1h.csv"
DAILY_FILE = DATA_DIR / "gold_usd_1d.csv"
MANIFEST_FILE = DATA_DIR / "gold_usd_manifest.json"


def clean_frame(frame):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(c[0]).lower() for c in frame.columns]
    else:
        frame.columns = [str(c).lower() for c in frame.columns]
    frame = frame.rename(columns={"adj close": "close"})
    frame = frame[[c for c in ["open", "high", "low", "close", "volume"] if c in frame.columns]].copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame[~frame.index.duplicated()].sort_index()
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["open", "high", "low", "close"])


def get_chunk(start, end, interval, cache_dir):
    cache = cache_dir / f"{start:%Y%m%d}_{end:%Y%m%d}_{interval}.csv"
    if cache.exists() and cache.stat().st_size > 100:
        out = pd.read_csv(cache, parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
        out.index = pd.to_datetime(out.index, utc=True)
        return out
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"{interval}: {start:%Y-%m-%d} to {end:%Y-%m-%d}, attempt {attempt}")
            raw = yf.download(SYMBOL, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval=interval, auto_adjust=False, prepost=False, progress=False, threads=False)
            cleaned = clean_frame(raw)
            if not cleaned.empty:
                cleaned.to_csv(cache, index_label="timestamp_utc")
            return cleaned
        except Exception as exc:
            print(f"Download error: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(SLEEP_SECONDS * attempt)
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def download_range(start, end, interval, chunk_days, cache_dir):
    pieces = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end)
        piece = get_chunk(cursor, chunk_end, interval, cache_dir)
        if not piece.empty:
            pieces.append(piece)
        cursor = chunk_end
        time.sleep(SLEEP_SECONDS)
    if not pieces:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(pieces).loc[lambda x: ~x.index.duplicated()].sort_index()


def main():
    now = pd.Timestamp.now(tz="UTC").floor("h")
    hourly_start = now - pd.Timedelta(days=HOURLY_LOOKBACK_DAYS)
    daily_start = pd.Timestamp(DAILY_START, tz="UTC")

    hourly = download_range(hourly_start, now, "1h", HOURLY_CHUNK_DAYS, HOURLY_CACHE)
    daily = download_range(daily_start, now, "1d", DAILY_CHUNK_DAYS, DAILY_CACHE)

    hourly.to_csv(HOURLY_FILE, index_label="timestamp_utc")
    daily.to_csv(DAILY_FILE, index_label="timestamp_utc")

    manifest = {
        "symbol": SYMBOL,
        "provider": "Yahoo Finance via yfinance",
        "currency": "USD",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "hourly": {
            "requested_start_utc": hourly_start.isoformat(),
            "requested_end_utc": now.isoformat(),
            "rows": int(len(hourly)),
            "actual_start_utc": hourly.index.min().isoformat() if len(hourly) else None,
            "actual_end_utc": hourly.index.max().isoformat() if len(hourly) else None,
            "file": str(HOURLY_FILE),
        },
        "daily": {
            "requested_start_utc": daily_start.isoformat(),
            "requested_end_utc": now.isoformat(),
            "rows": int(len(daily)),
            "actual_start_utc": daily.index.min().isoformat() if len(daily) else None,
            "actual_end_utc": daily.index.max().isoformat() if len(daily) else None,
            "file": str(DAILY_FILE),
        },
        "note": "GC=F is a continuous futures proxy; hourly history is limited by Yahoo, while daily history is used for the long sample.",
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
