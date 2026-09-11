from pathlib import Path
import json
import time
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

SYMBOL = "GC=F"
START_DATE = "2018-01-01"
END_DATE = None
INTERVAL = "1h"
CHUNK_DAYS = 59
SLEEP_SECONDS = 1.0
MAX_RETRIES = 4

DATA_DIR = Path("data_usd")
CACHE_DIR = DATA_DIR / "cache"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

FINAL_FILE = DATA_DIR / "gold_usd_1h.csv"
MANIFEST_FILE = DATA_DIR / "gold_usd_1h_manifest.json"


def utc_timestamp(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def chunk_key(start, end):
    return f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"


def clean_yahoo_frame(frame):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(column[0]).lower() for column in frame.columns]
    else:
        frame.columns = [str(column).lower() for column in frame.columns]

    frame = frame.rename(columns={"adj close": "close"})
    required = ["open", "high", "low", "close", "volume"]
    available = [column for column in required if column in frame.columns]
    frame = frame[available].copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    for column in available:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.dropna(subset=["open", "high", "low", "close"])


def download_chunk(start, end):
    key = chunk_key(start, end)
    cache_file = CACHE_DIR / f"{key}.csv"

    if cache_file.exists() and cache_file.stat().st_size > 100:
        cached = pd.read_csv(cache_file, parse_dates=["timestamp_utc"])
        cached = cached.set_index("timestamp_utc")
        cached.index = pd.to_datetime(cached.index, utc=True)
        return cached

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Downloading {start.date()} to {end.date()} — attempt {attempt}")
            raw = yf.download(
                SYMBOL,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=INTERVAL,
                auto_adjust=False,
                prepost=False,
                progress=False,
                threads=False,
            )
            cleaned = clean_yahoo_frame(raw)
            cleaned.to_csv(cache_file, index_label="timestamp_utc")
            return cleaned
        except Exception as error:
            print(f"Download failed: {error}")
            if attempt < MAX_RETRIES:
                time.sleep(SLEEP_SECONDS * attempt)

    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def main():
    start = utc_timestamp(START_DATE)
    end = utc_timestamp(END_DATE) if END_DATE else pd.Timestamp.now(tz="UTC")

    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=CHUNK_DAYS), end)
        chunk = download_chunk(cursor, chunk_end)
        if not chunk.empty:
            chunks.append(chunk)
        cursor = chunk_end
        time.sleep(SLEEP_SECONDS)

    if chunks:
        data = pd.concat(chunks)
        data = data[~data.index.duplicated(keep="last")].sort_index()
    else:
        data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    data.to_csv(FINAL_FILE, index_label="timestamp_utc")

    manifest = {
        "symbol": SYMBOL,
        "currency": "USD",
        "requested_start_utc": start.isoformat(),
        "requested_end_utc": end.isoformat(),
        "interval": INTERVAL,
        "chunk_days": CHUNK_DAYS,
        "provider": "Yahoo Finance via yfinance",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(data)),
        "actual_start_utc": data.index.min().isoformat() if len(data) else None,
        "actual_end_utc": data.index.max().isoformat() if len(data) else None,
        "file": str(FINAL_FILE),
        "cache_directory": str(CACHE_DIR),
        "note": "GC=F is a continuous futures proxy; verify hourly coverage before backtesting.",
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nCompleted")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
