from pathlib import Path
import json
import time
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

SYMBOL = "GC=F"
INTERVAL = "1h"
LOOKBACK_DAYS = 720
CHUNK_DAYS = 30
SLEEP_SECONDS = 2.0
MAX_RETRIES = 5

DATA_DIR = Path("data_usd")
CACHE_DIR = DATA_DIR / "cache"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

FINAL_FILE = DATA_DIR / "gold_usd_1h.csv"
MANIFEST_FILE = DATA_DIR / "gold_usd_1h_manifest.json"


def clean_frame(frame):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(c[0]).lower() for c in frame.columns]
    else:
        frame.columns = [str(c).lower() for c in frame.columns]

    frame = frame.rename(columns={"adj close": "close"})
    wanted = ["open", "high", "low", "close", "volume"]
    frame = frame[[c for c in wanted if c in frame.columns]].copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame[~frame.index.duplicated()].sort_index()

    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return frame.dropna(subset=["open", "high", "low", "close"])


def get_chunk(start, end):
    cache_file = CACHE_DIR / f"{start:%Y%m%d}_{end:%Y%m%d}.csv"

    if cache_file.exists() and cache_file.stat().st_size > 100:
        cached = pd.read_csv(cache_file, parse_dates=["timestamp_utc"])
        cached = cached.set_index("timestamp_utc")
        cached.index = pd.to_datetime(cached.index, utc=True)
        return cached

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Downloading {start:%Y-%m-%d} to {end:%Y-%m-%d}, attempt {attempt}")
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
            cleaned = clean_frame(raw)
            if not cleaned.empty:
                cleaned.to_csv(cache_file, index_label="timestamp_utc")
            return cleaned
        except Exception as exc:
            print(f"Chunk error: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(SLEEP_SECONDS * attempt)

    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def main():
    end = pd.Timestamp.now(tz="UTC").floor("h")
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    chunks = []
    cursor = start

    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=CHUNK_DAYS), end)
        chunk = get_chunk(cursor, chunk_end)
        if not chunk.empty:
            chunks.append(chunk)
        cursor = chunk_end
        time.sleep(SLEEP_SECONDS)

    if chunks:
        data = pd.concat(chunks)
        data = data[~data.index.duplicated()].sort_index()
    else:
        data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    data.to_csv(FINAL_FILE, index_label="timestamp_utc")

    manifest = {
        "symbol": SYMBOL,
        "currency": "USD",
        "interval": INTERVAL,
        "provider": "Yahoo Finance via yfinance",
        "requested_start_utc": start.isoformat(),
        "requested_end_utc": end.isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "chunk_days": CHUNK_DAYS,
        "rows": int(len(data)),
        "actual_start_utc": data.index.min().isoformat() if len(data) else None,
        "actual_end_utc": data.index.max().isoformat() if len(data) else None,
        "file": str(FINAL_FILE),
        "cache_directory": str(CACHE_DIR),
        "note": "GC=F hourly history is limited by Yahoo; this is not a 2018-present hourly dataset.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
