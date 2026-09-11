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
SLEEP_SECONDS = 2
MAX_RETRIES = 5

DATA_DIR = Path("data_usd")
CACHE_DIR = DATA_DIR / "cache_hourly"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "gold_usd_1h.csv"
MANIFEST_FILE = DATA_DIR / "gold_usd_1h_manifest.json"


def empty_frame():
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"]
    )


def flatten_columns(columns):
    result = []

    if not isinstance(columns, pd.MultiIndex):
        return [str(column).lower() for column in columns]

    for column in columns:
        parts = [
            str(part)
            for part in column
            if str(part) not in ("", "None")
        ]
        result.append(parts[0].lower() if parts else "")

    return result


def clean_frame(frame):
    if frame is None or frame.empty:
        return empty_frame()

    frame = frame.copy()
    frame.columns = flatten_columns(frame.columns)

    if "adj close" in frame.columns and "close" not in frame.columns:
        frame = frame.rename(columns={"adj close": "close"})

    wanted = ["open", "high", "low", "close", "volume"]
    available = [column for column in wanted if column in frame.columns]
    frame = frame[available].copy()

    frame.index = pd.to_datetime(
        frame.index,
        utc=True,
        errors="coerce",
    )
    frame = frame[~frame.index.isna()]
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.sort_index()

    for column in frame.columns:
        values = frame[column]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]
        frame[column] = pd.to_numeric(values, errors="coerce")

    return frame.dropna(
        subset=["open", "high", "low", "close"]
    )


def get_chunk(start, end):
    cache_file = CACHE_DIR / (
        f"{start:%Y%m%d}_{end:%Y%m%d}_1h.csv"
    )

    if cache_file.exists() and cache_file.stat().st_size > 100:
        cached = pd.read_csv(
            cache_file,
            parse_dates=["timestamp_utc"],
        )
        cached = cached.set_index("timestamp_utc")
        cached.index = pd.to_datetime(
            cached.index,
            utc=True,
        )
        return cached

    for attempt in range(1, MAX_RETRIES + 1):
        print(
            f"1h: {start:%Y-%m-%d} to {end:%Y-%m-%d}, "
            f"attempt {attempt}"
        )

        try:
            raw = yf.Ticker(SYMBOL).history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=INTERVAL,
                auto_adjust=False,
                actions=False,
                repair=False,
            )

            cleaned = clean_frame(raw)

            if not cleaned.empty:
                cleaned.to_csv(
                    cache_file,
                    index_label="timestamp_utc",
                )

            return cleaned

        except Exception as error:
            print(f"Download error: {error}")
            if attempt < MAX_RETRIES:
                time.sleep(SLEEP_SECONDS * attempt)

    return empty_frame()


def main():
    end = pd.Timestamp.now(tz="UTC").floor("h")
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)

    parts = []
    cursor = start

    while cursor < end:
        chunk_end = min(
            cursor + pd.Timedelta(days=CHUNK_DAYS),
            end,
        )

        part = get_chunk(cursor, chunk_end)
        if not part.empty:
            parts.append(part)

        cursor = chunk_end
        time.sleep(SLEEP_SECONDS)

    if parts:
        data = pd.concat(parts)
        data = data[~data.index.duplicated(keep="last")]
        data = data.sort_index()
    else:
        data = empty_frame()

    data.to_csv(
        OUTPUT_FILE,
        index_label="timestamp_utc",
    )

    manifest = {
        "symbol": SYMBOL,
        "currency": "USD",
        "interval": INTERVAL,
        "provider": "Yahoo Finance via yfinance",
        "requested_start_utc": start.isoformat(),
        "requested_end_utc": end.isoformat(),
        "rows": int(len(data)),
        "actual_start_utc": (
            data.index.min().isoformat()
            if len(data)
            else None
        ),
        "actual_end_utc": (
            data.index.max().isoformat()
            if len(data)
            else None
        ),
        "file": str(OUTPUT_FILE),
        "cache_directory": str(CACHE_DIR),
        "note": (
            "Yahoo hourly data is limited to the recent period."
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
