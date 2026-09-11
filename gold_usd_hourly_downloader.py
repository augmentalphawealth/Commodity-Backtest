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

SLEEP_SECONDS = 2
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


def empty_frame():
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"]
    )


def clean_frame(frame):
    if frame is None or frame.empty:
        return empty_frame()

    frame = frame.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            str(column[-1]).lower()
            for column in frame.columns
        ]
    else:
        frame.columns = [
            str(column).lower()
            for column in frame.columns
        ]

    frame = frame.rename(
        columns={
            "adj close": "close",
            "datetime": "timestamp_utc",
        }
    )

    wanted = ["open", "high", "low", "close", "volume"]
    available = [
        column for column in wanted
        if column in frame.columns
    ]

    frame = frame[available].copy()

    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(
            frame.index,
            utc=True,
            errors="coerce",
        )
    else:
        frame.index = pd.to_datetime(
            frame.index,
            utc=True,
        )

    frame = frame[~frame.index.isna()]
    frame = frame[~frame.index.duplicated()]
    frame = frame.sort_index()

    for column in frame.columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    return frame.dropna(
        subset=["open", "high", "low", "close"]
    )


def download_chunk(start, end, interval, cache_dir):
    cache_file = cache_dir / (
        f"{start:%Y%m%d}_{end:%Y%m%d}_{interval}.csv"
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
            f"{interval}: "
            f"{start:%Y-%m-%d} to {end:%Y-%m-%d}, "
            f"attempt {attempt}"
        )

        try:
            ticker = yf.Ticker(SYMBOL)

            raw = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
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


def download_range(
    start,
    end,
    interval,
    chunk_days,
    cache_dir,
):
    pieces = []
    cursor = start

    while cursor < end:
        chunk_end = min(
            cursor + pd.Timedelta(days=chunk_days),
            end,
        )

        piece = download_chunk(
            cursor,
            chunk_end,
            interval,
            cache_dir,
        )

        if not piece.empty:
            pieces.append(piece)

        cursor = chunk_end
        time.sleep(SLEEP_SECONDS)

    if not pieces:
        return empty_frame()

    combined = pd.concat(pieces)

    combined = combined[
        ~combined.index.duplicated()
    ].sort_index()

    return combined


def main():
    now = pd.Timestamp.now(
        tz="UTC"
    ).floor("h")

    hourly_start = (
        now - pd.Timedelta(days=HOURLY_LOOKBACK_DAYS)
    )

    daily_start = pd.Timestamp(
        DAILY_START,
        tz="UTC",
    )

    hourly = download_range(
        hourly_start,
        now,
        "1h",
        HOURLY_CHUNK_DAYS,
        HOURLY_CACHE,
    )

    daily = download_range(
        daily_start,
        now,
        "1d",
        DAILY_CHUNK_DAYS,
        DAILY_CACHE,
    )

    hourly.to_csv(
        HOURLY_FILE,
        index_label="timestamp_utc",
    )

    daily.to_csv(
        DAILY_FILE,
        index_label="timestamp_utc",
    )

    manifest = {
        "symbol": SYMBOL,
        "currency": "USD",
        "provider": "Yahoo Finance via yfinance",
        "downloaded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "hourly": {
            "requested_start_utc": hourly_start.isoformat(),
            "requested_end_utc": now.isoformat(),
            "rows": int(len(hourly)),
            "actual_start_utc": (
                hourly.index.min().isoformat()
                if len(hourly)
                else None
            ),
            "actual_end_utc": (
                hourly.index.max().isoformat()
                if len(hourly)
                else None
            ),
            "file": str(HOURLY_FILE),
        },
        "daily": {
            "requested_start_utc": daily_start.isoformat(),
            "requested_end_utc": now.isoformat(),
            "rows": int(len(daily)),
            "actual_start_utc": (
                daily.index.min().isoformat()
                if len(daily)
                else None
            ),
            "actual_end_utc": (
                daily.index.max().isoformat()
                if len(daily)
                else None
            ),
            "file": str(DAILY_FILE),
        },
    }

    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
