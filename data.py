from pathlib import Path
import pandas as pd
import yfinance as yf

# Always resolve paths from this file, not from Streamlit's working directory.
# This prevents FileNotFoundError when Community Cloud runs the app from repo root.
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(parents=True, exist_ok=True)

DAILY_FILE = DATA / "gold_daily.csv"
HOURLY_FILE = DATA / "gold_hourly.csv"
HOURLY_SEED = DATA / "gold_hourly_seed.csv"

TICKER = "GC=F"


def _normalise_yahoo(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Yahoo can return MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df.columns = [str(c).strip().title() for c in df.columns]

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            if c == "Volume":
                df[c] = 0
            else:
                raise ValueError(f"Yahoo response is missing {c}")

    df.index = pd.to_datetime(df.index, errors="coerce")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df = df[needed].copy()
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def _read_seed(path):
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "timestamp_utc" in df.columns:
        idx = pd.to_datetime(df.pop("timestamp_utc"), errors="coerce", utc=True)
        idx = idx.dt.tz_localize(None)
        df.index = idx
    elif "Datetime" in df.columns:
        idx = pd.to_datetime(df.pop("Datetime"), errors="coerce")
        df.index = idx
    elif "Date" in df.columns:
        idx = pd.to_datetime(df.pop("Date"), errors="coerce")
        df.index = idx
    else:
        # Last-resort support for an index saved by pandas.
        first = df.columns[0]
        idx = pd.to_datetime(df.pop(first), errors="coerce")
        df.index = idx

    return _normalise_yahoo(df)


def fetch_daily():
    """Return ~10 years of daily GC=F data.

    If the GitHub CSV exists, merge fresh Yahoo data into it.
    If the CSV was accidentally omitted from GitHub, download a fresh copy
    instead of crashing with FileNotFoundError.
    """
    seed = _read_seed(DAILY_FILE)

    fresh = _normalise_yahoo(
        yf.download(
            TICKER,
            period="10y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    )

    if fresh.empty and seed.empty:
        raise RuntimeError("Yahoo Finance returned no daily GC=F data.")

    df = pd.concat([seed, fresh])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(DAILY_FILE, index_label="timestamp_utc")
    return df


def fetch_hourly():
    """Return the available recent hourly GC=F history.

    Yahoo intraday history is limited, so use the seed when present and merge
    the latest rolling Yahoo window into it.
    """
    seed = _read_seed(HOURLY_FILE)
    if seed.empty:
        seed = _read_seed(HOURLY_SEED)

    fresh = _normalise_yahoo(
        yf.download(
            TICKER,
            period="729d",
            interval="1h",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    )

    if fresh.empty and seed.empty:
        raise RuntimeError("Yahoo Finance returned no hourly GC=F data.")

    df = pd.concat([seed, fresh])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Keep the latest ~2 years. This matches the dashboard's intended
    # hourly research window and prevents the GitHub CSV growing forever.
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)
    df = df[df.index >= cutoff]

    df.to_csv(HOURLY_FILE, index_label="timestamp_utc")
    return df
