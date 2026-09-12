import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Gold Strategy — Time Machine",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS / CONSTANTS
# ============================================================

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
EVENTS_FILE = DATA_DIR / "gold_research_events.csv"

APP_TITLE = "🟡 Gold Strategy — Time Machine"

RESEARCH_CONFIG = {
    "Market": "Gold Futures (GC=F)",
    "Direction": "Long only",
    "Breakout": "50-day Donchian",
    "ADX": "≥ 15",
    "Initial SL": "2.0 ATR",
    "Trailing SL": "1.5 ATR",
    "Max hold": "15 bars",
    "Milestone": "+3%",
    "Target type": "Milestone, not forced TP",
}


# ============================================================
# CLEAN TRADINGVIEW-STYLE CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .block-container {
        padding-top: 0.75rem;
        padding-bottom: 1.5rem;
        max-width: 1700px;
    }

    .stApp {
        background: #ffffff;
    }

    h1 {
        letter-spacing: -0.035em;
        font-weight: 750 !important;
        margin-bottom: 0.05rem !important;
    }

    h2, h3 {
        letter-spacing: -0.025em;
    }


    /* ======================================================
       TOP HEADER
       ====================================================== */

    .tm-header {
        display: flex;
        align-items: baseline;
        gap: 0.65rem;
        flex-wrap: wrap;
        margin-bottom: 0.15rem;
    }

    .tm-symbol {
        font-size: 1.45rem;
        font-weight: 800;
        color: #111827;
        letter-spacing: -0.025em;
    }

    .tm-market {
        font-size: 0.78rem;
        color: #667085;
        font-weight: 600;
    }

    .tm-subtitle {
        color: #667085;
        font-size: 0.82rem;
        margin-bottom: 0.65rem;
    }


    /* ======================================================
       STRATEGY BADGES
       ====================================================== */

    .tm-badge-row {
        display: flex;
        gap: 0.35rem;
        flex-wrap: wrap;
        margin: 0.25rem 0 0.75rem 0;
    }

    .tm-badge {
        padding: 0.22rem 0.52rem;
        border-radius: 6px;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        color: #475467;
        font-size: 0.68rem;
        line-height: 1;
        font-weight: 650;
    }

    .tm-badge.gold {
        background: #fff9e6;
        border-color: #ead38a;
        color: #785b00;
    }


    /* ======================================================
       KPI CARDS
       ====================================================== */

    .metric-card {
        border: 1px solid #eaecf0;
        border-radius: 8px;
        background: #ffffff;
        padding: 0.55rem 0.65rem;
        min-height: 68px;
        box-shadow: none;
    }

    .metric-label {
        color: #667085;
        font-size: 0.64rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.045em;
    }

    .metric-value {
        color: #101828;
        font-size: 1.05rem;
        line-height: 1.2;
        font-weight: 750;
        margin-top: 0.18rem;
    }

    .metric-positive {
        color: #07883b;
    }

    .metric-negative {
        color: #c62828;
    }

    .metric-muted {
        color: #344054;
    }


    /* ======================================================
       CHART HEADER
       ====================================================== */

    .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 1rem;
        margin-top: 0.8rem;
        margin-bottom: 0.15rem;
        flex-wrap: wrap;
    }

    .chart-header-left {
        display: flex;
        flex-direction: column;
        gap: 0.05rem;
    }

    .chart-title {
        font-size: 0.94rem;
        font-weight: 750;
        color: #111827;
    }

    .chart-note {
        color: #667085;
        font-size: 0.70rem;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background: #fafbfc;
        border-right: 1px solid #eaecf0;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }


    /* ======================================================
       TABLE
       ====================================================== */

    .section-note {
        color: #667085;
        font-size: 0.76rem;
        margin-top: -0.2rem;
        margin-bottom: 0.5rem;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .tm-footer {
        color: #98a2b3;
        font-size: 0.68rem;
        text-align: center;
        margin-top: 0.9rem;
    }


    /* ======================================================
       MOBILE
       ====================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 0.65rem;
            padding-right: 0.65rem;
        }

        .tm-symbol {
            font-size: 1.25rem;
        }

        .metric-card {
            min-height: 62px;
            padding: 0.45rem 0.5rem;
        }

        .metric-value {
            font-size: 0.95rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATETIME UTILITIES
# ============================================================

def normalize_datetime(value):
    """
    Convert datetime-like values into timezone-naive
    pandas timestamps / Series / DatetimeIndex.
    """

    result = pd.to_datetime(
        value,
        errors="coerce",
    )

    if isinstance(result, pd.Series):

        if isinstance(result.dtype, pd.DatetimeTZDtype):
            return result.dt.tz_localize(None)

        if result.dtype == "object":

            try:

                result = pd.to_datetime(
                    result,
                    errors="coerce",
                    utc=True,
                )

                return result.dt.tz_localize(None)

            except Exception:

                return result

        return result

    if isinstance(result, pd.DatetimeIndex):

        if result.tz is not None:
            return result.tz_localize(None)

        return result

    try:

        if getattr(result, "tzinfo", None) is not None:
            return result.tz_localize(None)

    except Exception:

        pass

    return result


def safe_timestamp(value):
    """
    Always return a timezone-naive pandas Timestamp.
    """

    ts = pd.Timestamp(value)

    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    return ts


def fmt_date(value):

    if pd.isna(value):
        return "—"

    return safe_timestamp(value).strftime("%d %b %Y")


def pct(value):

    try:
        return f"{float(value) * 100:.2f}%"

    except Exception:
        return "—"


def price(value):

    try:
        return f"{float(value):,.2f}"

    except Exception:
        return "—"


def bool_value(value):

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    if isinstance(value, str):

        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    return bool(value)


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(ttl=3600)
def load_daily():

    from data import fetch_daily

    return fetch_daily()


@st.cache_data(ttl=3600)
def load_hourly():

    from data import fetch_hourly

    return fetch_hourly()


@st.cache_data
def load_events():

    if not EVENTS_FILE.exists():

        raise FileNotFoundError(
            "Missing research event file: "
            f"{EVENTS_FILE}"
        )

    df = pd.read_csv(EVENTS_FILE)

    required = [
        "signal_date",
        "entry_date",
        "exit_date",
        "side",
        "entry",
        "exit",
        "target_hit",
        "exit_reason",
        "net_return",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "gold_research_events.csv is missing: "
            + ", ".join(missing)
        )

    for col in [
        "signal_date",
        "entry_date",
        "exit_date",
    ]:

        df[col] = normalize_datetime(df[col])

    return df


# ============================================================
# LOAD / DIAGNOSTICS
# ============================================================

try:

    daily = load_daily().copy()

    hourly = load_hourly().copy()

    events = load_events().copy()

except Exception as exc:

    st.error(
        "The dashboard could not load its data."
    )

    with st.expander(
        "Technical diagnostic",
        expanded=True,
    ):

        st.code(repr(exc))

        st.caption(
            "Check the Streamlit logs if this persists."
        )

    st.stop()


# ============================================================
# NORMALIZE PRICE DATA
# ============================================================

daily.index = normalize_datetime(daily.index)

hourly.index = normalize_datetime(hourly.index)

daily = daily[
    ~daily.index.isna()
].sort_index()

hourly = hourly[
    ~hourly.index.isna()
].sort_index()


# ============================================================
# NORMALIZE EVENT DATA
# ============================================================

for col in [
    "signal_date",
    "entry_date",
    "exit_date",
]:

    events[col] = normalize_datetime(
        events[col]
    )


events = events.dropna(
    subset=[
        "signal_date",
        "entry_date",
        "exit_date",
    ]
).copy()


events["side"] = (
    events["side"]
    .astype(str)
    .str.strip()
    .str.lower()
)


events["exit_reason"] = (
    events["exit_reason"]
    .astype(str)
    .str.strip()
    .str.upper()
)


events["entry"] = pd.to_numeric(
    events["entry"],
    errors="coerce",
)

events["exit"] = pd.to_numeric(
    events["exit"],
    errors="coerce",
)

events["net_return"] = pd.to_numeric(
    events["net_return"],
    errors="coerce",
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="tm-header">
        <span class="tm-symbol">GOLD · GC=F</span>
        <span class="tm-market">Gold Futures · USD</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="tm-subtitle">
        Historical signal replay · BUY → SL / TSL / TIME
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RESEARCH CONFIGURATION BADGES
# ============================================================

badges = [
    ("50D Donchian", True),
    ("ADX ≥ 15", False),
    ("2 ATR SL", False),
    ("1.5 ATR TSL", False),
    ("15-bar max hold", False),
    ("+3% milestone", True),
    ("Long only", False),
]

badge_html = "".join(
    f'<span class="tm-badge {"gold" if gold else ""}">'
    f'{label}</span>'
    for label, gold in badges
)

st.markdown(
    f'<div class="tm-badge-row">{badge_html}</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR / TIME MACHINE
# ============================================================

with st.sidebar:

    st.markdown("### ⏳ Time Machine")

    timeframe = st.radio(
        "Candle timeframe",
        options=[
            "Daily",
            "Hourly",
        ],
        horizontal=True,
        index=0,
        help=(
            "Daily is the research timeframe. "
            "Hourly is a zoomed historical view."
        ),
    )


# ============================================================
# ACTIVE PRICE DATAFRAME
# ============================================================

if timeframe == "Daily":

    prices = daily

else:

    prices = hourly


if prices.empty:

    st.error(
        f"No {timeframe.lower()} price data is available."
    )

    st.stop()


# ============================================================
# AVAILABLE DATE RANGE
# ============================================================

data_start = safe_timestamp(
    prices.index.min()
)

data_end = safe_timestamp(
    prices.index.max()
)


# ============================================================
# QUICK RANGE
# ============================================================

quick_ranges = {
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "2Y": pd.DateOffset(years=2),
    "5Y": pd.DateOffset(years=5),
}


with st.sidebar:

    st.markdown("#### Range")

    quick_range = st.selectbox(
        "Preset",
        [
            "1M",
            "3M",
            "6M",
            "1Y",
            "2Y",
            "5Y",
            "All",
            "Custom",
        ],
        index=3,
    )


# ============================================================
# DEFAULT RANGE
# ============================================================

if quick_range == "All":

    default_start = data_start

elif quick_range == "Custom":

    default_start = max(
        data_start,
        data_end - pd.DateOffset(years=1),
    )

else:

    default_start = max(
        data_start,
        data_end - quick_ranges[quick_range],
    )


# ============================================================
# DATE RANGE
# ============================================================

with st.sidebar:

    selected_range = st.date_input(
        "Historical chart range",
        value=(
            default_start.date(),
            data_end.date(),
        ),
        min_value=data_start.date(),
        max_value=data_end.date(),
        format="DD/MM/YYYY",
        help=(
            "Actual historical window shown "
            "on the Time Machine chart."
        ),
    )


if isinstance(
    selected_range,
    (tuple, list),
) and len(selected_range) == 2:

    start_date = selected_range[0]
    end_date = selected_range[1]

else:

    start_date = selected_range
    end_date = selected_range


start_ts = safe_timestamp(start_date)

end_ts = (
    safe_timestamp(end_date)
    + pd.Timedelta(days=1)
)


# ============================================================
# SIGNAL CONTROLS
# ============================================================

with st.sidebar:

    st.markdown("#### Signals")

    col_a, col_b = st.columns(2)

    with col_a:

        show_buy = st.checkbox(
            "🟢 BUY",
            value=True,
        )

        show_tsl = st.checkbox(
            "🟠 TSL",
            value=True,
        )

    with col_b:

        show_sl = st.checkbox(
            "🔴 SL",
            value=True,
        )

        show_time = st.checkbox(
            "⚪ TIME",
            value=False,
        )


# ============================================================
# CHART LAYERS
# ============================================================

with st.sidebar:

    st.markdown("#### Chart layers")

    show_trade_links = st.checkbox(
        "Connect trade entry → exit",
        value=False,
        help=(
            "Shows subtle lines between completed "
            "trade entries and exits."
        ),
    )

    show_volume = st.checkbox(
        "Show volume",
        value=False,
    )

    st.divider()

    if st.button(
        "↻ Refresh market data",
        use_container_width=True,
    ):

        load_daily.clear()
        load_hourly.clear()

        st.rerun()


# ============================================================
# VALIDATE RANGE
# ============================================================

if start_ts >= end_ts:

    st.error(
        "The start date must be before the end date."
    )

    st.stop()


# ============================================================
# FILTER PRICE DATA
# ============================================================

visible = prices.loc[
    (prices.index >= start_ts)
    &
    (prices.index < end_ts)
].copy()


if visible.empty:

    st.warning(
        "No candles exist in the selected historical range."
    )

    st.stop()


# ============================================================
# FILTER RESEARCH TRADES
# ============================================================

period_events = events[
    (events["entry_date"] >= start_ts)
    &
    (events["entry_date"] < end_ts)
].copy()


# ============================================================
# KPI CALCULATIONS
# ============================================================

trade_count = len(period_events)

positive_trades = (
    period_events["net_return"] > 0
).sum()

win_rate = (
    positive_trades / trade_count
    if trade_count
    else np.nan
)

avg_return = (
    period_events["net_return"].mean()
    if trade_count
    else np.nan
)

median_return = (
    period_events["net_return"].median()
    if trade_count
    else np.nan
)

target_hits = (
    period_events["target_hit"]
    .map(bool_value)
    .sum()
    if trade_count
    else 0
)

target_hit_rate = (
    target_hits / trade_count
    if trade_count
    else np.nan
)


# ============================================================
# KPI STRIP
# ============================================================

st.markdown("### Selected period")

k1, k2, k3, k4, k5, k6 = st.columns(6)


def metric_html(
    label,
    value,
    css_class="metric-muted",
):

    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value}</div>
    </div>
    """


with k1:

    st.markdown(
        metric_html(
            "Timeframe",
            timeframe,
        ),
        unsafe_allow_html=True,
    )


with k2:

    st.markdown(
        metric_html(
            "Candles",
            f"{len(visible):,}",
        ),
        unsafe_allow_html=True,
    )


with k3:

    st.markdown(
        metric_html(
            "Trades",
            f"{trade_count}",
        ),
        unsafe_allow_html=True,
    )


with k4:

    color_class = (
        "metric-positive"
        if np.isfinite(win_rate)
        and win_rate >= 0.5
        else "metric-negative"
    )

    value = (
        f"{win_rate * 100:.1f}%"
        if np.isfinite(win_rate)
        else "—"
    )

    st.markdown(
        metric_html(
            "Positive trades",
            value,
            color_class,
        ),
        unsafe_allow_html=True,
    )


with k5:

    color_class = (
        "metric-positive"
        if np.isfinite(avg_return)
        and avg_return >= 0
        else "metric-negative"
    )

    value = (
        f"{avg_return * 100:+.2f}%"
        if np.isfinite(avg_return)
        else "—"
    )

    st.markdown(
        metric_html(
            "Avg net / trade",
            value,
            color_class,
        ),
        unsafe_allow_html=True,
    )


with k6:

    value = (
        f"{target_hit_rate * 100:.1f}%"
        if np.isfinite(target_hit_rate)
        else "—"
    )

    st.
