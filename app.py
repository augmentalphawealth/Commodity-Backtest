import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

@st.cache_data(ttl=900)
def load_daily():

    from data import fetch_daily

    return fetch_daily()


@st.cache_data(ttl=300)
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

    if timeframe == "Daily":

        show_donchian = st.checkbox(
            "Show 50-day breakout level",
            value=True,
        )

    else:

        show_donchian = False

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


# Collapse non-trading dates in the daily view so sparse source
# data does not leave visual gaps between adjacent candles.
if timeframe == "Daily":

    visible_days = pd.DatetimeIndex(
        visible.index
    ).normalize().unique()

    calendar_days = pd.date_range(
        visible_days.min(),
        visible_days.max(),
        freq="D",
    )

    xaxis_rangebreaks = [
        dict(
            values=calendar_days.difference(
                visible_days
            )
        )
    ]

else:

    xaxis_rangebreaks = [
        dict(
            bounds=["sat", "mon"]
        )
    ]


if show_donchian:

    donchian_high = (
        prices["High"]
        .rolling(50, min_periods=50)
        .max()
        .shift(1)
        .reindex(visible.index)
    )


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

    st.markdown(
        metric_html(
            "+3% milestone",
            value,
            "metric-positive",
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# CHART HEADER
# ============================================================

st.markdown(
    f"""
    <div class="chart-title-row">
        <div>
            <div class="chart-title">
                {timeframe} historical replay
            </div>
            <div class="chart-note">
                {fmt_date(start_ts)}
                → {fmt_date(end_ts - pd.Timedelta(days=1))}
                &nbsp; • &nbsp;
                {trade_count} research trade(s)
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN CHART
# ============================================================

last_candle = safe_timestamp(
    visible.index.max()
)

checked_at = pd.Timestamp.now().strftime(
    "%d %b %Y %H:%M"
)

st.caption(
    f"Last available {timeframe.lower()} candle: "
    f"{fmt_date(last_candle)} · "
    f"Data checked: {checked_at}"
)

if show_volume:

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.84, 0.16],
    )

else:

    fig = go.Figure()


def add_price_trace(trace):

    if show_volume:

        fig.add_trace(
            trace,
            row=1,
            col=1,
        )

    else:

        fig.add_trace(trace)


# ------------------------------------------------------------
# Price candles
# ------------------------------------------------------------

add_price_trace(
    go.Candlestick(

        x=visible.index,

        open=visible["Open"],

        high=visible["High"],

        low=visible["Low"],

        close=visible["Close"],

        name="Gold",

        increasing=dict(
            line=dict(
                color="#16a34a",
                width=1,
            ),
            fillcolor="#16a34a",
        ),

        decreasing=dict(
            line=dict(
                color="#dc2626",
                width=1,
            ),
            fillcolor="#dc2626",
        ),

        whiskerwidth=0.35,

        hoverlabel=dict(
            bgcolor="white",
            font=dict(
                color="#101828"
            ),
        ),

        hovertemplate=(
            "<b>%{x|%d %b %Y}</b><br>"
            "Open: %{open:,.2f}<br>"
            "High: %{high:,.2f}<br>"
            "Low: %{low:,.2f}<br>"
            "Close: %{close:,.2f}"
            "<extra></extra>"
        ),
    )
)


if show_donchian:

    add_price_trace(
        go.Scatter(
            x=visible.index,
            y=donchian_high,
            mode="lines",
            name="50D breakout",
            line=dict(
                color="#3b82f6",
                width=1.25,
                dash="dot",
            ),
            hovertemplate=(
                "50D breakout: %{y:,.2f}"
                "<extra></extra>"
            ),
        )
    )


# ============================================================
# TRADE LINKS
# ============================================================

if show_trade_links and not period_events.empty:

    for _, row in period_events.iterrows():

        entry_dt = safe_timestamp(
            row["entry_date"]
        )

        exit_dt = safe_timestamp(
            row["exit_date"]
        )

        # ----------------------------------------------------
        # Find closest visible candles.
        # ----------------------------------------------------

        entry_candidates = visible.index[
            visible.index >= entry_dt
        ]

        exit_candidates = visible.index[
            visible.index >= exit_dt
        ]

        if len(entry_candidates):

            entry_x = entry_candidates[0]

        else:

            entry_candidates = visible.index[
                visible.index <= entry_dt
            ]

            if not len(entry_candidates):
                continue

            entry_x = entry_candidates[-1]


        if len(exit_candidates):

            exit_x = exit_candidates[0]

        else:

            exit_candidates = visible.index[
                visible.index <= exit_dt
            ]

            if not len(exit_candidates):
                continue

            exit_x = exit_candidates[-1]


        entry_price = float(
            row["entry"]
        )

        exit_price = float(
            row["exit"]
        )


        # Light line only. Markers carry the meaning.
        add_price_trace(
            go.Scatter(
                x=[
                    entry_x,
                    exit_x,
                ],
                y=[
                    entry_price,
                    exit_price,
                ],
                mode="lines",
                line=dict(
                    color="rgba(71,84,103,0.16)",
                    width=1,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )


# ============================================================
# SIGNAL MARKERS
# ============================================================

marker_defs = [

    {
        "name": "BUY",
        "enabled": show_buy,
        "date_col": "entry_date",
        "symbol": "triangle-up",
        "color": "#16a34a",
        "size": 12,
        "anchor": "below",
    },

    {
        "name": "SL",
        "enabled": show_sl,
        "date_col": "exit_date",
        "symbol": "x",
        "color": "#dc2626",
        "size": 11,
        "anchor": "above",
    },

    {
        "name": "TSL",
        "enabled": show_tsl,
        "date_col": "exit_date",
        "symbol": "triangle-down",
        "color": "#f59e0b",
        "size": 11,
        "anchor": "above",
    },

    {
        "name": "TIME",
        "enabled": show_time,
        "date_col": "exit_date",
        "symbol": "circle-open",
        "color": "#667085",
        "size": 10,
        "anchor": "above",
    },
]


for spec in marker_defs:

    if not spec["enabled"]:
        continue


    name = spec["name"]

    date_col = spec["date_col"]


    # --------------------------------------------------------
    # Filter event subset.
    # --------------------------------------------------------

    if name == "BUY":

        subset = events[
            events["side"].eq("long")
        ].copy()

    else:

        subset = events[
            events["exit_reason"].eq(name)
        ].copy()


    # --------------------------------------------------------
    # Normalize the date column.
    # --------------------------------------------------------

    subset[date_col] = normalize_datetime(
        subset[date_col]
    )


    # --------------------------------------------------------
    # Time Machine range filter.
    # --------------------------------------------------------

    subset = subset[
        (subset[date_col] >= start_ts)
        &
        (subset[date_col] < end_ts)
    ].copy()


    if subset.empty:
        continue


    x_values = []

    y_values = []

    hover_values = []


    for _, row in subset.iterrows():

        event_dt = safe_timestamp(
            row[date_col]
        )


        # ----------------------------------------------------
        # Match event to visible candle.
        #
        # DAILY:
        # exact daily candle when available.
        #
        # HOURLY:
        # first available hourly candle at/after event.
        # ----------------------------------------------------

        candidates = visible.index[
            visible.index >= event_dt
        ]


        if len(candidates):

            x = candidates[0]

        else:

            candidates = visible.index[
                visible.index <= event_dt
            ]

            if not len(candidates):
                continue

            x = candidates[-1]


        candle = visible.loc[x]


        # ----------------------------------------------------
        # Event price.
        # ----------------------------------------------------

        if name == "BUY":

            event_price = float(
                row["entry"]
            )

        else:

            event_price = float(
                row["exit"]
            )


        # ----------------------------------------------------
        # Put markers close to actual event price rather than
        # far away from the candle.
        # ----------------------------------------------------

        candle_range = (
            float(candle["High"])
            - float(candle["Low"])
        )

        if not np.isfinite(
            candle_range
        ) or candle_range <= 0:

            candle_range = max(
                abs(float(candle["Close"]))
                * 0.002,
                1.0,
            )


        if name == "BUY":

            y = float(
                candle["Low"]
            ) - candle_range * 0.35

        else:

            y = float(
                candle["High"]
            ) + candle_range * 0.35


        # ----------------------------------------------------
        # Hover details.
        # ----------------------------------------------------

        if name == "BUY":

            hover = (
                "<b>🟢 BUY</b><br>"
                f"Entry: {fmt_date(row['entry_date'])}"
                f" @ {price(row['entry'])}<br>"
                f"Exit: {fmt_date(row['exit_date'])}"
                f" @ {price(row['exit'])}<br>"
                f"Exit type: {row['exit_reason']}<br>"
                f"Net: {pct(row['net_return'])}<br>"
                f"+3% milestone: "
                f"{'YES' if bool(row['target_hit']) else 'NO'}"
                "<extra></extra>"
            )

        else:

            hover = (
                f"<b>{name}</b><br>"
                f"Exit: {fmt_date(row['exit_date'])}"
                f" @ {price(row['exit'])}<br>"
                f"Entry: {fmt_date(row['entry_date'])}"
                f" @ {price(row['entry'])}<br>"
                f"Net: {pct(row['net_return'])}<br>"
                f"+3% milestone: "
                f"{'YES' if bool(row['target_hit']) else 'NO'}"
                "<extra></extra>"
            )


        x_values.append(x)

        y_values.append(y)

        hover_values.append(hover)


    # --------------------------------------------------------
    # Add marker trace.
    # --------------------------------------------------------

    if x_values:

        add_price_trace(
            go.Scatter(

                x=x_values,

                y=y_values,

                mode="markers",

                name=name,

                marker=dict(

                    symbol=spec["symbol"],

                    size=spec["size"],

                    color=spec["color"],

                    line=dict(
                        color="white",
                        width=1,
                    ),
                ),

                hovertemplate=(
                    hover_values
                ),

                hovertext=hover_values,

                hoverlabel=dict(
                    bgcolor="white",
                    bordercolor="#d0d5dd",
                    font=dict(
                        color="#101828"
                    ),
                ),
            )
        )


# ============================================================
# VOLUME
# ============================================================

if show_volume and "Volume" in visible.columns:

    volume = visible["Volume"].fillna(0)

    # Use a second y-axis so price scale is untouched.

    fig.add_trace(
        go.Bar(
            x=visible.index,
            y=volume,
            name="Volume",
            opacity=0.13,
            marker_line_width=0,
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "Volume: %{y:,.0f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )


# ============================================================
# CHART LAYOUT
# ============================================================

fig.update_layout(

    height=780 if show_volume else 740,

    template="plotly_white",

    margin=dict(
        l=8,
        r=18,
        t=16,
        b=12,
    ),

    paper_bgcolor="white",

    plot_bgcolor="white",

    hovermode="x unified",

    dragmode="pan",

    showlegend=True,

    legend=dict(

        orientation="h",

        x=0,

        y=1.015,

        xanchor="left",

        yanchor="bottom",

        bgcolor="rgba(255,255,255,0.88)",

        bordercolor="rgba(208,213,221,0.65)",

        borderwidth=1,

        font=dict(
            size=11,
            color="#344054",
        ),
    ),

    xaxis=dict(

        title=None,

        rangebreaks=xaxis_rangebreaks,

        tickformat="%b\n%Y",

        ticklabelmode="period",

        showgrid=False,

        rangeslider=dict(
            visible=False,
        ),

        showline=True,

        linecolor="#d0d5dd",

        linewidth=1,

        tickfont=dict(
            size=10,
            color="#667085",
        ),

        showspikes=True,

        spikecolor="#98a2b3",

        spikethickness=1,

        spikedash="dot",

        spikesnap="cursor",

        rangeslider_thickness=0.04,

        fixedrange=False,
    ),

    yaxis=dict(

        title=None,

        side="right",

        showgrid=True,

        gridcolor="rgba(16,24,40,0.055)",

        zeroline=False,

        showline=True,

        linecolor="#d0d5dd",

        linewidth=1,

        tickfont=dict(
            size=10,
            color="#667085",
        ),

        tickformat=",.0f",

        fixedrange=False,
    ),

    font=dict(
        family="Inter, -apple-system, BlinkMacSystemFont, "
               "'Segoe UI', sans-serif",
        color="#101828",
    ),
)


if show_volume:

    fig.update_yaxes(
        title=None,
        showgrid=False,
        showticklabels=False,
        fixedrange=True,
        row=2,
        col=1,
    )


fig.update_xaxes(
    rangebreaks=xaxis_rangebreaks,
    tickformat="%b\n%Y",
    ticklabelmode="period",
    rangeslider_visible=False,
    row=2 if show_volume else 1,
    col=1,
)


# ============================================================
# CHART BUTTONS / TOOLBAR
# ============================================================

config = {

    "displaylogo": False,

    "scrollZoom": True,

    "responsive": True,

    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "autoScale2d",
        "toggleSpikelines",
    ],

    "toImageButtonOptions": {
        "format": "png",
        "filename": (
            "gold_strategy_"
            f"{timeframe.lower()}_"
            f"{start_ts.strftime('%Y%m%d')}_"
            f"{(end_ts - pd.Timedelta(days=1)).strftime('%Y%m%d')}"
        ),
        "height": 1000,
        "width": 1800,
        "scale": 2,
    },
}


# ============================================================
# RENDER CHART
# ============================================================

st.plotly_chart(
    fig,
    use_container_width=True,
    config=config,
    key="gold_time_machine_chart",
)


# ============================================================
# RESEARCH CONFIGURATION
# ============================================================

with st.expander(
    "Strategy configuration",
    expanded=False,
):

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            "**Market**"
        )

        st.write(
            RESEARCH_CONFIG["Market"]
        )

        st.markdown(
            "**Direction**"
        )

        st.write(
            RESEARCH_CONFIG["Direction"]
        )

    with c2:

        st.markdown(
            "**Breakout**"
        )

        st.write(
            RESEARCH_CONFIG["Breakout"]
        )

        st.markdown(
            "**ADX**"
        )

        st.write(
            RESEARCH_CONFIG["ADX"]
        )

    with c3:

        st.markdown(
            "**Initial stop**"
        )

        st.write(
            RESEARCH_CONFIG["Initial SL"]
        )

        st.markdown(
            "**Trailing stop**"
        )

        st.write(
            RESEARCH_CONFIG["Trailing SL"]
        )

    with c4:

        st.markdown(
            "**Max hold**"
        )

        st.write(
            RESEARCH_CONFIG["Max hold"]
        )

        st.markdown(
            "**Target**"
        )

        st.write(
            RESEARCH_CONFIG["Milestone"]
        )

    st.caption(
        "The +3% level is a research milestone, "
        "not a forced take-profit."
    )


# ============================================================
# TRADE TABLE
# ============================================================

st.markdown(
    "### Historical trades"
)

st.markdown(
    f"""
    <div class="section-note">
        Trades whose entry falls inside
        {fmt_date(start_ts)}
        → {fmt_date(end_ts - pd.Timedelta(days=1))}.
    </div>
    """,
    unsafe_allow_html=True,
)


if period_events.empty:

    st.info(
        "No research trades occurred in the selected period."
    )

else:

    table = period_events.copy()

    table["BUY"] = table["entry_date"].map(
        fmt_date
    )

    table["ENTRY"] = table["entry"].map(
        price
    )

    table["EXIT DATE"] = table["exit_date"].map(
        fmt_date
    )

    table["EXIT"] = table["exit"].map(
        price
    )

    table["EXIT TYPE"] = table[
        "exit_reason"
    ]

    table["NET"] = table[
        "net_return"
    ].map(
        pct
    )

    table["+3%"] = table[
        "target_hit"
    ].map(
        lambda x:
            "YES"
            if bool(x)
            else "NO"
    )

    table = table[
        [
            "BUY",
            "ENTRY",
            "EXIT DATE",
            "EXIT",
            "EXIT TYPE",
            "NET",
            "+3%",
        ]
    ]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=min(
            430,
            90 + len(table) * 35,
        ),
    )


# ============================================================
# PERIOD DIAGNOSTIC
# ============================================================

with st.expander(
    "Data / diagnostic details",
    expanded=False,
):

    d1, d2, d3 = st.columns(3)

    with d1:

        st.metric(
            "Price data starts",
            fmt_date(data_start),
        )

        st.metric(
            "Selected candles",
            f"{len(visible):,}",
        )

    with d2:

        st.metric(
            "Price data ends",
            fmt_date(data_end),
        )

        st.metric(
            "Trades in range",
            f"{trade_count}",
        )

    with d3:

        st.metric(
            "+3% hits",
            f"{target_hits}",
        )

        st.metric(
            "Median net / trade",
            (
                f"{median_return * 100:+.2f}%"
                if np.isfinite(
                    median_return
                )
                else "—"
            ),
        )

    st.caption(
        "All chart filtering and event-date comparisons "
        "are performed on timezone-naive timestamps to "
        "prevent pandas timezone comparison failures."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="tm-footer">
        Gold Strategy Time Machine
        • {timeframe} view
        • {fmt_date(start_ts)} → {fmt_date(end_ts - pd.Timedelta(days=1))}
        • Data through {fmt_date(data_end)}
    </div>
    """,
    unsafe_allow_html=True,
)