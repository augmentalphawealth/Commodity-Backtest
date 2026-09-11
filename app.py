import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from data import fetch_daily, fetch_hourly


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Gold Strategy Time Machine",
    page_icon="🟡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1500px;
    }

    [data-testid="stMetric"] {
        padding: .35rem .6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
EVENTS_FILE = DATA_DIR / "gold_research_events.csv"


# ============================================================
# DATETIME HELPERS
# ============================================================

def normalize_datetime(value):
    """
    Convert timestamps to pandas datetime and ALWAYS return
    timezone-naive values.

    This is important because Yahoo price data can contain
    timezone-aware timestamps while CSV research-event dates
    can be timezone-naive.

    Pandas does not allow direct comparison between:

        tz-aware datetime
        and
        tz-naive datetime

    This helper makes both sides consistently timezone-naive.
    """

    result = pd.to_datetime(value, errors="coerce")

    # --------------------------------------------------------
    # Series
    # --------------------------------------------------------
    if isinstance(result, pd.Series):

        # Normal timezone-aware datetime Series
        if isinstance(result.dtype, pd.DatetimeTZDtype):
            return result.dt.tz_localize(None)

        # Sometimes mixed timezone values become object dtype.
        if result.dtype == "object":
            try:
                converted = pd.to_datetime(
                    result,
                    errors="coerce",
                    utc=True,
                )

                return converted.dt.tz_localize(None)

            except Exception:
                return result

        return result

    # --------------------------------------------------------
    # DatetimeIndex
    # --------------------------------------------------------
    if isinstance(result, pd.DatetimeIndex):

        if result.tz is not None:
            return result.tz_localize(None)

        return result

    # --------------------------------------------------------
    # Scalar Timestamp / other datetime-like object
    # --------------------------------------------------------
    try:
        if getattr(result, "tz", None) is not None:
            return result.tz_localize(None)
    except Exception:
        pass

    return result


def safe_timestamp(value):
    """
    Convert any datetime-like value into a timezone-naive
    pandas Timestamp.
    """

    ts = pd.Timestamp(value)

    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    return ts


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(ttl=3600)
def load_daily():
    return fetch_daily()


@st.cache_data(ttl=3600)
def load_hourly():
    return fetch_hourly()


@st.cache_data
def load_events():

    if not EVENTS_FILE.exists():

        st.error(
            "Research event file was not found:\n\n"
            f"{EVENTS_FILE}\n\n"
            "Make sure `data/gold_research_events.csv` "
            "is committed to the GitHub repository."
        )

        st.stop()

    df = pd.read_csv(EVENTS_FILE)

    required_columns = [
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
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        st.error(
            "gold_research_events.csv is missing required "
            "columns:\n\n"
            + ", ".join(missing)
        )

        st.stop()

    # --------------------------------------------------------
    # IMPORTANT:
    # Normalize all event dates immediately after reading CSV.
    # --------------------------------------------------------

    for column in [
        "signal_date",
        "entry_date",
        "exit_date",
    ]:
        df[column] = normalize_datetime(df[column])

    return df


# ============================================================
# HEADER
# ============================================================

st.title("🟡 Gold Strategy — Time Machine")

st.caption(
    "Historical BUY → SL / TSL / TIME • "
    "Approved research configuration"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("View")

    timeframe = st.radio(
        "Candle timeframe",
        ["Daily", "Hourly"],
        index=0,
    )

    period = st.selectbox(
        "Quick range",
        [
            "3 months",
            "6 months",
            "1 year",
            "2 years",
            "5 years",
            "All available",
        ],
        index=2,
    )

    st.subheader("Signals")

    show_buy = st.checkbox(
        "🟢 BUY",
        True,
    )

    show_sl = st.checkbox(
        "🔴 SL",
        True,
    )

    show_tsl = st.checkbox(
        "🟠 TSL",
        True,
    )

    show_time = st.checkbox(
        "⚪ TIME",
        False,
    )

    st.divider()

    if st.button(
        "↻ Refresh Yahoo data",
        use_container_width=True,
    ):

        load_daily.clear()
        load_hourly.clear()

        st.rerun()


# ============================================================
# LOAD DATA
# ============================================================

daily = load_daily().copy()

hourly = load_hourly().copy()

events = load_events().copy()


# ============================================================
# NORMALIZE PRICE INDEXES
# ============================================================

daily.index = normalize_datetime(
    daily.index
)

hourly.index = normalize_datetime(
    hourly.index
)


# ============================================================
# NORMALIZE EVENT DATES AGAIN
# ============================================================
#
# This is intentionally repeated defensively.
#
# It guarantees that the objects used later in comparisons
# are timezone-naive even if Streamlit caching or pandas
# reconstruction changes the dtype.
# ============================================================

for column in [
    "signal_date",
    "entry_date",
    "exit_date",
]:

    events[column] = normalize_datetime(
        events[column]
    )


# Remove unusable timestamps.

events = events.dropna(
    subset=[
        "signal_date",
        "entry_date",
        "exit_date",
    ]
).copy()


daily = daily[
    ~daily.index.isna()
].sort_index()


hourly = hourly[
    ~hourly.index.isna()
].sort_index()


# ============================================================
# SELECT PRICE TIMEFRAME
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
# NORMALIZE PRICE INDEX ONE MORE TIME
# ============================================================

prices.index = normalize_datetime(
    prices.index
)

prices = prices[
    ~prices.index.isna()
].sort_index()


# ============================================================
# AVAILABLE DATE RANGE
# ============================================================

first_date = safe_timestamp(
    prices.index.min()
)

last_date = safe_timestamp(
    prices.index.max()
)


# ============================================================
# QUICK RANGE
# ============================================================

offsets = {

    "3 months":
        pd.DateOffset(months=3),

    "6 months":
        pd.DateOffset(months=6),

    "1 year":
        pd.DateOffset(years=1),

    "2 years":
        pd.DateOffset(years=2),

    "5 years":
        pd.DateOffset(years=5),
}


offset = offsets.get(
    period
)


if offset is None:

    default_start = first_date

else:

    default_start = max(
        first_date,
        last_date - offset,
    )


# ============================================================
# DATE INPUTS
# ============================================================

with st.sidebar:

    start_date = st.date_input(
        "From",
        value=default_start.date(),
        min_value=first_date.date(),
        max_value=last_date.date(),
    )

    end_date = st.date_input(
        "To",
        value=last_date.date(),
        min_value=first_date.date(),
        max_value=last_date.date(),
    )


# ============================================================
# CONVERT USER DATES TO TIMEZONE-NAIVE TIMESTAMPS
# ============================================================

start_ts = pd.Timestamp(
    start_date
)

end_ts = (
    pd.Timestamp(end_date)
    + pd.Timedelta(days=1)
)


# Explicitly guarantee no timezone.

start_ts = safe_timestamp(
    start_ts
)

end_ts = safe_timestamp(
    end_ts
)


# ============================================================
# DATE VALIDATION
# ============================================================

if start_date > end_date:

    st.error(
        "The From date must be before the To date."
    )

    st.stop()


# ============================================================
# FILTER VISIBLE PRICE DATA
# ============================================================

visible = prices.loc[
    (prices.index >= start_ts)
    &
    (prices.index < end_ts)
].copy()


if visible.empty:

    st.warning(
        "No price candles exist in the selected period."
    )

    st.stop()


# ============================================================
# CHART
# ============================================================

fig = go.Figure()


# ------------------------------------------------------------
# Candlestick
# ------------------------------------------------------------

fig.add_trace(
    go.Candlestick(

        x=visible.index,

        open=visible["Open"],

        high=visible["High"],

        low=visible["Low"],

        close=visible["Close"],

        name="Gold",

        increasing_line_color="#198754",

        decreasing_line_color="#dc3545",

        increasing_fillcolor="#198754",

        decreasing_fillcolor="#dc3545",

        whiskerwidth=0.45,
    )
)


# ============================================================
# EVENT MARKER DEFINITIONS
# ============================================================

marker_defs = [

    (
        "BUY",
        show_buy,
        "entry_date",
        "triangle-up",
        "#198754",
        "low",
    ),

    (
        "SL",
        show_sl,
        "exit_date",
        "x",
        "#dc3545",
        "high",
    ),

    (
        "TSL",
        show_tsl,
        "exit_date",
        "triangle-down",
        "#f39c12",
        "high",
    ),

    (
        "TIME",
        show_time,
        "exit_date",
        "circle-open",
        "#6c757d",
        "high",
    ),
]


# ============================================================
# DRAW EVENT MARKERS
# ============================================================

for (
    name,
    enabled,
    date_col,
    symbol,
    color,
    placement,
) in marker_defs:

    if not enabled:
        continue


    # --------------------------------------------------------
    # Start with a copy.
    # --------------------------------------------------------

    e = events.copy()


    # --------------------------------------------------------
    # BUY events
    # --------------------------------------------------------

    if name == "BUY":

        e = e[
            e["side"]
            .astype(str)
            .str.lower()
            .eq("long")
        ]


    # --------------------------------------------------------
    # Exit events
    # --------------------------------------------------------

    else:

        e = e[
            e["exit_reason"]
            .astype(str)
            .str.upper()
            .eq(name)
        ]


    # --------------------------------------------------------
    # CRITICAL TIMEZONE-SAFE FILTER
    # --------------------------------------------------------
    #
    # e[date_col] is guaranteed timezone-naive.
    #
    # start_ts and end_ts are also guaranteed timezone-naive.
    #
    # Therefore this comparison cannot produce:
    #
    # TypeError:
    # Invalid comparison between dtype=datetime64[ns, UTC]
    # and Timestamp
    #
    # --------------------------------------------------------

    event_dates = normalize_datetime(
        e[date_col]
    )

    e[date_col] = event_dates

    e = e[
        (e[date_col] >= start_ts)
        &
        (e[date_col] < end_ts)
    ].copy()


    if e.empty:
        continue


    xs = []

    ys = []

    texts = []


    # ========================================================
    # INDIVIDUAL EVENTS
    # ========================================================

    for _, row in e.iterrows():

        dt = safe_timestamp(
            row[date_col]
        )


        # ----------------------------------------------------
        # Find first available candle at or after event.
        # ----------------------------------------------------

        candidates = visible.index[
            visible.index >= dt
        ]


        if len(candidates):

            x = candidates[0]

        else:

            # ------------------------------------------------
            # If no candle after event, use latest candle
            # before event.
            # ------------------------------------------------

            candidates = visible.index[
                visible.index <= dt
            ]

            if not len(candidates):
                continue

            x = candidates[-1]


        candle = visible.loc[x]


        # ----------------------------------------------------
        # Event price
        # ----------------------------------------------------

        if name == "BUY":

            price = float(
                row["entry"]
            )

        else:

            price = float(
                row["exit"]
            )


        # ----------------------------------------------------
        # Marker vertical position
        # ----------------------------------------------------

        if placement == "low":

            y = (
                float(candle["Low"])
                * 0.997
            )

        else:

            y = (
                float(candle["High"])
                * 1.003
            )


        # ====================================================
        # BUY HOVER TEXT
        # ====================================================

        if name == "BUY":

            entry_date_text = safe_timestamp(
                row["entry_date"]
            ).strftime(
                "%d %b %Y"
            )

            exit_date_text = safe_timestamp(
                row["exit_date"]
            ).strftime(
                "%d %b %Y"
            )


            target_hit = (
                "YES"
                if bool(row["target_hit"])
                else "NO"
            )


            text = (

                f"<b>BUY</b><br>"

                f"{entry_date_text} "
                f"@ {price:,.2f}<br>"

                f"Exit: "
                f"{exit_date_text} "
                f"@ {float(row['exit']):,.2f}<br>"

                f"Exit: "
                f"{row['exit_reason']}<br>"

                f"+3% milestone: "
                f"{target_hit}"
            )


        # ====================================================
        # EXIT HOVER TEXT
        # ====================================================

        else:

            exit_date_text = safe_timestamp(
                row["exit_date"]
            ).strftime(
                "%d %b %Y"
            )

            entry_date_text = safe_timestamp(
                row["entry_date"]
            ).strftime(
                "%d %b %Y"
            )


            net_return = (
                float(row["net_return"])
                * 100
            )


            text = (

                f"<b>{name}</b><br>"

                f"{exit_date_text} "
                f"@ {price:,.2f}<br>"

                f"Entry: "
                f"{entry_date_text} "
                f"@ {float(row['entry']):,.2f}<br>"

                f"Net return: "
                f"{net_return:.2f}%"
            )


        xs.append(x)

        ys.append(y)

        texts.append(text)


    # ========================================================
    # ADD MARKER TRACE
    # ========================================================

    if xs:

        fig.add_trace(

            go.Scatter(

                x=xs,

                y=ys,

                mode="markers",

                name=name,

                marker=dict(

                    symbol=symbol,

                    size=(
                        14
                        if name == "BUY"
                        else 12
                    ),

                    color=color,

                    line=dict(
                        color="white",
                        width=1.5,
                    ),
                ),

                text=texts,

                hovertemplate=(
                    "%{text}"
                    "<extra></extra>"
                ),
            )
        )


# ============================================================
# CHART LAYOUT
# ============================================================

fig.update_layout(

    height=700,

    template="plotly_white",

    margin=dict(
        l=10,
        r=20,
        t=45,
        b=15,
    ),

    paper_bgcolor="white",

    plot_bgcolor="white",

    hovermode="x",

    dragmode="zoom",

    legend=dict(

        orientation="h",

        y=1.04,

        x=0,

        bgcolor=(
            "rgba(255,255,255,.9)"
        ),
    ),

    xaxis=dict(

        title=None,

        showgrid=False,

        rangeslider=dict(
            visible=False
        ),

        showspikes=True,

        spikemode="across",

        spikesnap="cursor",

        showline=True,

        linecolor="#adb5bd",

        rangeselector=dict(

            buttons=[

                dict(
                    count=3,
                    label="3M",
                    step="month",
                    stepmode="backward",
                ),

                dict(
                    count=6,
                    label="6M",
                    step="month",
                    stepmode="backward",
                ),

                dict(
                    count=1,
                    label="1Y",
                    step="year",
                    stepmode="backward",
                ),

                dict(
                    count=2,
                    label="2Y",
                    step="year",
                    stepmode="backward",
                ),

                dict(
                    step="all",
                    label="ALL",
                ),
            ],

            x=0,

            y=1.08,
        ),
    ),

    yaxis=dict(

        title="Gold price",

        side="right",

        showgrid=True,

        gridcolor=(
            "rgba(108,117,125,.12)"
        ),

        zeroline=False,

        tickformat=",.0f",

        showline=True,

        linecolor="#adb5bd",
    ),
)


# ============================================================
# DISPLAY CHART
# ============================================================

st.plotly_chart(

    fig,

    use_container_width=True,

    config={

        "displaylogo": False,

        "scrollZoom": True,

        "displayModeBar": True,

        "modeBarButtonsToRemove": [
            "lasso2d",
            "select2d",
            "autoScale2d",
        ],
    },
)


# ============================================================
# TRADE TABLE
# ============================================================

# Normalize one more time before filtering.

events["entry_date"] = normalize_datetime(
    events["entry_date"]
)

events["exit_date"] = normalize_datetime(
    events["exit_date"]
)


ev = events[
    (events["entry_date"] >= start_ts)
    &
    (events["entry_date"] < end_ts)
].copy()


# ============================================================
# TRADE TABLE HEADER
# ============================================================

st.subheader(
    "Trades in selected period"
)


if ev.empty:

    st.info(
        "No research trades in this period."
    )


else:

    # --------------------------------------------------------
    # Build display table.
    # --------------------------------------------------------

    out = pd.DataFrame(

        {

            "BUY": ev["entry_date"].map(
                lambda x:
                    safe_timestamp(x).strftime(
                        "%d %b %Y"
                    )
            ),

            "Entry": ev["entry"].map(
                lambda x:
                    f"{float(x):,.2f}"
            ),

            "EXIT": ev["exit_date"].map(
                lambda x:
                    safe_timestamp(x).strftime(
                        "%d %b %Y"
                    )
            ),

            "Exit": ev["exit"].map(
                lambda x:
                    f"{float(x):,.2f}"
            ),

            "Type": ev["exit_reason"],

            "Net": ev["net_return"].map(
                lambda x:
                    f"{float(x) * 100:.2f}%"
            ),

            "+3%": ev["target_hit"].map(
                lambda x:
                    "YES"
                    if bool(x)
                    else "NO"
            ),
        }
    )


    # --------------------------------------------------------
    # Newest trade first.
    # --------------------------------------------------------

    out = out.sort_values(
        "BUY",
        ascending=False,
    )


    st.dataframe(

        out,

        use_container_width=True,

        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(

    f"Data through "
    f"{last_date.strftime('%d %b %Y %H:%M')} "
    f"• "

    f"{len(visible):,} "
    f"{timeframe.lower()} candles "
    f"• "

    f"Times shown in the dataset's "
    f"normalized timezone."
)
