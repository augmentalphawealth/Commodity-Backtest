import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data import fetch_daily, fetch_hourly

st.set_page_config(page_title="Gold Strategy Time Machine", page_icon="🟡", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:1rem; padding-bottom:1rem; max-width:1500px;}
[data-testid="stMetric"] {padding:.35rem .6rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_daily():
    return fetch_daily()

@st.cache_data(ttl=3600)
def load_hourly():
    return fetch_hourly()

@st.cache_data
def load_events():
    df = pd.read_csv("data/gold_research_events.csv")
    for c in ["signal_date", "entry_date", "exit_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def naive_index(idx):
    idx = pd.to_datetime(idx, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx

st.title("🟡 Gold Strategy — Time Machine")
st.caption("Historical BUY → SL / TSL / TIME • Approved research configuration")

with st.sidebar:
    st.header("View")
    timeframe = st.radio("Candle timeframe", ["Daily", "Hourly"], index=0)
    period = st.selectbox("Quick range", ["3 months", "6 months", "1 year", "2 years", "5 years", "All available"], index=2)

    st.subheader("Signals")
    show_buy = st.checkbox("🟢 BUY", True)
    show_sl = st.checkbox("🔴 SL", True)
    show_tsl = st.checkbox("🟠 TSL", True)
    show_time = st.checkbox("⚪ TIME", False)

    if st.button("↻ Refresh Yahoo data", use_container_width=True):
        load_daily.clear()
        load_hourly.clear()
        st.rerun()

daily = load_daily().copy()
hourly = load_hourly().copy()
events = load_events().copy()

daily.index = naive_index(daily.index)
hourly.index = naive_index(hourly.index)
for c in ["signal_date", "entry_date", "exit_date"]:
    events[c] = naive_index(events[c])

prices = daily.sort_index() if timeframe == "Daily" else hourly.sort_index()
last_date = prices.index.max()

offset = {
    "3 months": pd.DateOffset(months=3),
    "6 months": pd.DateOffset(months=6),
    "1 year": pd.DateOffset(years=1),
    "2 years": pd.DateOffset(years=2),
    "5 years": pd.DateOffset(years=5),
}.get(period)

default_start = prices.index.min() if offset is None else last_date - offset

with st.sidebar:
    start_date = st.date_input("From", value=default_start.date(), min_value=prices.index.min().date(), max_value=last_date.date())
    end_date = st.date_input("To", value=last_date.date(), min_value=prices.index.min().date(), max_value=last_date.date())

if start_date > end_date:
    st.error("The From date must be before the To date.")
    st.stop()

visible = prices.loc[
    (prices.index >= pd.Timestamp(start_date)) &
    (prices.index < pd.Timestamp(end_date) + pd.Timedelta(days=1))
].copy()

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=visible.index,
    open=visible["Open"], high=visible["High"],
    low=visible["Low"], close=visible["Close"],
    name="Gold",
    increasing_line_color="#198754",
    decreasing_line_color="#dc3545",
    increasing_fillcolor="#198754",
    decreasing_fillcolor="#dc3545",
    whiskerwidth=0.45,
))

# Marker definitions: event, enabled, date field, symbol, color, vertical placement.
marker_defs = [
    ("BUY", show_buy, "entry_date", "triangle-up", "#198754", "low"),
    ("SL", show_sl, "exit_date", "x", "#dc3545", "high"),
    ("TSL", show_tsl, "exit_date", "triangle-down", "#f39c12", "high"),
    ("TIME", show_time, "exit_date", "circle-open", "#6c757d", "high"),
]

for name, enabled, date_col, symbol, color, placement in marker_defs:
    if not enabled:
        continue

    e = events.copy()
    if name == "BUY":
        e = e[e["side"].eq("long")]
    else:
        e = e[e["exit_reason"].eq(name)]

    e = e[
        (e[date_col] >= pd.Timestamp(start_date)) &
        (e[date_col] < pd.Timestamp(end_date) + pd.Timedelta(days=1))
    ].copy()

    if e.empty:
        continue

    xs, ys, texts = [], [], []
    for _, r in e.iterrows():
        dt = r[date_col]
        # Put historical daily events on the exact daily candle for Daily view.
        # For Hourly view, place them on the first available hourly candle at/after the event.
        candidates = visible.index[visible.index >= dt]
        if len(candidates):
            x = candidates[0]
        else:
            candidates = visible.index[visible.index <= dt]
            if not len(candidates):
                continue
            x = candidates[-1]

        candle = visible.loc[x]
        price = float(r["entry"] if name == "BUY" else r["exit"])
        y = float(candle["Low"]) * 0.997 if placement == "low" else float(candle["High"]) * 1.003

        if name == "BUY":
            text = (
                f"<b>BUY</b><br>{pd.Timestamp(r['entry_date']).strftime('%d %b %Y')} "
                f"@ {price:,.2f}<br>"
                f"Exit: {pd.Timestamp(r['exit_date']).strftime('%d %b %Y')} "
                f"@ {float(r['exit']):,.2f}<br>"
                f"Exit: {r['exit_reason']}<br>"
                f"+3% milestone: {'YES' if bool(r['target_hit']) else 'NO'}"
            )
        else:
            text = (
                f"<b>{name}</b><br>{pd.Timestamp(r['exit_date']).strftime('%d %b %Y')} "
                f"@ {price:,.2f}<br>"
                f"Entry: {pd.Timestamp(r['entry_date']).strftime('%d %b %Y')} "
                f"@ {float(r['entry']):,.2f}<br>"
                f"Net return: {float(r['net_return'])*100:.2f}%"
            )

        xs.append(x); ys.append(y); texts.append(text)

    if xs:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=name,
            marker=dict(symbol=symbol, size=14 if name == "BUY" else 12,
                        color=color, line=dict(color="white", width=1.5)),
            text=texts, hovertemplate="%{text}<extra></extra>",
        ))

fig.update_layout(
    height=700,
    template="plotly_white",
    margin=dict(l=10, r=20, t=45, b=15),
    paper_bgcolor="white",
    plot_bgcolor="white",
    hovermode="x",
    dragmode="zoom",
    legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(255,255,255,.9)"),
    xaxis=dict(
        title=None, showgrid=False, rangeslider=dict(visible=False),
        showspikes=True, spikemode="across", spikesnap="cursor",
        showline=True, linecolor="#adb5bd",
        rangeselector=dict(
            buttons=[
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=2, label="2Y", step="year", stepmode="backward"),
                dict(step="all", label="ALL"),
            ],
            x=0, y=1.08
        ),
    ),
    yaxis=dict(
        title="Gold price",
        side="right",
        showgrid=True,
        gridcolor="rgba(108,117,125,.12)",
        zeroline=False,
        tickformat=",.0f",
        showline=True, linecolor="#adb5bd",
    ),
)

st.plotly_chart(
    fig, use_container_width=True,
    config={
        "displaylogo": False,
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    },
)

ev = events[
    (events["entry_date"] >= pd.Timestamp(start_date)) &
    (events["entry_date"] < pd.Timestamp(end_date) + pd.Timedelta(days=1))
].copy()

st.subheader("Trades in selected period")

if ev.empty:
    st.info("No research trades in this period.")
else:
    out = pd.DataFrame({
        "BUY": ev["entry_date"].dt.strftime("%d %b %Y"),
        "Entry": ev["entry"].map(lambda x: f"{x:,.2f}"),
        "EXIT": ev["exit_date"].dt.strftime("%d %b %Y"),
        "Exit": ev["exit"].map(lambda x: f"{x:,.2f}"),
        "Type": ev["exit_reason"],
        "Net": ev["net_return"].map(lambda x: f"{x*100:.2f}%"),
        "+3%": ev["target_hit"].map(lambda x: "YES" if bool(x) else "NO"),
    })
    st.dataframe(out.sort_values("BUY", ascending=False), use_container_width=True, hide_index=True)

st.caption(
    f"Data through {last_date.strftime('%d %b %Y %H:%M')} • "
    f"{len(visible):,} {timeframe.lower()} candles • "
    f"Times shown in the dataset's normalized timezone."
)
