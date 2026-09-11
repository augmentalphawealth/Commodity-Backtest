from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.data import fetch_daily, fetch_hourly
from src.strategy import add_indicators, backtest_daily

ROOT = Path(__file__).resolve().parent
HIST_EVENTS_PATH = ROOT / 'data' / 'gold_research_events.csv'

st.set_page_config(page_title='Gold Breakout Time Machine', layout='wide')
st.title('Gold Futures — Breakout Time Machine')
st.caption(
    'Historical chart = exact research trades for the agreed system: 50D Donchian + ADX ≥ 15, '
    'long only, 2 ATR initial stop, 1.5 ATR trailing stop, 15-bar maximum hold. No 0.80 filter.'
)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=['timestamp_utc'])


def load_daily() -> pd.DataFrame:
    return load_csv(ROOT / 'data' / 'gold_daily.csv')


def load_hourly() -> pd.DataFrame:
    live_path = ROOT / 'data' / 'gold_hourly.csv'
    if live_path.exists():
        return load_csv(live_path)
    return load_csv(ROOT / 'data' / 'gold_hourly_seed.csv')


def load_research_events() -> pd.DataFrame:
    if not HIST_EVENTS_PATH.exists():
        return pd.DataFrame()
    e = pd.read_csv(HIST_EVENTS_PATH, parse_dates=['signal_date', 'entry_date', 'exit_date'])
    for c in ['signal_date', 'entry_date', 'exit_date']:
        e[c] = pd.to_datetime(e[c], utc=True)
    e['buy_event'] = 'BUY'
    e['exit_event'] = e['exit_reason'].map({
        'initial_stop': 'SL',
        'trailing_stop': 'TSL',
        'time_exit': 'TIME',
    }).fillna(e['exit_reason'].astype(str))
    return e.sort_values('entry_date').reset_index(drop=True)


if 'daily' not in st.session_state:
    st.session_state.daily = load_daily()
if 'hourly' not in st.session_state:
    st.session_state.hourly = load_hourly()

research_events = load_research_events()

with st.sidebar:
    st.header('Time travel')
    mode = st.radio('Candle timeframe', ['Daily — full 8+ year history', 'Hourly — recent 2 years'])

    if research_events.empty:
        st.error('Historical research event file is missing.')

    # Date bounds are driven by the selected candle set.
    if mode.startswith('Daily'):
        min_date = pd.to_datetime(st.session_state.daily['timestamp_utc'], utc=True).min().date()
        max_date = pd.to_datetime(st.session_state.daily['timestamp_utc'], utc=True).max().date()
        default_start = min_date
        default_end = max_date
    else:
        htmp = pd.to_datetime(st.session_state.hourly['timestamp_utc'], utc=True)
        max_ts = htmp.max()
        min_ts = max_ts - pd.Timedelta(days=730)
        min_date = max(min_ts.date(), htmp.min().date())
        max_date = max_ts.date()
        default_start = max(min_date, date(2024, 9, 22))
        default_end = max_date

    selected = st.date_input(
        'Chart date range',
        value=(default_start, default_end),
        min_value=min_date,
        max_value=max_date,
        format='DD/MM/YYYY',
        help='Pick any historical period. This is the dashboard\'s time-travel control.',
    )

    st.divider()
    if st.button('Refresh Yahoo data', use_container_width=True):
        with st.spinner('Fetching Yahoo Finance data...'):
            st.session_state.daily = fetch_daily()
            st.session_state.hourly = fetch_hourly()
        st.success('Data refreshed. Reloading dashboard…')
        st.rerun()

    st.caption('GitHub Actions can refresh Yahoo data automatically. The dashboard itself is read-only for the stored historical trades.')

# Normalize data.
daily = st.session_state.daily.copy()
daily['timestamp_utc'] = pd.to_datetime(daily['timestamp_utc'], utc=True)
daily = add_indicators(daily)
hourly = st.session_state.hourly.copy()
hourly['timestamp_utc'] = pd.to_datetime(hourly['timestamp_utc'], utc=True)
hourly = hourly.sort_values('timestamp_utc').drop_duplicates('timestamp_utc')

# Exact historical research events: only the approved candidate strategy.
if not research_events.empty:
    research_events = research_events[research_events['side'].eq('long')].copy()
    research_events = research_events[
        research_events['donchian'].eq(50)
        & research_events['adx_min'].eq(15)
        & research_events['hold_bars'].eq(15)
        & research_events['initial_stop_atr'].eq(2)
        & research_events['trail_atr'].eq(1.5)
        & research_events['target'].eq(0.03)
    ].copy()

# Operational forward layer. It starts from the daily data currently available.
# This layer is intentionally kept separate from historical research so the chart never silently
# substitutes a different implementation for the research history.
live_events = backtest_daily(daily)
if not live_events.empty:
    live_events['timestamp_utc'] = pd.to_datetime(live_events['timestamp_utc'], utc=True)

last = daily.iloc[-1]
latest_hist = research_events.iloc[-1] if not research_events.empty else None

c1, c2, c3, c4 = st.columns(4)
c1.metric('Last daily close', f"{last.close:,.2f}")
c2.metric('ADX', f"{last.adx:,.1f}" if pd.notna(last.get('adx')) else '—')
c3.metric('Historical trades', f"{len(research_events):,}")
c4.metric('Data through', pd.Timestamp(last.timestamp_utc).strftime('%d %b %Y'))

# Normalize date selection.
if isinstance(selected, tuple) and len(selected) == 2:
    start_date, end_date = selected
else:
    start_date = end_date = selected
start_ts = pd.Timestamp(start_date, tz='UTC')
end_ts = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

if mode.startswith('Daily'):
    chart_df = daily[(daily.timestamp_utc >= start_ts) & (daily.timestamp_utc <= end_ts)].copy()
    chart_label = 'Daily candles — full available history'
else:
    # Enforce the recent ~2-year hourly window, then apply the user's time-travel range.
    latest_hour = hourly['timestamp_utc'].max()
    hourly_floor = latest_hour - pd.Timedelta(days=730)
    chart_df = hourly[
        (hourly.timestamp_utc >= hourly_floor)
        & (hourly.timestamp_utc >= start_ts)
        & (hourly.timestamp_utc <= end_ts)
    ].copy()
    chart_label = 'Hourly candles — recent ~2 years'

if chart_df.empty:
    st.warning('No candles exist in the selected date range.')
    st.stop()

# Historical event markers. On daily charts, use the actual research entry/exit dates and prices.
# On hourly charts, snap the event timestamp to the first available hourly candle at/after the event.
def snap_events_to_chart(events: pd.DataFrame, ts_col: str, price_col: str = 'price') -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=['timestamp_utc', 'price', 'event'])
    base = chart_df[['timestamp_utc', 'open', 'high', 'low', 'close']].sort_values('timestamp_utc')
    q = events[[ts_col, price_col]].copy().rename(columns={ts_col: 'timestamp_utc', price_col: 'price'})
    q = q.sort_values('timestamp_utc')
    if mode.startswith('Hourly'):
        q = pd.merge_asof(q, base[['timestamp_utc']], on='timestamp_utc', direction='forward')
    return q.dropna(subset=['timestamp_utc']).drop_duplicates(['timestamp_utc', 'price'])

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=chart_df.timestamp_utc,
    open=chart_df.open,
    high=chart_df.high,
    low=chart_df.low,
    close=chart_df.close,
    name='GC=F',
))

# Research BUY markers.
if not research_events.empty:
    hist_buy = research_events[
        (research_events.entry_date >= start_ts) & (research_events.entry_date <= end_ts)
    ].copy()
    buy = snap_events_to_chart(hist_buy, 'entry_date', 'entry')
    if not buy.empty:
        buy['event'] = 'BUY'
        fig.add_trace(go.Scatter(
            x=buy.timestamp_utc, y=buy.price, mode='markers+text',
            text=['BUY'] * len(buy), textposition='top center',
            marker=dict(symbol='triangle-up', size=12), name='BUY (research)',
            customdata=buy[['price']].values,
            hovertemplate='BUY<br>%{x|%d %b %Y %H:%M}<br>Price: %{y:,.2f}<extra></extra>',
        ))

    # Research exits: SL / TSL / TIME.
    hist_exit = research_events[
        (research_events.exit_date >= start_ts) & (research_events.exit_date <= end_ts)
    ].copy()
    for event_name, reason, symbol in [
        ('SL', 'initial_stop', 'x'),
        ('TSL', 'trailing_stop', 'diamond'),
        ('TIME', 'time_exit', 'circle'),
    ]:
        q = hist_exit[hist_exit.exit_reason.eq(reason)]
        q = snap_events_to_chart(q, 'exit_date', 'exit')
        if not q.empty:
            q['event'] = event_name
            fig.add_trace(go.Scatter(
                x=q.timestamp_utc, y=q.price, mode='markers+text',
                text=[event_name] * len(q), textposition='bottom center',
                marker=dict(symbol=symbol, size=11), name=f'{event_name} (research)',
                hovertemplate=f'{event_name}<br>%{{x|%d %b %Y %H:%M}}<br>Price: %{{y:,.2f}}<extra></extra>',
            ))

# Optional forward/live layer: only events after the last research entry/exit date.
# This keeps the historical time machine anchored to the authoritative research file.
if latest_hist is not None and not live_events.empty:
    live_cut = max(research_events.entry_date.max(), research_events.exit_date.max())
    forward = live_events[live_events.timestamp_utc > live_cut].copy()
    forward = forward[(forward.timestamp_utc >= start_ts) & (forward.timestamp_utc <= end_ts)]
    for typ, symbol in [('BUY', 'triangle-up'), ('SL', 'x'), ('TSL', 'diamond'), ('TIME', 'circle')]:
        q = forward[forward.event.eq(typ)][['timestamp_utc', 'price']].copy()
        if q.empty:
            continue
        q = snap_events_to_chart(q, 'timestamp_utc', 'price')
        fig.add_trace(go.Scatter(
            x=q.timestamp_utc, y=q.price, mode='markers+text',
            text=[typ] * len(q), textposition='top center',
            marker=dict(symbol=symbol, size=10), name=f'{typ} (forward)',
            hovertemplate=f'FORWARD {typ}<br>%{{x|%d %b %Y %H:%M}}<br>Price: %{{y:,.2f}}<extra></extra>',
        ))

fig.update_layout(
    height=760,
    xaxis_rangeslider_visible=True,
    xaxis_title='Date / time',
    yaxis_title='Gold price',
    template='plotly_white',
    hovermode='x unified',
    legend_orientation='h',
    margin=dict(l=10, r=10, t=40, b=10),
)

st.subheader(chart_label)
st.plotly_chart(
    fig,
    use_container_width=True,
    config={'scrollZoom': True, 'displaylogo': False, 'responsive': True},
)

# Trade lookup table for the selected window.
st.subheader('Trades in selected period')
if not research_events.empty:
    selected_trades = research_events[
        (research_events.entry_date <= end_ts) & (research_events.exit_date >= start_ts)
    ].copy()
    selected_trades['Exit signal'] = selected_trades['exit_reason'].map({
        'initial_stop': 'SL', 'trailing_stop': 'TSL', 'time_exit': 'TIME'
    })
    selected_trades['Return'] = selected_trades['net_return'].map(lambda x: f'{x:.2%}')
    selected_trades['+3% hit'] = selected_trades['target_hit'].map(lambda x: 'YES' if bool(x) else 'NO')
    table = selected_trades[
        ['signal_date', 'entry_date', 'entry', 'exit_date', 'exit', 'Exit signal', 'Return', '+3% hit']
    ].sort_values('entry_date', ascending=False)
    st.dataframe(table, use_container_width=True, hide_index=True)
else:
    st.info('No historical research events found.')

# “What happened around this date?” lookup.
st.subheader('Historical signal lookup')
lookup_date = st.date_input(
    'Choose a date to inspect',
    value=min(max(date(2024, 1, 1), min_date), max_date),
    min_value=min_date,
    max_value=max_date,
    format='DD/MM/YYYY',
    key='lookup_date',
)
lookup_ts = pd.Timestamp(lookup_date, tz='UTC')
if not research_events.empty:
    nearby = research_events[
        (research_events.entry_date >= lookup_ts - pd.Timedelta(days=10))
        & (research_events.exit_date <= lookup_ts + pd.Timedelta(days=20))
    ].copy()
    if nearby.empty:
        st.info('No research trade starts/ends close to this date.')
    else:
        nearby['BUY'] = nearby['entry_date'].dt.strftime('%d %b %Y')
        nearby['EXIT'] = nearby['exit_date'].dt.strftime('%d %b %Y')
        nearby['EXIT TYPE'] = nearby['exit_reason'].map({
            'initial_stop': 'SL', 'trailing_stop': 'TSL', 'time_exit': 'TIME'
        })
        nearby['NET'] = nearby['net_return'].map(lambda x: f'{x:.2%}')
        st.dataframe(
            nearby[['BUY', 'entry', 'EXIT', 'exit', 'EXIT TYPE', 'NET', 'target_hit']],
            use_container_width=True,
            hide_index=True,
        )

st.info(
    'Historical BUY/SL/TSL/TIME markers come from the approved research event file, so the chart is a true '
    'time-travel view of the tested system. The forward layer is intentionally labeled separately because its '
    'live-safe implementation has not been certified as byte-for-byte identical to the research engine.'
)
