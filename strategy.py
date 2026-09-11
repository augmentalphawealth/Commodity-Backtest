from __future__ import annotations
import numpy as np
import pandas as pd

DONCHIAN = 50
ADX_MIN = 15.0
ATR_PERIOD = 14
INITIAL_STOP_ATR = 2.0
TRAIL_ATR = 1.5
MAX_HOLD_BARS = 15
TARGET = 0.03


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_w
    mdi = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_w
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values('timestamp_utc').drop_duplicates('timestamp_utc')
    x['atr'] = atr(x)
    x['adx'] = adx(x)
    x['donchian_high'] = x['high'].rolling(DONCHIAN, min_periods=DONCHIAN).max().shift(1)
    return x


def backtest_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Operational/live-safe daily approximation used only for the forward layer.

    Historical research markers in the dashboard do NOT come from this function; they come
    from data/gold_research_events.csv, which is the approved research output.
    """
    d = add_indicators(df).reset_index(drop=True)
    events = []
    position = None
    pending = None

    for i in range(len(d)):
        row = d.iloc[i]

        if pending is not None and position is None:
            entry_idx = pending
            e = d.iloc[entry_idx]
            if pd.isna(e['atr']):
                pending = None
                continue
            entry = float(e['open'])
            atr0 = float(d.iloc[entry_idx - 1]['atr']) if entry_idx > 0 and pd.notna(d.iloc[entry_idx - 1]['atr']) else float(e['atr'])
            position = {
                'entry_idx': entry_idx,
                'entry_date': e['timestamp_utc'],
                'entry': entry,
                'atr0': atr0,
                'initial_stop': entry - INITIAL_STOP_ATR * atr0,
                'highest_close': entry,
                'target': entry * (1 + TARGET),
                'target_hit': False,
            }
            events.append({
                'timestamp_utc': e['timestamp_utc'],
                'event': 'BUY',
                'price': entry,
                'entry': entry,
                'initial_stop': position['initial_stop'],
                'target': position['target'],
            })
            pending = None

        if position is not None and i >= position['entry_idx']:
            r = row
            position['highest_close'] = max(position['highest_close'], float(r['close']))
            trail = position['highest_close'] - TRAIL_ATR * float(r['atr']) if pd.notna(r['atr']) else np.nan
            active_stop = max(position['initial_stop'], trail) if pd.notna(trail) else position['initial_stop']
            position['trail_stop'] = trail

            if float(r['high']) >= position['target']:
                position['target_hit'] = True

            held = i - position['entry_idx']
            exit_reason = None
            exit_price = None
            if float(r['low']) <= active_stop:
                exit_reason = 'SL' if active_stop <= position['initial_stop'] + 1e-9 else 'TSL'
                exit_price = active_stop
            elif held >= MAX_HOLD_BARS:
                exit_reason = 'TIME'
                exit_price = float(r['close'])

            if exit_reason:
                events.append({
                    'timestamp_utc': r['timestamp_utc'],
                    'event': exit_reason,
                    'price': exit_price,
                    'entry': position['entry'],
                    'initial_stop': position['initial_stop'],
                    'trail_stop': position.get('trail_stop', np.nan),
                    'target': position['target'],
                    'target_hit': position['target_hit'],
                    'gross_return': exit_price / position['entry'] - 1,
                    'hold_bars': held,
                })
                position = None

        # Signal after today's close for next day's open. No look-ahead.
        if position is None and i + 1 < len(d):
            if (
                pd.notna(row['donchian_high'])
                and pd.notna(row['adx'])
                and row['close'] > row['donchian_high']
                and row['adx'] >= ADX_MIN
            ):
                pending = i + 1

    return pd.DataFrame(events)
