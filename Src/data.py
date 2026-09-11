from __future__ import annotations
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
TICKER='GC=F'

def _norm(df):
    if df is None or len(df)==0: return pd.DataFrame(columns=['timestamp_utc','open','high','low','close','volume'])
    if isinstance(df.columns, pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    x=df.reset_index()
    dt='Datetime' if 'Datetime' in x.columns else 'Date'
    x=x.rename(columns={dt:'timestamp_utc','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    x['timestamp_utc']=pd.to_datetime(x['timestamp_utc'],utc=True)
    return x[['timestamp_utc','open','high','low','close','volume']].dropna(subset=['close']).drop_duplicates('timestamp_utc').sort_values('timestamp_utc')

def fetch_hourly():
    # Yahoo Finance intraday history is limited; refresh the rolling window and merge with the seed.
    seed=pd.read_csv(DATA/'gold_hourly_seed.csv',parse_dates=['timestamp_utc'])
    seed['timestamp_utc']=pd.to_datetime(seed['timestamp_utc'],utc=True)
    fresh=_norm(yf.download(TICKER,period='729d',interval='1h',auto_adjust=False,progress=False,threads=False))
    x=pd.concat([seed,fresh],ignore_index=True).drop_duplicates('timestamp_utc',keep='last').sort_values('timestamp_utc')
    x.to_csv(DATA/'gold_hourly.csv',index=False)
    return x

def fetch_daily():
    seed=pd.read_csv(DATA/'gold_daily.csv',parse_dates=['timestamp_utc'])
    seed['timestamp_utc']=pd.to_datetime(seed['timestamp_utc'],utc=True)
    fresh=_norm(yf.download(TICKER,period='10y',interval='1d',auto_adjust=False,progress=False,threads=False))
    x=pd.concat([seed,fresh],ignore_index=True).drop_duplicates('timestamp_utc',keep='last').sort_values('timestamp_utc')
    x.to_csv(DATA/'gold_daily.csv',index=False)
    return x
