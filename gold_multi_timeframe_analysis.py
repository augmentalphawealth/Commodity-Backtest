from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA_DIR = Path("data_usd")
OUT = Path("gold_results")
OUT.mkdir(exist_ok=True)

HOURLY_FILE = DATA_DIR / "gold_usd_1h.csv"
DAILY_FILE = DATA_DIR / "gold_usd_1d.csv"
COST = 0.0005
TARGETS = [0.03, 0.04, 0.05]
DONCHIAN = [10, 20, 50]
ADX_LEVELS = [15, 20, 25]
INITIAL_STOPS = [1.0, 1.25, 1.5, 2.0]
TRAIL_STOPS = [1.0, 1.5, 2.0, 2.5, 3.0]
HOLDS = {"1h": [24, 48, 120], "4h": [12, 24, 60], "1d": [5, 10, 15]}


def load_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path, parse_dates=["timestamp_utc"]).set_index("timestamp_utc").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df.dropna(subset=["open", "high", "low", "close"])


def resample_ohlc(df, rule):
    return df.resample(rule).agg({"open":"first", "high":"max", "low":"min", "close":"last", "volume":"sum"}).dropna()


def indicators(df, n):
    x = df.copy()
    prev = x.close.shift(1)
    tr = pd.concat([(x.high-x.low), (x.high-prev).abs(), (x.low-prev).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14).mean()
    x["ema20"] = x.close.ewm(span=20, adjust=False).mean()
    x["ema50"] = x.close.ewm(span=50, adjust=False).mean()
    up, down = x.high.diff(), -x.low.diff()
    plus = up.where((up > down) & (up > 0), 0.0).rolling(14).mean()
    minus = down.where((down > up) & (down > 0), 0.0).rolling(14).mean()
    pdi, mdi = 100*plus/x.atr, 100*minus/x.atr
    x["adx"] = (100*(pdi-mdi).abs()/(pdi+mdi)).rolling(14).mean()
    change = x.close.diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = -change.clip(upper=0).rolling(14).mean()
    x["rsi"] = 100-100/(1+gain/loss.replace(0, np.nan))
    x["prior_high"] = x.high.shift(1).rolling(n).max()
    x["prior_low"] = x.low.shift(1).rolling(n).min()
    x["location"] = (x.close-x.low)/(x.high-x.low).replace(0, np.nan)
    return x.dropna()


def evaluate(x, i, side, target, hold, initial_mult, trail_mult):
    entry_i = i+1
    if entry_i >= len(x): return None
    entry = float(x.iloc[entry_i].open)
    signal_atr = float(x.iloc[i].atr)
    if not np.isfinite(entry) or not np.isfinite(signal_atr) or signal_atr <= 0: return None
    path = x.iloc[entry_i:min(len(x), entry_i+hold)]
    target_price = entry*(1+target) if side == "long" else entry*(1-target)
    stop = entry-initial_mult*signal_atr if side == "long" else entry+initial_mult*signal_atr
    best = entry
    target_hit = False
    target_bars = np.nan
    exit_price = None
    exit_bar = None
    exit_reason = None
    for j, (_, row) in enumerate(path.iterrows()):
        atr = float(row.atr)
        if side == "long":
            best = max(best, float(row.high))
            if not target_hit and float(row.high) >= target_price:
                target_hit, target_bars = True, j+1
            if target_hit: stop = max(stop, best-trail_mult*atr)
            if float(row.low) <= stop:
                exit_price, exit_bar = stop, j
                exit_reason = "trailing_stop" if target_hit else "initial_stop"
                break
        else:
            best = min(best, float(row.low))
            if not target_hit and float(row.low) <= target_price:
                target_hit, target_bars = True, j+1
            if target_hit: stop = min(stop, best+trail_mult*atr)
            if float(row.high) >= stop:
                exit_price, exit_bar = stop, j
                exit_reason = "trailing_stop" if target_hit else "initial_stop"
                break
    if exit_price is None:
        exit_bar = len(path)-1
        exit_price = float(path.iloc[-1].close)
        exit_reason = "time_exit"
    gross = exit_price/entry-1 if side == "long" else 1-exit_price/entry
    mfe = best/entry-1 if side == "long" else 1-best/entry
    return {"signal_date":x.index[i],"entry_date":x.index[entry_i],"exit_date":path.index[exit_bar],"side":side,"entry":entry,"exit":exit_price,"target":target,"target_hit":target_hit,"target_bars":target_bars,"exit_reason":exit_reason,"gross_return":gross,"net_return":gross-COST,"mfe":mfe,"initial_stop_atr":initial_mult,"trail_atr":trail_mult,"hold_bars":hold}


def run(frame, timeframe):
    rows=[]
    for n in DONCHIAN:
        x=indicators(frame,n)
        for adx_min in ADX_LEVELS:
            long_sig=(x.close>x.prior_high*1.0015)&(x.ema20>x.ema50)&(x.adx>=adx_min)&(x.rsi>=50)&(x.location>=.65)
            short_sig=(x.close<x.prior_low*.9985)&(x.ema20<x.ema50)&(x.adx>=adx_min)&(x.rsi<=50)&(x.location<=.35)
            for side,mask in [("long",long_sig),("short",short_sig)]:
                for target in TARGETS:
                    for hold in HOLDS[timeframe]:
                        for initial_mult in INITIAL_STOPS:
                            for trail_mult in TRAIL_STOPS:
                                for i in np.flatnonzero(mask.to_numpy()):
                                    r=evaluate(x,int(i),side,target,hold,initial_mult,trail_mult)
                                    if r:
                                        r.update({"timeframe":timeframe,"donchian":n,"adx_min":adx_min})
                                        rows.append(r)
    return pd.DataFrame(rows)


def summarize(events):
    keys=["timeframe","side","target","donchian","adx_min","hold_bars","initial_stop_atr","trail_atr"]
    g=events.groupby(keys,dropna=False)
    out=g.agg(trades=("net_return","size"),target_rate=("target_hit","mean"),avg_net_return=("net_return","mean"),median_net_return=("net_return","median"),average_mfe=("mfe","mean"),gross_profit=("net_return",lambda s:s[s>0].sum()),gross_loss=("net_return",lambda s:abs(s[s<0].sum()))).reset_index()
    out["profit_factor"]=out.gross_profit/out.gross_loss.replace(0,np.nan)
    out["score"]=out.target_rate*out.profit_factor.clip(upper=5)*np.log1p(out.trades)
    return out.sort_values(["score","target_rate","profit_factor"],ascending=False)


def main():
    daily=load_file(DAILY_FILE)
    hourly=load_file(HOURLY_FILE)
    frames={"1d":daily,"1h":hourly,"4h":resample_ohlc(hourly,"4h")}
    manifest={tf:{"rows":len(frame),"start":frame.index.min().isoformat() if len(frame) else None,"end":frame.index.max().isoformat() if len(frame) else None} for tf,frame in frames.items()}
    (OUT/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    all_events=[]
    for tf,frame in frames.items():
        if len(frame)<250: continue
        events=run(frame,tf)
        if not events.empty:
            events.to_csv(OUT/f"gold_events_{tf}.csv",index=False)
            all_events.append(events)
    if not all_events: raise RuntimeError("No backtest results. Check data coverage.")
    events=pd.concat(all_events,ignore_index=True)
    events.to_csv(OUT/"gold_events_all.csv",index=False)
    summary=summarize(events)
    summary.to_csv(OUT/"gold_summary_all.csv",index=False)
    summary.head(100).to_csv(OUT/"gold_best_configurations.csv",index=False)
    print(summary.head(30).to_string(index=False))

if __name__=="__main__": main()
