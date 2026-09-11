# Gold Breakout Time Machine — Streamlit Dashboard

This dashboard is built for two jobs:

1. **Historical time travel:** move to any date/window across the available 8+ years of daily Gold Futures data and see where the approved strategy generated BUY, SL and TSL exits on the chart.
2. **Forward monitoring:** keep the latest daily data and recent hourly candles refreshed from Yahoo Finance so the current/next signals can be monitored.

## The approved strategy

- Symbol: `GC=F` (Gold Futures)
- Direction: long only
- Breakout: 50-day Donchian high
- Trend filter: ADX >= 15
- Entry: next daily open after a confirmed daily close
- Initial stop: 2 ATR
- Trailing stop: 1.5 ATR
- Maximum hold: 15 daily bars
- +3%: milestone/measurement, **not** an automatic take-profit
- No EMA filter
- No 0.80 close-location filter

## Why the dashboard uses two event layers

### Historical research layer
`data/gold_research_events.csv` contains the exact approved candidate trades from the research event engine. The dashboard uses this file for the historical BUY/SL/TSL/TIME markers. This is the layer to use for questions such as:

> "Go back to 2024. When was the first BUY, where did it exit, and was it SL or TSL?"

### Forward/live layer
The dashboard also runs a **live-safe operational approximation** on the current daily data. It is labeled separately because its event bookkeeping has not been proven byte-for-byte identical to the historical research engine.

Do not silently mix these two layers when evaluating historical performance.

## Time-travel controls

Choose:

- **Daily — full 8+ year history**: the main research view.
- **Hourly — recent ~2 years**: zoom into intraday price action around the recent signals.

Then choose any date range. Plotly gives an interactive range slider and zooming on the chart, while the date controls make it easy to inspect a specific year such as 2024.

## Example: inspect 2024

1. Open the app.
2. Choose `Daily — full 8+ year history`.
3. Set the date range to `01/01/2024` through `31/12/2024`.
4. BUY markers are the research entries; SL/TSL/TIME markers are the research exits.
5. The table below the chart lists the exact entry and exit dates, prices, exit type and net return.
6. Use `Historical signal lookup` for a specific date to inspect nearby trades.

## Data update

GitHub Actions runs `scripts/update_data.py` hourly and commits refreshed Yahoo data into `data/`.

Yahoo intraday data is treated as a rolling recent window. The dashboard's historical daily research history remains separate from the hourly monitoring data.

## Streamlit Community Cloud

Deploy the repository as a Streamlit app with:

```text
Main file path: app.py
```

The app reads the committed CSV files directly from the repository.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important

This is a decision-support dashboard, not an automatic order execution system. Before using it for live capital, reconcile the forward operational engine against the original research event engine and define the exact intraday execution rules for stop orders.
