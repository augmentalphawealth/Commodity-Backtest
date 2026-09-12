import yfinance as yf


TICKER = "GC=F"


def main():
    print(f"Checking {TICKER} market status...")

    try:
        ticker = yf.Ticker(TICKER)

        # Request recent 1-minute data.
        # If Yahoo is currently receiving trades, recent rows should exist.
        data = ticker.history(
            period="1d",
            interval="1m",
            prepost=True,
        )

        if data is None or data.empty:
            print("No recent Yahoo Gold data found.")
            print("market_open=false")
            return

        data = data.dropna(subset=["Close"])

        if data.empty:
            print("No valid Gold prices found.")
            print("market_open=false")
            return

        latest_time = data.index[-1]
        latest_price = float(data["Close"].iloc[-1])

        print(f"Latest Gold observation: {latest_time}")
        print(f"Latest Gold price: {latest_price}")

        # Yahoo returned a current intraday Gold observation.
        print("market_open=true")

    except Exception as exc:
        print(f"Market-status check failed: {exc}")

        # Fail closed.
        # We do NOT want a failed market check to trigger a data update.
        print("market_open=false")


if __name__ == "__main__":
    main()
