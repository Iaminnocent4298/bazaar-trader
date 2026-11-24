import pandas as pd

TAX = 0.0125            # 1.25% sell tax (lower with community upgrades)
MIN_WEEKLY_VOLUME = 1_000_000

df = pd.read_csv("data/bazaar_snapshots.csv", parse_dates=["timestamp"])
latest = df[df["timestamp"] == df["timestamp"].max()].copy()

# Estimated flip: buy via order near sell_price, sell via order near buy_price
latest["profit_per_item"] = latest["buy_price"] * (1 - TAX) - latest["sell_price"]
latest["margin_pct"] = 100 * latest["profit_per_item"] / latest["sell_price"]

# Only keep items that actually trade a lot, and have a positive profit
liquid = latest[
    (latest["buy_moving_week"] > MIN_WEEKLY_VOLUME)
    & (latest["sell_moving_week"] > MIN_WEEKLY_VOLUME)
    & (latest["profit_per_item"] > 0)
]

top = liquid.sort_values("margin_pct", ascending=False).head(15)
print(top[["product_id", "sell_price", "buy_price",
           "profit_per_item", "margin_pct", "buy_moving_week"]])