import glob

import pandas as pd

TAX = 0.0125
MIN_WEEKLY_VOLUME = 1_000_000   # on both sides
PER_TIER = 8                    # candidates to show per price tier

files = sorted(glob.glob("data/book_2*.csv"))
if not files:
    raise FileNotFoundError("No book_ files found in data/ (run git pull?)")

df = pd.read_csv(files[-1])
latest = df[df["timestamp"] == df["timestamp"].max()].copy()

# Drop items with an empty side of the book
latest = latest.dropna(subset=["ask_1_price", "bid_1_price"])

# Theoretical flip: place a buy order near the best bid, sell near the best ask
latest["profit_per_item"] = latest["ask_1_price"] * (1 - TAX) - latest["bid_1_price"]
latest["margin_pct"] = 100 * latest["profit_per_item"] / latest["bid_1_price"]
latest["weekly_volume"] = latest[["buy_moving_week", "sell_moving_week"]].min(axis=1)

candidates = latest[
    (latest["weekly_volume"] >= MIN_WEEKLY_VOLUME) & (latest["margin_pct"] > 0)
].copy()

# Group by price level so the pool isn't all cheap items (or all expensive ones)
candidates["tier"] = pd.cut(
    candidates["bid_1_price"],
    bins=[0, 100, 5_000, float("inf")],
    labels=["cheap (<100)", "mid (100-5k)", "expensive (>5k)"],
)

pd.set_option("display.width", 200)
print(f"Snapshot: {latest['timestamp'].iloc[0]}")
print(f"{len(candidates)} items pass the volume and margin filters\n")

for tier, group in candidates.groupby("tier", observed=True):
    top = group.sort_values("weekly_volume", ascending=False).head(PER_TIER)
    print(f"--- {tier} ---")
    print(top[["product_id", "bid_1_price", "ask_1_price",
               "margin_pct", "weekly_volume"]].round(2).to_string(index=False))
    print()