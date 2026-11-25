import glob

import pandas as pd

from pool import POOL

pd.set_option("display.width", 220)
pd.options.display.float_format = "{:,.2f}".format

files = sorted(glob.glob("data/book_2*.csv"))
if not files:
    raise FileNotFoundError("No book_ files in data/ (run git pull?)")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df[df["product_id"].isin(POOL)]
print(f"{df['timestamp'].nunique()} snapshots, {df['product_id'].nunique()} pool items found\n")


def level_coins(g, side, i):
    """Coin value of order-book level i on one side (price x amount), 0 if the level is empty."""
    return (g[f"{side}_{i}_price"] * g[f"{side}_{i}_amount"]).fillna(0)


rows = []
for tag, g in df.groupby("product_id"):
    rows.append({
        "item": tag,
        "snapshots": len(g),
        "median_price": g["ask_1_price"].median(),
        "top_ask_coins": level_coins(g, "ask", 1).median(),
        "top3_ask_coins": sum(level_coins(g, "ask", i) for i in (1, 2, 3)).median(),
        "top_bid_coins": level_coins(g, "bid", 1).median(),
        "top3_bid_coins": sum(level_coins(g, "bid", i) for i in (1, 2, 3)).median(),
        "ask_span_%": (100 * (g["ask_3_price"] - g["ask_1_price"]) / g["ask_1_price"]).median(),
        "bid_span_%": (100 * (g["bid_1_price"] - g["bid_3_price"]) / g["bid_1_price"]).median(),
    })
print(pd.DataFrame(rows).to_string(index=False))