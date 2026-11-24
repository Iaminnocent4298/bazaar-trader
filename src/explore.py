import pandas as pd

# Load the CSV into a "DataFrame" (a table)
df = pd.read_csv("data/bazaar_snapshots.csv", parse_dates=["timestamp"])

# 1. How big is it?
print("Rows and columns:", df.shape)

# 2. First few rows
print(df.head())

# 3. Look at one item over time
item = df[df["product_id"] == "ENCHANTED_DIAMOND"]
print(item[["timestamp", "buy_price", "sell_price"]])

# 4. Add a new column: the spread (gap between the two prices)
df["spread"] = df["buy_price"] - df["sell_price"]

# 5. Which items have the biggest spread right now?
latest = df[df["timestamp"] == df["timestamp"].max()]
top = latest.sort_values("spread", ascending=False).head(10)
print(top[["product_id", "buy_price", "sell_price", "spread"]])