import pandas as pd

from load_data import load_all


def to_utc(series):
    s = pd.to_datetime(series, format="ISO8601")
    s = s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
    return s.astype("datetime64[ns, UTC]")


TAG = "ENCHANTED_GOLD"

mine = load_all()
mine["timestamp"] = to_utc(mine["timestamp"])
a = mine[mine["product_id"] == TAG].sort_values("timestamp")

b = pd.read_csv(f"history/{TAG}.csv")
b["timestamp"] = to_utc(b["timestamp"])
b = b.sort_values("timestamp")

m = pd.merge_asof(
    a, b, on="timestamp", direction="backward", tolerance=pd.Timedelta("6min")
).dropna(subset=["buy"])

m["buy_diff_pct"] = (m["buy_price"] - m["buy"]) / m["buy"] * 100

pd.set_option("display.width", 200)
print(m[["timestamp", "buy_price", "buy", "buy_diff_pct"]].to_string(index=False))

print("\nVolume columns (median % difference):")
pairs = [
    ("buy_volume", "buyVolume"),
    ("sell_volume", "sellVolume"),
    ("buy_moving_week", "buyMovingWeek"),
    ("sell_moving_week", "sellMovingWeek"),
]
for mine_col, their_col in pairs:
    diff = ((m[mine_col] - m[their_col]).abs() / m[their_col] * 100).median()
    print(f"  {mine_col} vs {their_col}: {diff:.2f}%")