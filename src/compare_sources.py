import pandas as pd

from load_data import load_all

ITEMS = ["ENCHANTED_DIAMOND", "ENCHANTED_IRON", "ENCHANTED_GOLD", "BOOSTER_COOKIE"]


def to_utc(series):
    """Make timestamps timezone-aware (UTC) and the same type, so they can be matched."""
    s = pd.to_datetime(series, format="ISO8601")
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    else:
        s = s.dt.tz_convert("UTC")
    return s.astype("datetime64[ns, UTC]")


mine = load_all()
mine["timestamp"] = to_utc(mine["timestamp"])
print(f"Your collector: {mine['timestamp'].nunique()} snapshots\n")

for tag in ITEMS:
    a = mine[mine["product_id"] == tag][["timestamp", "buy_price", "sell_price"]]
    a = a.sort_values("timestamp")
    if a.empty:
        print(f"{tag}: not in your collector's data (probably below the volume filter)\n")
        continue

    b = pd.read_csv(f"history/{tag}.csv")
    b["timestamp"] = to_utc(b["timestamp"])
    b = b[["timestamp", "buy", "sell"]].sort_values("timestamp")

    # For each of my snapshots, find the latest Coflnet row at or before it
    m = pd.merge_asof(a, b, on="timestamp", direction="backward",
                      tolerance=pd.Timedelta("6min")).dropna()
    if m.empty:
        print(f"{tag}: no overlapping time range\n")
        continue

    def median_pct_diff(x, y):
        return ((x - y).abs() / y * 100).median()

    print(f"{tag}: {len(m)} matched rows")
    print(f"  my buy_price  vs their buy : {median_pct_diff(m['buy_price'], m['buy']):.2f}% median difference")
    print(f"  my sell_price vs their sell: {median_pct_diff(m['sell_price'], m['sell']):.2f}% median difference")
    print(f"  my buy_price  vs their sell: {median_pct_diff(m['buy_price'], m['sell']):.2f}%  (swap check)")
    print(f"  my sell_price vs their buy : {median_pct_diff(m['sell_price'], m['buy']):.2f}%  (swap check)\n")