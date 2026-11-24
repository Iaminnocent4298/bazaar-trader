import csv
import os
from datetime import datetime, timezone

from collect import fetch_snapshot

MIN_WEEKLY_VOLUME = 1_000_000
LEVELS = 3  # how many order-book levels to save per side

# "ask" = what you pay to instantly buy (Hypixel's buy_summary, cheapest first)
# "bid" = what you receive for an instant sell (sell_summary, highest first)
LEVEL_COLUMNS = [
    f"{side}_{i}_{field}"
    for side in ("ask", "bid")
    for i in range(1, LEVELS + 1)
    for field in ("price", "amount")
]

COLUMNS = [
    "timestamp", "api_updated", "product_id",
    "sell_price", "sell_volume", "sell_moving_week", "sell_orders",
    "buy_price", "buy_volume", "buy_moving_week", "buy_orders",
] + LEVEL_COLUMNS


def top_levels(summary):
    """Return [price, amount, price, amount, ...] for the first LEVELS orders."""
    out = []
    for i in range(LEVELS):
        if i < len(summary):
            out += [summary[i]["pricePerUnit"], summary[i]["amount"]]
        else:
            out += ["", ""]  # fewer orders than levels
    return out


def main():
    data = fetch_snapshot()
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    api_updated = data.get("lastUpdated", "")

    rows = []
    for product_id, p in data["products"].items():
        q = p["quick_status"]
        if q["sellMovingWeek"] <= MIN_WEEKLY_VOLUME or q["buyMovingWeek"] <= MIN_WEEKLY_VOLUME:
            continue
        rows.append([
            timestamp, api_updated, product_id,
            round(q["sellPrice"], 4), q["sellVolume"], q["sellMovingWeek"], q["sellOrders"],
            round(q["buyPrice"], 4), q["buyVolume"], q["buyMovingWeek"], q["buyOrders"],
        ] + top_levels(p["buy_summary"]) + top_levels(p["sell_summary"]))

    path = f"data/book_{now:%Y-%m-%d}.csv"
    os.makedirs("data", exist_ok=True)
    file_is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if file_is_new:
            writer.writerow(COLUMNS)
        writer.writerows(rows)

    print(f"Saved {len(rows)} items to {path}")


if __name__ == "__main__":
    main()