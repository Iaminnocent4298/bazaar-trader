import csv
import os
from datetime import datetime, timezone

from collect import COLUMNS, fetch_snapshot, snapshot_to_rows

MIN_WEEKLY_VOLUME = 1_000_000


def main():
    data = fetch_snapshot()
    rows = snapshot_to_rows(data)

    # Keep only items that trade a lot on both sides
    sell_week = COLUMNS.index("sell_moving_week")
    buy_week = COLUMNS.index("buy_moving_week")
    rows = [r for r in rows
            if r[sell_week] > MIN_WEEKLY_VOLUME and r[buy_week] > MIN_WEEKLY_VOLUME]

    # One file per day, e.g. data/bazaar_2026-11-23.csv
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = f"data/bazaar_{today}.csv"
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