import csv
import os
import time
from datetime import datetime, timezone

import requests

URL = "https://api.hypixel.net/v2/skyblock/bazaar"
OUTPUT_FILE = "data/bazaar_snapshots.csv"
INTERVAL_SECONDS = 30   # how long to wait between fetches
NUM_FETCHES = 5         # how many times to fetch (small, for testing)

COLUMNS = [
    "timestamp", "product_id",
    "sell_price", "sell_volume", "sell_moving_week", "sell_orders",
    "buy_price", "buy_volume", "buy_moving_week", "buy_orders",
]


def fetch_snapshot():
    """Ask Hypixel for the current Bazaar data and return it as a dictionary."""
    response = requests.get(URL, timeout=30)
    response.raise_for_status()  # raises an error if the request failed
    return response.json()


def snapshot_to_rows(data):
    """Turn the raw data into a list of rows, one per item."""
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for product_id, product in data["products"].items():
        q = product["quick_status"]
        rows.append([
            timestamp, product_id,
            q["sellPrice"], q["sellVolume"], q["sellMovingWeek"], q["sellOrders"],
            q["buyPrice"], q["buyVolume"], q["buyMovingWeek"], q["buyOrders"],
        ])
    return rows


def save_rows(rows):
    """Append rows to the CSV file, writing the header if the file is new."""
    os.makedirs("data", exist_ok=True)
    file_is_new = not os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if file_is_new:
            writer.writerow(COLUMNS)
        writer.writerows(rows)


def main():
    for i in range(NUM_FETCHES):
        try:
            data = fetch_snapshot()
            rows = snapshot_to_rows(data)
            save_rows(rows)
            print(f"Fetch {i + 1}/{NUM_FETCHES}: saved {len(rows)} items")
        except Exception as e:
            print(f"Fetch {i + 1} failed: {e}")
        if i < NUM_FETCHES - 1:
            time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()