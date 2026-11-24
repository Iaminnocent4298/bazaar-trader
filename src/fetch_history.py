import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

BASE_URL = "https://sky.coflnet.com/api/bazaar"
OUT_DIR = "history"
DAYS = 2            # start small, we'll go bigger once it works
PAUSE_SECONDS = 1.0  # stay well under the rate limit

ITEMS = ["ENCHANTED_DIAMOND", "ENCHANTED_IRON", "ENCHANTED_GOLD", "BOOSTER_COOKIE"]


def fetch_history(tag, start, end):
    """Download history for one item, retrying if we hit the rate limit."""
    url = f"{BASE_URL}/{tag}/history"
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for attempt in range(5):
        response = requests.get(url, params=params, timeout=60)
        if response.status_code == 429:  # "too many requests"
            wait = int(float(response.headers.get("Retry-After", 10)))
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Gave up on {tag} after repeated rate limits")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS)

    for tag in ITEMS:
        try:
            rows = fetch_history(tag, start, end)
        except Exception as e:
            print(f"{tag}: failed ({e})")
            continue

        if not rows:
            print(f"{tag}: no data returned")
            continue

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        df.insert(0, "product_id", tag)
        df.to_csv(f"{OUT_DIR}/{tag}.csv", index=False)

        gap = df["timestamp"].sort_values().diff().median()
        print(f"{tag}: {len(df)} rows, typical gap between rows: {gap}")

        time.sleep(PAUSE_SECONDS)


if __name__ == "__main__":
    main()