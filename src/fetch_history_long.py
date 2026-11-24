import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from fetch_history import fetch_history  # reuses our retry logic

OUT_DIR = "history"
DAYS = 14
CHUNK_HOURS = 12
PAUSE_SECONDS = 1.0

ITEMS = ["ENCHANTED_DIAMOND", "ENCHANTED_IRON", "ENCHANTED_GOLD", "BOOSTER_COOKIE"]


def download_item(tag, end, days):
    """Walk backwards from `end` in chunks, collecting all rows."""
    earliest = end - timedelta(days=days)
    chunk_end = end
    all_rows = []
    first_chunk = True
    while chunk_end > earliest:
        chunk_start = max(chunk_end - timedelta(hours=CHUNK_HOURS), earliest)
        try:
            all_rows.extend(fetch_history(tag, chunk_start, chunk_end))
        except Exception as e:
            print(f"  chunk ending {chunk_end:%Y-%m-%d %H:%M} failed: {e}")
            if first_chunk:
                return []  # item probably doesn't exist under this name
        first_chunk = False
        chunk_end = chunk_start
        time.sleep(PAUSE_SECONDS)
    return all_rows


def clean(rows, tag):
    """Snap rows to 5-minute slots, keeping one row per slot."""
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values("timestamp")
    df["timestamp"] = df["timestamp"].dt.floor("5min")
    df = df.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    df.insert(0, "product_id", tag)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    end = datetime.now(timezone.utc)

    for tag in ITEMS:
        print(f"{tag}: downloading {DAYS} days...")
        rows = download_item(tag, end, DAYS)
        if not rows:
            print(f"{tag}: no data, skipping")
            continue

        df = clean(rows, tag)
        all_slots = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="5min")
        missing = len(all_slots) - len(df)
        df.to_csv(f"{OUT_DIR}/{tag}.csv", index=False)
        print(f"{tag}: saved {len(df)} rows ({missing} empty 5-minute slots)")


if __name__ == "__main__":
    main()