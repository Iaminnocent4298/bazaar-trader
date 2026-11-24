import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

TAG = "ENCHANTED_DIAMOND"
BASE = f"https://sky.coflnet.com/api/bazaar/{TAG}"
now = datetime.now(timezone.utc)


def fmt(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def summarize(label, url, params=None):
    response = requests.get(url, params=params, timeout=60)
    if response.status_code != 200:
        print(f"{label}: HTTP {response.status_code}")
        return
    rows = response.json()
    if not rows:
        print(f"{label}: no rows")
        return
    times = pd.to_datetime(pd.DataFrame(rows)["timestamp"], format="ISO8601").sort_values()
    gaps = times.diff()
    print(f"{label}: {len(rows)} rows, typical gap {gaps.median()}, smallest gap {gaps.min()}")


# Custom ranges of different sizes and ages: (label, hours ago it starts, hours ago it ends)
windows = [
    ("custom: last 6 hours", 6, 0),
    ("custom: last 12 hours", 12, 0),
    ("custom: 6 hours, ending 7 days ago", 174, 168),
    ("custom: 6 hours, ending 30 days ago", 726, 720),
    ("custom: 1 hour, ending 30 days ago", 721, 720),
]
for label, start_h, end_h in windows:
    params = {
        "start": fmt(now - timedelta(hours=start_h)),
        "end": fmt(now - timedelta(hours=end_h)),
    }
    summarize(label, f"{BASE}/history", params)
    time.sleep(1)

# The fixed-window endpoints from the docs
for name in ["hour", "day", "week"]:
    summarize(f"/history/{name}", f"{BASE}/history/{name}")
    time.sleep(1)