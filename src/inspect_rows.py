from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

TAG = "ENCHANTED_DIAMOND"
now = datetime.now(timezone.utc)
params = {
    "start": (now - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "end": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
}
rows = requests.get(
    f"https://sky.coflnet.com/api/bazaar/{TAG}/history", params=params, timeout=60
).json()

df = pd.DataFrame(rows)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
df = df.sort_values("timestamp").reset_index(drop=True)
df["gap"] = df["timestamp"].diff()

print(df[["timestamp", "buy", "sell", "gap"]].head(20))
print("Gaps under 60 seconds:", (df["gap"] < pd.Timedelta(seconds=60)).sum())
print("Gaps over 60 seconds: ", (df["gap"] > pd.Timedelta(seconds=60)).sum())