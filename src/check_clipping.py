import json

import pandas as pd

from features import compute_features
from scaling import CLIP, scale_features

pd.set_option("display.width", 200)

with open("splits.json") as f:
    splits = json.load(f)


def scaled_rows(period, items):
    start, end = (pd.Timestamp(t) for t in period)
    pieces = []
    for tag in items:
        df = pd.read_csv(f"history/{tag}.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].sort_values("timestamp")
        pieces.append(compute_features(df.reset_index(drop=True)).dropna())
    return scale_features(pd.concat(pieces, ignore_index=True))


groups = [
    ("train", splits["train_period"], splits["train_items"]),
    ("validation", splits["validation_period"], splits["train_items"]),
    ("held-out items", splits["validation_period"], splits["held_out_items"]),
]

result = {}
for label, period, items in groups:
    scaled = scaled_rows(period, items)
    result[label] = (scaled.abs() >= CLIP - 1e-9).mean() * 100

print(f"% of rows clipped at +/-{CLIP}:\n")
print(pd.DataFrame(result).round(2).to_string())