import json
import random

import pandas as pd

from pool import POOL

SEED = 42
N_HELD_OUT_ITEMS = 2   # items the model never trains on (generalization check)
VAL_DAYS = 5
TEST_DAYS = 5
GAP_DAYS = 1           # unused buffer between sections, so windows can't straddle a boundary

# --- Load every item's history ---
frames = {}
for tag in POOL:
    df = pd.read_csv(f"history/{tag}.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    frames[tag] = df.sort_values("timestamp")

# --- Time cutoffs (use the period every item covers) ---
start = max(df["timestamp"].min() for df in frames.values())
end = min(df["timestamp"].max() for df in frames.values())

test_start = end - pd.Timedelta(days=TEST_DAYS)
val_end = test_start - pd.Timedelta(days=GAP_DAYS)
val_start = val_end - pd.Timedelta(days=VAL_DAYS)
train_end = val_start - pd.Timedelta(days=GAP_DAYS)

# --- Pick the held-out items (fixed seed, so it's reproducible) ---
held_out = sorted(random.Random(SEED).sample(POOL, N_HELD_OUT_ITEMS))
train_items = [t for t in POOL if t not in held_out]

# --- Quick per-item summary ---
print("Item                  median price   % slots with no price change   role")
for tag in POOL:
    df = frames[tag]
    unchanged = ((df["buy"].diff() == 0) & (df["sell"].diff() == 0)).mean() * 100
    role = "HELD OUT" if tag in held_out else "train pool"
    print(f"{tag:<22}{df['buy'].median():>12,.1f}{unchanged:>22.1f}%   {role}")

print(f"\nData covers:  {start}  ->  {end}")
print(f"Train:        {start}  ->  {train_end}")
print(f"Validation:   {val_start}  ->  {val_end}")
print(f"Test:         {test_start}  ->  {end}")

splits = {
    "seed": SEED,
    "train_items": train_items,
    "held_out_items": held_out,
    "train_period": [str(start), str(train_end)],
    "validation_period": [str(val_start), str(val_end)],
    "test_period": [str(test_start), str(end)],
}
with open("splits.json", "w") as f:
    json.dump(splits, f, indent=2)
print("\nSaved splits.json")