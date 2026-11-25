"""Scale features using numbers measured on the TRAINING period only (so nothing leaks from validation/test)."""

import json

import pandas as pd
import numpy as np

from features import FEATURE_NAMES, compute_features

SCALING_FILE = "scaling.json"
CLIP = 5.0  # after scaling, cap values at +/- this many "scale units"


def fit_scaling():
    with open("splits.json") as f:
        splits = json.load(f)
    start, end = (pd.Timestamp(t) for t in splits["train_period"])

    pieces = []
    for tag in splits["train_items"]:
        df = pd.read_csv(f"history/{tag}.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].sort_values("timestamp")
        pieces.append(compute_features(df.reset_index(drop=True)).dropna())
    pooled = pd.concat(pieces, ignore_index=True)

    stats = {}
    for name in FEATURE_NAMES:
        col = pooled[name]
        center = float(col.median())
        scale = float(col.quantile(0.75) - col.quantile(0.25))  # interquartile range
        if scale < 1e-9:
            scale = float(col.std()) or 1.0  # fall back if the middle half is constant
        stats[name] = {"center": center, "scale": scale}

    with open(SCALING_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    return stats, len(pooled)


def scale_features(features):
    with open(SCALING_FILE) as f:
        stats = json.load(f)
    out = features.copy()
    for name in FEATURE_NAMES:
        z = (out[name] - stats[name]["center"]) / stats[name]["scale"]
        out[name] = (np.sign(z) * np.log1p(z.abs())).clip(-CLIP, CLIP)  # compress tails smoothly
    return out


if __name__ == "__main__":
    stats, n = fit_scaling()
    print(f"Fitted on {n} training rows (pooled across training items)\n")
    for name, s in stats.items():
        print(f"{name:<22} center = {s['center']:>10.5f}   scale = {s['scale']:>10.5f}")

    from sampler import EpisodeSampler
    from features import compute_features

    tag, window = EpisodeSampler("validation", seed=0).sample()
    scaled = scale_features(compute_features(window)).dropna()
    print(f"\nScaled features for one validation episode ({tag}):")
    print(scaled.describe().loc[["mean", "std", "min", "max"]].round(2).T)