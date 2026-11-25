"""Unit-free features, so one model can handle items priced at 500 coins or 15 million.

Every feature at row i uses only rows up to and including i (no peeking at the future).
The first LOOKBACK rows are NaN (not enough history yet), so decisions should wait until then.
"""

import numpy as np
import pandas as pd

LOOKBACK = 12  # 12 five-minute rows = 1 hour

FEATURE_NAMES = [
    "spread_pct",       # (best ask - best bid) / best bid
    "bid_return_1h",    # change in best bid over the last hour
    "volatility_1h",    # std of 5-minute changes in the midpoint over the last hour
    "mid_vs_avg_1h",    # midpoint compared with its own 1-hour average
    "bid_dip_freq_1h",  # share of recent rows where the best bid dipped inside the bucket
    "ask_spike_freq_1h",  # share of recent rows where the best ask spiked inside the bucket
    "imbalance_change_1h",  # change over the last hour in log(resting buy volume / resting sell volume)
]


def compute_features(window):
    ask = window["buy"]
    bid = window["sell"]
    mid = (ask + bid) / 2

    out = pd.DataFrame(index=window.index)
    out["spread_pct"] = (ask - bid) / bid
    out["bid_return_1h"] = bid / bid.shift(LOOKBACK) - 1
    out["volatility_1h"] = mid.pct_change().rolling(LOOKBACK).std()
    out["mid_vs_avg_1h"] = mid / mid.rolling(LOOKBACK).mean() - 1
    out["bid_dip_freq_1h"] = (window["minSell"] < bid).astype(float).rolling(LOOKBACK).mean()
    out["ask_spike_freq_1h"] = (window["maxBuy"] > ask).astype(float).rolling(LOOKBACK).mean()
    imbalance = np.log((window["buyVolume"] + 1) / (window["sellVolume"] + 1))
    out["imbalance_change_1h"] = imbalance - imbalance.shift(LOOKBACK)
    return out[FEATURE_NAMES]


if __name__ == "__main__":
    from sampler import EpisodeSampler

    tag, window = EpisodeSampler("validation", seed=0).sample()
    f = compute_features(window)
    print(tag, "-", len(window), "rows")
    print("NaN rows at the start:", int(f.isna().any(axis=1).sum()))
    print(f.describe().loc[["mean", "std", "min", "max"]].round(4).T)