import json

import pandas as pd

from sampler import EpisodeSampler
from simulator import Baseline, run_episode

pd.set_option("display.width", 200)

# ---------- Part 1: how noisy is the top of the book inside one 5-minute bucket? ----------
with open("splits.json") as f:
    splits = json.load(f)
start, end = (pd.Timestamp(t) for t in splits["validation_period"])

rows = []
for tag in splits["train_items"] + splits["held_out_items"]:
    df = pd.read_csv(f"history/{tag}.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    rows.append({
        "item": tag,
        "median_spread_%": 100 * ((df["buy"] - df["sell"]) / df["sell"]).median(),
        "median_bid_dip_%": 100 * ((df["sell"] - df["minSell"]) / df["sell"]).median(),
        "median_ask_spike_%": 100 * ((df["maxBuy"] - df["buy"]) / df["buy"]).median(),
        "rows_where_bid_dipped_%": 100 * (df["minSell"] < df["sell"]).mean(),
        "rows_where_ask_spiked_%": 100 * (df["maxBuy"] > df["buy"]).mean(),
    })
print("=== Part 1: spread and within-bucket noise (validation period) ===")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------- Part 2: how fragile are the profits to a stricter fill rule? ----------
print("\n=== Part 2: baseline results as the fill rule gets stricter ===")
print("(fill_margin = how far past the old top price the market must move before we count a fill)\n")
results = []
for fm in [0.0, 0.001, 0.005, 0.01, 0.02]:
    sampler = EpisodeSampler("validation", seed=0, window_hours=(24, 48))  # same episodes each time
    strat = Baseline(min_margin=0.01, max_hold=36)
    n_trades = wins = attempts = fills = 0
    sum_ret = days = 0.0
    for _ in range(200):
        tag, window = sampler.sample()
        trades, a, f = run_episode(window, strat, fill_margin=fm)
        n_trades += len(trades)
        wins += sum(t["return"] > 0 for t in trades)
        sum_ret += sum(t["return"] for t in trades)
        attempts += a
        fills += f
        days += len(window) / 288
    results.append({
        "fill_margin_%": 100 * fm,
        "trades": n_trades,
        "win_rate_%": 100 * wins / n_trades if n_trades else float("nan"),
        "avg_return_per_trade_%": 100 * sum_ret / n_trades if n_trades else float("nan"),
        "return_per_day_%": 100 * sum_ret / days,
        "buy_fill_rate_%": 100 * fills / attempts if attempts else float("nan"),
    })
print(pd.DataFrame(results).round(2).to_string(index=False))