import pandas as pd

from sampler import EpisodeSampler
from simulator import Baseline, run_episode

N_EPISODES = 200
WINDOW_HOURS = (24, 48)


def evaluate(kind, strategy, n_episodes=N_EPISODES, seed=0):
    sampler = EpisodeSampler(kind, seed=seed, window_hours=WINDOW_HOURS)
    records = []
    for _ in range(n_episodes):
        tag, window = sampler.sample()
        trades, attempts, fills = run_episode(window, strategy)
        days = len(window) / 288  # 288 five-minute rows per day
        records.append({
            "item": tag,
            "days": days,
            "trades": len(trades),
            "wins": sum(t["return"] > 0 for t in trades),
            "sum_return": sum(t["return"] for t in trades),
            "attempts": attempts,
            "fills": fills,
        })
    return pd.DataFrame(records)


def summarize(df):
    g = df.groupby("item").agg(
        episodes=("days", "size"), days=("days", "sum"), trades=("trades", "sum"),
        wins=("wins", "sum"), sum_return=("sum_return", "sum"),
        attempts=("attempts", "sum"), fills=("fills", "sum"),
    )
    out = pd.DataFrame({
        "episodes": g["episodes"],
        "trades": g["trades"],
        "win_rate_%": 100 * g["wins"] / g["trades"].where(g["trades"] > 0),
        "avg_return_per_trade_%": 100 * g["sum_return"] / g["trades"].where(g["trades"] > 0),
        "return_per_day_%": 100 * g["sum_return"] / g["days"],
        "buy_fill_rate_%": 100 * g["fills"] / g["attempts"].where(g["attempts"] > 0),
    })
    return out.round(2)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    strategy = Baseline(min_margin=0.01, max_hold=36)
    for kind in ["validation", "heldout"]:
        print(f"=== Baseline on {kind} ===")
        result = evaluate(kind, strategy)
        print(summarize(result).to_string())
        total_days = result["days"].sum()
        print(f"Overall return per day: {100 * result['sum_return'].sum() / total_days:.2f}%\n")