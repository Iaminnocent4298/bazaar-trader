import contextlib
import io
import json
import sys

import numpy as np

from features import compute_features
from portfolio_policy import OBS_NAMES, PortfolioPolicy, RuleBaseline, mutate, random_genome
from portfolio_sim import INITIAL_COINS, simulate
from sampler import EpisodeSampler
from scaling import scale_features

# ---- Settings you can change ----
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
POPULATION = 40
GENERATIONS = 60
DISCARD_FRACTION = 0.5      # the worst half is discarded after every generation
N_IMMIGRANTS = 2            # brand-new random models added each generation
GENERATION_HOURS = 48       # one generation = one item over one window this long
FILL_MARGIN = 0.005         # pessimistic fill rule used for selection
N_CHECK_EPISODES = 20       # fixed episodes used to monitor validation / held-out items
INACTIVITY_PENALTY_PCT = 1.0   # % of starting coins subtracted (for RANKING only) from idle models
MIN_BUY_FILLS = 1              # a model is "idle" if fewer than this many of its buy orders ever filled
PENALTY = INITIAL_COINS * INACTIVITY_PENALTY_PCT / 100


def draw(sampler):
    """One (item, window, scaled features) episode; silences the sampler's own rotation message."""
    with contextlib.redirect_stdout(io.StringIO()):
        tag, window = sampler.sample()
    return tag, window, scale_features(compute_features(window)).to_numpy()


def score(result):
    """Ranking score: raw profit, minus the inactivity penalty if the model never got a buy filled."""
    return result["profit"] - (PENALTY if result["buy_fills"] < MIN_BUY_FILLS else 0.0)


def evaluate(strategy, episodes, fill_margin):
    results = [simulate(w, strategy, f, fill_margin) for _, w, f in episodes]
    profits = np.array([r["profit"] for r in results])
    return {
        "mean_profit": float(profits.mean()),                       # raw coins, what you see in the tables
        "mean_score": float(np.mean([score(r) for r in results])),  # used for picking the champion
        "profitable": int((profits > 0).sum()),
        "idle": sum(r["buy_fills"] < MIN_BUY_FILLS for r in results),
    }


def main():
    rng = np.random.default_rng(SEED)
    # One item at a time: n_active=1 and rotate_every=1 swaps in a different item every generation
    train = EpisodeSampler("train", seed=SEED, n_active=1, n_swap=1, rotate_every=1,
                           window_hours=(GENERATION_HOURS,))
    val_sampler = EpisodeSampler("validation", seed=123, window_hours=(GENERATION_HOURS,))
    val_eps = [draw(val_sampler) for _ in range(N_CHECK_EPISODES)]
    baseline = RuleBaseline(min_margin=0.01)
    base_val = evaluate(baseline, val_eps, FILL_MARGIN)["mean_profit"]

    population = [random_genome(rng) for _ in range(POPULATION)]
    per_generation_best = []   # (validation mean score, genome)

    print(f"Start: {INITIAL_COINS:,} coins | fill margin {100 * FILL_MARGIN:.1f}% | "
          f"{GENERATION_HOURS}h per generation | doing nothing = 0 profit")
    print(f"Inactivity penalty: {PENALTY:,.0f} coins ({INACTIVITY_PENALTY_PCT}% of start) for any model with "
          f"fewer than {MIN_BUY_FILLS} filled buy(s). It affects ranking only; all profits shown are raw coins.")
    print(f"Baseline rule, mean profit per validation episode: {base_val:,.0f}\n")
    print(f"{'gen':>3}  {'item':<20}{'best':>14}{'median':>14}{'worst':>14}"
          f"{'profitable':>12}{'idle':>7}{'baseline':>14}{'buys':>6}{'sells':>6}{'best_val_mean':>16}")

    for gen in range(1, GENERATIONS + 1):
        tag, window, feats = draw(train)                   # a fresh item and window each generation
        results = [simulate(window, PortfolioPolicy(g), feats, FILL_MARGIN) for g in population]
        profits = np.array([r["profit"] for r in results])           # raw coins gained or lost
        fits = np.array([score(r) for r in results])                 # raw profit minus inactivity penalty
        n_idle = sum(r["buy_fills"] < MIN_BUY_FILLS for r in results)
        base_profit = simulate(window, baseline, feats, FILL_MARGIN)["profit"]

        order = np.argsort(-fits)                          # best ranking score first
        best = population[order[0]]
        best_res = results[order[0]]
        val = evaluate(PortfolioPolicy(best), val_eps, FILL_MARGIN)   # monitoring only
        per_generation_best.append((val["mean_score"], best.copy()))

        print(f"{gen:>3}  {tag:<20}{profits[order[0]]:>14,.0f}{np.median(profits):>14,.0f}"
              f"{profits.min():>14,.0f}{int((profits > 0).sum()):>9}/{POPULATION}{n_idle:>7}"
              f"{base_profit:>14,.0f}{best_res['buy_fills']:>6}{best_res['sell_fills']:>6}"
              f"{val['mean_profit']:>16,.0f}")

        if fits.max() == fits.min():                       # every model tied: nothing to learn from
            continue                                       # so nobody is discarded this generation

        # ---- discard the worst, refill with mutated copies of survivors + a few newcomers ----
        n_survivors = int(round(POPULATION * (1 - DISCARD_FRACTION)))
        survivors = [population[i] for i in order[:n_survivors]]
        n_children = POPULATION - n_survivors - N_IMMIGRANTS
        children = [mutate(survivors[rng.integers(n_survivors)], rng) for _ in range(n_children)]
        immigrants = [random_genome(rng) for _ in range(N_IMMIGRANTS)]
        population = survivors + children + immigrants

    champ_val, champ = max(per_generation_best, key=lambda t: t[0])
    policy = PortfolioPolicy(champ)
    print("\nChampion (picked by validation mean score, penalty included):")
    print(f"  {'input':<22}{'buy weight':>12}{'sell weight':>13}")
    for name, wb, ws in zip(OBS_NAMES, policy.buy_w, policy.sell_w):
        print(f"  {name:<22}{wb:>12.2f}{ws:>13.2f}")
    print(f"  {'bias':<22}{policy.buy_b:>12.2f}{policy.sell_b:>13.2f}")

    held_sampler = EpisodeSampler("heldout", seed=123, window_hours=(GENERATION_HOURS,))
    held = [draw(held_sampler) for _ in range(N_CHECK_EPISODES)]
    print(f"\nMean raw profit per {GENERATION_HOURS}h episode (coins), champion vs baseline "
          f"[profitable episodes | idle episodes, out of {N_CHECK_EPISODES}]")
    print(f"{'fill_margin%':>12}{'val_champ':>24}{'val_base':>24}{'heldout_champ':>24}{'heldout_base':>24}")
    for fm in [0.001, 0.005, 0.01, 0.02]:
        cells = []
        for episodes in (val_eps, held):
            for strat in (policy, baseline):
                r = evaluate(strat, episodes, fm)
                cells.append(f"{r['mean_profit']:>15,.0f} [{r['profitable']:>2}|{r['idle']:>2}]")
        print(f"{100 * fm:>12.1f}" + "".join(f"{c:>24}" for c in cells))

    with open(f"champion2_seed{SEED}.json", "w") as f:
        json.dump({"genome": [float(v) for v in champ], "obs_names": OBS_NAMES,
                   "fill_margin": FILL_MARGIN, "validation_mean_score": champ_val,
                   "inactivity_penalty": PENALTY}, f, indent=2)
    print(f"\nSaved champion2_seed{SEED}.json")


if __name__ == "__main__":
    main()