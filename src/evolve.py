import json
import sys

import numpy as np

from features import FEATURE_NAMES, compute_features
from policy import LinearPolicy, mutate, random_genome
from sampler import EpisodeSampler
from scaling import scale_features
from simulator import Baseline, run_episode

# ---- Settings you can change ----
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
POPULATION = 40
GENERATIONS = 30
DISCARD_FRACTION = 0.5     # the worst half is discarded after every generation
N_IMMIGRANTS = 2           # brand-new random models added each generation (keeps variety up)
EPISODES_PER_GEN = 40      # episodes every model is scored on, per generation
FILL_MARGIN = 0.005        # pessimistic fill rule used for selection
WINDOW_HOURS = (24, 48)
N_VALIDATION_EPISODES = 60


def make_batch(sampler, n):
    """Draw n episodes and precompute their scaled features (done once, shared by all models)."""
    batch = []
    for _ in range(n):
        tag, window = sampler.sample()
        feats = scale_features(compute_features(window)).to_numpy()
        batch.append((window, feats, len(window) / 288))
    return batch


def fitness(strategy, batch, fill_margin):
    """Average return per day across the batch (a no-trade model scores 0)."""
    total, n_trades = 0.0, 0
    for window, feats, days in batch:
        trades, _, _ = run_episode(window, strategy, fill_margin, feats)
        total += sum(t["return"] for t in trades) / days
        n_trades += len(trades)
    return total / len(batch), n_trades


def main():
    rng = np.random.default_rng(SEED)
    # rotate_every = EPISODES_PER_GEN, so the active items change once per generation
    train = EpisodeSampler("train", seed=SEED, n_active=5, n_swap=2,
                           rotate_every=EPISODES_PER_GEN, window_hours=WINDOW_HOURS)
    val_batch = make_batch(EpisodeSampler("validation", seed=123, window_hours=WINDOW_HOURS),
                           N_VALIDATION_EPISODES)
    baseline = Baseline(min_margin=0.01, max_hold=36)
    base_val, _ = fitness(baseline, val_batch, FILL_MARGIN)

    population = [random_genome(rng) for _ in range(POPULATION)]
    per_generation_best = []   # (validation score, genome)

    print(f"Fill margin {100 * FILL_MARGIN:.1f}% | baseline on validation: {100 * base_val:.2f}%/day\n")
    print("gen  best_train%  median_train%  baseline_train%  best_trades  best_validation%")

    for gen in range(1, GENERATIONS + 1):
        batch = make_batch(train, EPISODES_PER_GEN)       # a fresh batch every generation
        scores = [fitness(LinearPolicy(g), batch, FILL_MARGIN) for g in population]
        fits = np.array([s[0] for s in scores])
        base_train, _ = fitness(baseline, batch, FILL_MARGIN)

        order = np.argsort(-fits)                          # best first
        best = population[order[0]]
        val_score, _ = fitness(LinearPolicy(best), val_batch, FILL_MARGIN)  # monitoring only
        per_generation_best.append((val_score, best.copy()))

        print(f"{gen:>3}  {100 * fits[order[0]]:>11.2f}  {100 * np.median(fits):>13.2f}  "
              f"{100 * base_train:>15.2f}  {scores[order[0]][1]:>11}  {100 * val_score:>16.2f}")

        # ---- discard the worst, refill with mutated copies of survivors + a few newcomers ----
        n_survivors = int(round(POPULATION * (1 - DISCARD_FRACTION)))
        survivors = [population[i] for i in order[:n_survivors]]
        n_children = POPULATION - n_survivors - N_IMMIGRANTS
        children = [mutate(survivors[rng.integers(n_survivors)], rng) for _ in range(n_children)]
        immigrants = [random_genome(rng) for _ in range(N_IMMIGRANTS)]
        population = survivors + children + immigrants

    # Champion = the generation-best model that did best on validation
    champ_val, champ = max(per_generation_best, key=lambda t: t[0])
    policy = LinearPolicy(champ)
    print("\nChampion (picked by validation score):")
    for name, w in zip(FEATURE_NAMES, policy.weights):
        print(f"  weight {name:<22} {w:>7.2f}")
    print(f"  bias {policy.bias:.2f} | max hold {policy.max_hold} steps ({policy.max_hold * 5} min)")

    # Robustness: champion vs baseline across several fill margins, never used for selection
    held = make_batch(EpisodeSampler("heldout", seed=123, window_hours=WINDOW_HOURS), N_VALIDATION_EPISODES)
    print("\nReturn per day %, champion vs baseline")
    print("fill_margin%   val_champ  val_base   heldout_champ  heldout_base")
    for fm in [0.001, 0.005, 0.01, 0.02]:
        row = [fitness(s, b, fm)[0] * 100
               for b in (val_batch, held) for s in (policy, baseline)]
        print(f"{100 * fm:>11.1f}   {row[0]:>9.2f}  {row[1]:>8.2f}   {row[2]:>13.2f}  {row[3]:>12.2f}")

    with open(f"champion_seed{SEED}.json", "w") as f:
        json.dump({"genome": [float(v) for v in champ], "feature_names": FEATURE_NAMES,
                   "fill_margin": FILL_MARGIN, "validation_score": float(champ_val)}, f, indent=2)
    print(f"\nSaved champion_seed{SEED}.json")


if __name__ == "__main__":
    main()