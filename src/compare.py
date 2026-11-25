import glob
import json

from evolve import WINDOW_HOURS, fitness, make_batch
from policy import LinearPolicy
from sampler import EpisodeSampler
from simulator import Baseline

TUNE_MARGIN = 0.005          # fill margin used for tuning (same as evolution)
EVAL_MARGINS = [0.005, 0.01]

# ---- 1) Tune the plain baseline (just 2 numbers) on a training batch ----
train_batch = make_batch(EpisodeSampler("train", seed=777, n_active=10, window_hours=WINDOW_HOURS), 120)
best = None
for min_margin in [0.005, 0.01, 0.02, 0.03, 0.05]:
    for max_hold in [12, 36, 72, 144, 288]:
        score, n_trades = fitness(Baseline(min_margin, max_hold), train_batch, TUNE_MARGIN)
        if best is None or score > best[0]:
            best = (score, min_margin, max_hold, n_trades)
print(f"Tuned baseline: min_margin={100 * best[1]:.1f}%, max_hold={best[2]} steps "
      f"(training score {100 * best[0]:.2f}%/day, {best[3]} trades)\n")

strategies = {
    "baseline (original)": Baseline(0.01, 36),
    "baseline (tuned)": Baseline(best[1], best[2]),
}
for path in sorted(glob.glob("champion*.json")):
    with open(path) as f:
        strategies[path.replace(".json", "")] = LinearPolicy(json.load(f)["genome"])

# ---- 2) Score everything on FRESH batches (different random windows than selection used) ----
val_batch = make_batch(EpisodeSampler("validation", seed=999, window_hours=WINDOW_HOURS), 150)
held_batch = make_batch(EpisodeSampler("heldout", seed=999, window_hours=WINDOW_HOURS), 150)

header = f"{'strategy':<24}" + "".join(
    f"{name}@{100 * fm:.1f}%".rjust(13) for name in ("val", "held") for fm in EVAL_MARGINS)
print("Return per day %, fresh batches\n" + header)
for name, strat in strategies.items():
    cells = [100 * fitness(strat, batch, fm)[0] for batch in (val_batch, held_batch) for fm in EVAL_MARGINS]
    print(f"{name:<24}" + "".join(f"{c:>13.2f}" for c in cells))