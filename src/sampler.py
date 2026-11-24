import json
import random

import pandas as pd


class EpisodeSampler:
    """Hands out random (item, time window) episodes from one part of the data.

    kind:
      "train"      - training items, training period (rotates the active items)
      "validation" - training items, validation period
      "heldout"    - held-out items, validation period (tests generalization to new items)
      "test"       - held-out period; blocked unless allow_test=True
    """

    def __init__(self, kind="train", seed=0, n_active=5, n_swap=2,
                 rotate_every=50, window_hours=(6, 12, 24, 48), allow_test=False):
        if kind == "test" and not allow_test:
            raise PermissionError("The test period is locked. Pass allow_test=True only for the final evaluation.")

        with open("splits.json") as f:
            splits = json.load(f)

        if kind == "train":
            items, period = splits["train_items"], splits["train_period"]
        elif kind == "validation":
            items, period = splits["train_items"], splits["validation_period"]
        elif kind == "heldout":
            items, period = splits["held_out_items"], splits["validation_period"]
        elif kind == "test":
            items, period = splits["train_items"] + splits["held_out_items"], splits["test_period"]
        else:
            raise ValueError(f"Unknown kind: {kind}")

        self.rng = random.Random(seed)
        self.start, self.end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
        self.window_hours = window_hours
        self.rotate_every = rotate_every
        self.n_swap = n_swap
        self.episodes_given = 0

        self.data = {}
        for tag in items:
            df = pd.read_csv(f"history/{tag}.csv")
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
            df = df[(df["timestamp"] >= self.start) & (df["timestamp"] <= self.end)]
            self.data[tag] = df.sort_values("timestamp").reset_index(drop=True)

        n_active = min(n_active, len(items))
        self.rotating = (kind == "train")
        self.active = self.rng.sample(items, n_active) if self.rotating else list(items)
        self.all_items = list(items)

    def _rotate(self):
        inactive = [t for t in self.all_items if t not in self.active]
        n = min(self.n_swap, len(inactive), len(self.active))
        if n == 0:
            return
        leaving = self.rng.sample(self.active, n)
        joining = self.rng.sample(inactive, n)
        self.active = [t for t in self.active if t not in leaving] + joining
        print(f"[rotation] out: {leaving}  in: {joining}")

    def sample(self):
        """Return (item_tag, window_dataframe) for one episode."""
        if self.rotating and self.episodes_given > 0 and self.episodes_given % self.rotate_every == 0:
            self._rotate()
        self.episodes_given += 1

        for _ in range(100):  # retry if a window has too much missing data
            tag = self.rng.choice(self.active)
            hours = self.rng.choice(self.window_hours)
            span = pd.Timedelta(hours=hours)
            latest_start = self.end - span
            if latest_start <= self.start:
                continue  # this window length doesn't fit in the period
            offset = (latest_start - self.start) * self.rng.random()
            w_start = self.start + offset
            df = self.data[tag]
            window = df[(df["timestamp"] >= w_start) & (df["timestamp"] < w_start + span)]
            if len(window) >= 0.9 * hours * 12:  # expect 12 five-minute rows per hour
                return tag, window.reset_index(drop=True)
        raise RuntimeError("Could not find a valid window; check window sizes against the period length")


if __name__ == "__main__":
    s = EpisodeSampler("train", seed=1, rotate_every=5)
    for i in range(12):
        tag, w = s.sample()
        print(f"episode {i + 1:>2}: {tag:<22} {len(w):>4} rows  {w['timestamp'].iloc[0]} -> {w['timestamp'].iloc[-1]}")

    v = EpisodeSampler("validation", seed=1)
    print("validation sample:", v.sample()[0])
    try:
        EpisodeSampler("test")
    except PermissionError as e:
        print("test guard works:", e)