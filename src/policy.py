import numpy as np

from features import FEATURE_NAMES
from simulator import TAX, TICK

N_FEATURES = len(FEATURE_NAMES)
GENOME_LENGTH = N_FEATURES + 2   # weights, a bias, and a maximum holding time
MIN_HOLD, MAX_HOLD = 3, 72       # allowed range for the holding limit (in 5-minute steps)


class LinearPolicy:
    """Enter a flip when (weights . features + bias) > 0; sell instantly after max_hold steps."""

    def __init__(self, genome):
        genome = np.asarray(genome, dtype=float)
        self.weights = genome[:N_FEATURES]
        self.bias = genome[N_FEATURES]
        self.max_hold = int(round(np.clip(genome[N_FEATURES + 1], MIN_HOLD, MAX_HOLD)))

    def buy_price(self, ask, bid, x=None):
        if x is None or np.isnan(x).any():
            return None                       # not enough history yet
        if x @ self.weights + self.bias <= 0:
            return None                       # the model says wait
        buy, sell = bid + TICK, ask - TICK
        if sell * (1 - TAX) / buy - 1 <= 0:
            return None                       # never enter a flip the tax makes a sure loss
        return buy

    def sell_price(self, ask, bid, held_steps, x=None):
        if held_steps >= self.max_hold:
            return None                       # None = sell instantly at the best bid
        return ask - TICK


def random_genome(rng):
    g = np.empty(GENOME_LENGTH)
    g[:N_FEATURES] = rng.normal(0, 1, N_FEATURES)
    g[N_FEATURES] = rng.normal(0, 1)
    g[N_FEATURES + 1] = rng.integers(MIN_HOLD, MAX_HOLD + 1)
    return g


def mutate(genome, rng, sigma=0.3, hold_sigma=6.0):
    child = genome.copy()
    child[:N_FEATURES + 1] += rng.normal(0, sigma, N_FEATURES + 1)
    child[N_FEATURES + 1] = np.clip(child[N_FEATURES + 1] + rng.normal(0, hold_sigma), MIN_HOLD, MAX_HOLD)
    return child