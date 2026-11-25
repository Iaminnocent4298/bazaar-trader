import numpy as np

from features import FEATURE_NAMES
from portfolio_sim import N_STATE
from simulator import TAX, TICK

STATE_NAMES = ["inventory_share", "unrealized_gain", "time_left"]
OBS_NAMES = FEATURE_NAMES + STATE_NAMES
N_OBS = len(OBS_NAMES)
GENOME_LENGTH = 2 * (N_OBS + 1)          # a buy head and a sell head: weights + bias each
assert N_OBS == len(FEATURE_NAMES) + N_STATE


class PortfolioPolicy:
    """Two linear 'heads'. Each outputs a score; the score clipped to [0, 1] is the fraction to use."""

    def __init__(self, genome):
        g = np.asarray(genome, dtype=float)
        self.buy_w, self.buy_b = g[:N_OBS], g[N_OBS]
        self.sell_w, self.sell_b = g[N_OBS + 1:2 * N_OBS + 1], g[2 * N_OBS + 1]

    def act(self, obs, ask, bid):
        buy_frac = float(np.clip(obs @ self.buy_w + self.buy_b, 0.0, 1.0))
        sell_frac = float(np.clip(obs @ self.sell_w + self.sell_b, 0.0, 1.0))
        if (ask - TICK) * (1 - TAX) / (bid + TICK) - 1 <= 0:
            buy_frac = 0.0                # never place a buy the tax makes a sure loss
        return buy_frac, sell_frac


class RuleBaseline:
    """Commit all coins to buy orders whenever the after-tax margin is big enough; always offer everything."""

    def __init__(self, min_margin=0.01):
        self.min_margin = min_margin

    def act(self, obs, ask, bid):
        margin = (ask - TICK) * (1 - TAX) / (bid + TICK) - 1
        return (1.0 if margin >= self.min_margin else 0.0), 1.0


def random_genome(rng):
    return rng.normal(0, 0.5, GENOME_LENGTH)


def mutate(genome, rng, sigma=0.2):
    return genome + rng.normal(0, sigma, GENOME_LENGTH)