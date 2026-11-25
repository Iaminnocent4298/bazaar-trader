"""Portfolio simulator: one item, one time window, real coins and real inventory.

Every 5-minute step the policy may:
  - commit a fraction of its coins to a BUY order at (best bid + 1 tick)
  - offer a fraction of its items in a SELL order at (best ask - 1 tick)
Orders are re-priced every step (the old ones are cancelled). At the end of the window every
remaining item is sold instantly, so only coins on hand count.
"""

import numpy as np

from simulator import TAX, TICK

INITIAL_COINS = 200_000_000
FLOW_SHARE = 0.10               # share of the market's estimated 5-minute volume one order can capture
LIQUIDATION_SLIPPAGE = 0.01     # penalty for dumping the leftover items instantly at the end
STEPS_PER_WEEK = 7 * 288
N_STATE = 3                     # extra inputs: share of wealth held as items, unrealized gain, time left


def simulate(window, policy, features, fill_margin=0.005, initial_coins=INITIAL_COINS,
             flow_share=FLOW_SHARE, liq_slip=LIQUIDATION_SLIPPAGE):
    ask = window["buy"].to_numpy(float)
    bid = window["sell"].to_numpy(float)
    min_bid = window["minSell"].to_numpy(float)
    max_ask = window["maxBuy"].to_numpy(float)
    # Estimated units traded per 5-minute step, scaled by how much of it we can capture
    flow = (np.minimum(window["buyMovingWeek"].to_numpy(float),
                       window["sellMovingWeek"].to_numpy(float)) / STEPS_PER_WEEK * flow_share)
    n = len(window)

    cash, inv, avg_cost = float(initial_coins), 0, 0.0
    buy_fills = sell_fills = 0

    for i in range(n - 1):
        x = features[i]
        if np.isnan(x).any():
            continue                                  # not enough history yet

        bid_i, ask_i = bid[i], ask[i]
        equity = cash + inv * bid_i
        inv_share = inv * bid_i / equity if equity > 0 else 0.0
        if inv > 0 and avg_cost > 0:
            z = (bid_i / avg_cost - 1) / 0.01         # unrealized gain in 1%-units
            unrealized = np.sign(z) * np.log1p(abs(z))
        else:
            unrealized = 0.0
        time_left = (n - 1 - i) / (n - 1)
        obs = np.concatenate([x, [inv_share, unrealized, time_left]])

        buy_frac, sell_frac = policy.act(obs, ask_i, bid_i)

        # --- build this step's orders (decided with row i only) ---
        buy_price, buy_units = bid_i + TICK, 0
        if buy_frac > 0:
            buy_units = int(cash * buy_frac // buy_price)
        sell_price, sell_units = ask_i - TICK, 0
        if sell_frac > 0 and inv > 0 and sell_price > 0:
            sell_units = min(inv, max(1, int(inv * sell_frac)))

        # --- judge the fills using row i+1 ---
        cap = int(flow[i + 1])                        # most units either order can fill this step
        if buy_units > 0 and min_bid[i + 1] < min(buy_price, bid_i) * (1 - fill_margin):
            filled = min(buy_units, cap)
            if filled > 0:
                cost = filled * buy_price
                avg_cost = (avg_cost * inv + cost) / (inv + filled)
                cash -= cost
                inv += filled
                buy_fills += 1
        if sell_units > 0 and max_ask[i + 1] > max(sell_price, ask_i) * (1 + fill_margin):
            filled = min(sell_units, cap)
            if filled > 0:
                cash += filled * sell_price * (1 - TAX)
                inv -= filled
                if inv == 0:
                    avg_cost = 0.0
                sell_fills += 1

    liquidated = inv
    if inv > 0:                                       # only coins on hand count
        cash += inv * bid[-1] * (1 - liq_slip) * (1 - TAX)
        inv = 0

    return {"profit": cash - initial_coins, "buy_fills": buy_fills,
            "sell_fills": sell_fills, "liquidated_units": liquidated}