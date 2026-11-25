"""A deliberately simple, conservative paper-trading simulator for one item and one time window.

Column meanings (Coflnet history):
  buy     = best ask  (price an instant buyer pays)
  sell    = best bid  (price an instant seller receives)
  minSell = lowest best-bid seen during the 5-minute bucket
  maxBuy  = highest best-ask seen during the 5-minute bucket
"""

TAX = 0.0125   # sell tax; the real rate can be lower with upgrades
TICK = 0.1     # smallest price step on the Bazaar

REQUIRED_COLUMNS = ["buy", "sell", "minSell", "maxBuy"]


class Baseline:
    """Rule-based strategy: sit just above the best bid, sell just below the best ask."""

    def __init__(self, min_margin=0.01, max_hold=36):
        self.min_margin = min_margin   # only enter if the after-tax margin is at least this
        self.max_hold = max_hold       # give up and sell instantly after this many 5-min steps

    def buy_price(self, ask, bid, x=None):
        buy = bid + TICK
        sell = ask - TICK
        margin = sell * (1 - TAX) / buy - 1
        return buy if margin >= self.min_margin else None

    def sell_price(self, ask, bid, held_steps, x=None):
        if held_steps >= self.max_hold:
            return None  # None means "sell instantly at the best bid"
        return ask - TICK


def run_episode(window, strategy, fill_margin=0.0, features=None):
    missing = [c for c in REQUIRED_COLUMNS if c not in window.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Available: {list(window.columns)}")

    ask = window["buy"].to_numpy()
    bid = window["sell"].to_numpy()
    min_bid = window["minSell"].to_numpy()
    max_ask = window["maxBuy"].to_numpy()
    n = len(window)

    trades = []
    buy_attempts = 0     # 5-minute steps with a buy order sitting in the market
    buy_fills = 0
    holding = False
    entry_price = None
    entry_step = None
    held_steps = 0

    for i in range(n - 1):
        # Decide using only row i, then judge the fill using row i+1.
        x = None if features is None else features[i]  # scaled features at row i
        if not holding:
            price = strategy.buy_price(ask[i], bid[i], x)
            if price is None:
                continue
            buy_attempts += 1
            # Filled only if the best bid fell below both our price and the old top bid,
            # i.e. the level in front of (or at) our order was cleared.
            if min_bid[i + 1] < min(price, bid[i]) * (1 - fill_margin):
                buy_fills += 1
                holding, entry_price, entry_step, held_steps = True, price, i + 1, 0
        else:
            held_steps += 1
            price = strategy.sell_price(ask[i], bid[i], held_steps, x)
            if price is None:  # forced exit: instant sell at the best bid
                trades.append(_trade(entry_step, i, entry_price, bid[i], "forced"))
                holding = False
            elif max_ask[i + 1] > max(price, ask[i]) * (1 + fill_margin):  # asks above our price were cleared
                trades.append(_trade(entry_step, i + 1, entry_price, price, "order"))
                holding = False

    if holding:  # window ended while holding: liquidate at the last best bid
        trades.append(_trade(entry_step, n - 1, entry_price, bid[-1], "end"))

    return trades, buy_attempts, buy_fills


def _trade(entry_step, exit_step, buy, sell, how):
    ret = sell * (1 - TAX) / buy - 1
    return {"entry": entry_step, "exit": exit_step, "buy": buy, "sell": sell,
            "return": ret, "exit_type": how}