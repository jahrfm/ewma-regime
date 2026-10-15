"""Long-only regime filter, evaluated on the bar after the label is known.

Buy and hold is compounded on that same set of bars. Strategy hit rate counts
only bars that were actually long. Turnover pays `fee_rate` of equity, and a
position still open at the end pays the exit fee as well.
"""

from ewma_regime import annualized_sharpe, compute_ewma_regime, iter_forward_pairs

# Bybit linear USDT perpetual taker fee is about 5 bps per side.
DEFAULT_FEE_RATE = 0.0005


def backtest_strategy(result, regime_filter, fee_rate=DEFAULT_FEE_RATE, eval_start=None):
    """Compounded performance of being long only when the prior label is allowed.

    regime_filter lists allowed regimes, for example [0, 1] to be long in
    Low or Medium and flat in High.
    """
    _validate_filter(regime_filter)
    _validate_fee(fee_rate)
    simple_returns = []
    active_returns = []
    prev_pos = 0
    entries = 0
    for _, label, forward in iter_forward_pairs(result, eval_start):
        pos = 1 if label in regime_filter else 0
        turnover = abs(pos - prev_pos)
        if prev_pos == 0 and pos == 1:
            entries += 1
        simple_returns.append((1.0 - turnover * fee_rate) * (1.0 + pos * forward) - 1.0)
        if pos == 1:
            active_returns.append(forward)
        prev_pos = pos
    if prev_pos == 1 and fee_rate and simple_returns:
        growth = 1.0 + simple_returns[-1]
        simple_returns[-1] = growth * (1.0 - fee_rate) - 1.0
    return _pack_metrics(simple_returns, active_returns, entries, result.get("interval"))


def buy_and_hold(result, eval_start=None):
    """Compounded price return on the same forward bars the strategy is scored on."""
    forwards = [forward for _, _, forward in iter_forward_pairs(result, eval_start)]
    return _pack_metrics(forwards, forwards, entries=0, interval=result.get("interval"))


def print_strategy_results(symbol, interval, results_dict, fee_rate):
    print(f"\n=== Strategy backtest: {symbol} {interval} ===")
    print(
        "Position on bar t uses the label from bar t - 1. "
        f"Strategy figures are net of {fee_rate:.2%} per side, including an exit if still long. "
        "Buy & hold is the raw price return on the same bars."
    )
    print(
        f"{'Strategy':<16} {'Return':>10} {'Hit rate':>10} {'Exposure':>10} "
        f"{'Trades':>8} {'Max DD':>10} {'Sharpe':>8} {'Bars':>6}"
    )
    for name, res in results_dict.items():
        print(
            f"{name:<16} {_format_pct(res['total_return']):>10} "
            f"{_format_pct(res['hit_rate']):>10} {_format_pct(res['exposure']):>10} "
            f"{res['trades']:>8} {_format_pct(res['max_drawdown']):>10} "
            f"{_format_num(res['sharpe'], 8, '.2f')} {res['bars']:>6}"
        )


def _pack_metrics(simple_returns, active_returns, entries, interval):
    bars = len(simple_returns)
    if bars == 0:
        return {
            "total_return": None,
            "hit_rate": None,
            "exposure": None,
            "trades": 0,
            "max_drawdown": None,
            "sharpe": None,
            "bars": 0,
            "simple_returns": [],
        }
    total_return, max_drawdown = _equity_path(simple_returns)
    hit_rate = None
    if active_returns:
        hit_rate = sum(1 for value in active_returns if value > 0) / len(active_returns)
    sharpe = None
    if interval is not None:
        sharpe = annualized_sharpe(simple_returns, interval)
    return {
        "total_return": total_return,
        "hit_rate": hit_rate,
        "exposure": len(active_returns) / bars,
        "trades": entries,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "bars": bars,
        "simple_returns": simple_returns,
    }


def _equity_path(simple_returns):
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in simple_returns:
        equity *= 1.0 + value
        if equity > peak:
            peak = equity
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
        if equity <= 0:
            return -1.0, 1.0
    return equity - 1.0, max_drawdown


def _validate_filter(regime_filter):
    if isinstance(regime_filter, (str, bytes)) or not regime_filter:
        raise ValueError("regime_filter must be a non-empty collection of regimes 0, 1, and 2")
    invalid = [label for label in regime_filter if label not in (0, 1, 2)]
    if invalid:
        raise ValueError(f"regime_filter contains {invalid}; allowed regimes are 0, 1, and 2")


def _validate_fee(fee_rate):
    if isinstance(fee_rate, bool) or not isinstance(fee_rate, (int, float)):
        raise ValueError("fee_rate must be a number in [0, 1)")
    if not 0 <= fee_rate < 1:
        raise ValueError("fee_rate must be in [0, 1)")


def _format_pct(value):
    return "n/a" if value is None else f"{value:.2%}"


def _format_num(value, width, spec):
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{width}{spec}}"


if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "15"
    result = compute_ewma_regime(symbol, interval, limit=1000)
    strategies = {
        "Long Low/Med": [0, 1],
        "Long Low Only": [0],
        "Long Med/High": [1, 2],
        "Long High Only": [2],
        "Long Low/High": [0, 2],
        "Long Med Only": [1],
    }
    results = {"Buy & Hold": buy_and_hold(result)}
    for name, regime_filter in strategies.items():
        results[name] = backtest_strategy(result, regime_filter)
    print_strategy_results(symbol, interval, results, DEFAULT_FEE_RATE)
