"""Parameter sweep on one downloaded sample.

Every lambda and threshold row sees the same closes and the same forward bars.
Lookback rows are scored from the first bar where the longest lookback already
has a real label, so a longer warmup is not rewarded for sitting out a different
slice of history.
"""

from backtest_strategy import DEFAULT_FEE_RATE, backtest_strategy
from ewma_regime import compute_ewma_regime, fetch_klines

LAMBDA_VALUES = (0.8, 0.9, 0.94, 0.97, 0.99)
LOOKBACK_VALUES = (10, 20, 30, 50, 70)
THRESHOLD_VALUES = ((20, 80), (25, 75), (30, 70), (35, 65), (40, 60))


def sensitivity_rows(klines, symbol="BTCUSDT", interval="15", fee_rate=DEFAULT_FEE_RATE,
                     regime_filter=(0, 1)):
    """Long Low/Medium results for each setting, fetched data held fixed."""
    base = {
        "lambda_param": 0.94,
        "lookback": 30,
        "low_thresh": 30,
        "high_thresh": 70,
    }
    lookback_start = max(LOOKBACK_VALUES) + 1
    rows = []
    for lam in LAMBDA_VALUES:
        params = {**base, "lambda_param": lam}
        rows.append(("lambda", lam, _run(klines, symbol, interval, params, fee_rate, regime_filter, None)))
    for lookback in LOOKBACK_VALUES:
        params = {**base, "lookback": lookback}
        rows.append((
            "lookback",
            lookback,
            _run(klines, symbol, interval, params, fee_rate, regime_filter, lookback_start),
        ))
    for low, high in THRESHOLD_VALUES:
        params = {**base, "low_thresh": low, "high_thresh": high}
        rows.append((
            "thresholds",
            f"{low}/{high}",
            _run(klines, symbol, interval, params, fee_rate, regime_filter, None),
        ))
    return rows


def print_sensitivity(rows, symbol, interval, fee_rate):
    print(f"Parameter sensitivity: {symbol} {interval}, long in Low/Medium")
    print(
        "One kline download is reused. Lookback rows share the window of the longest lookback. "
        f"Returns are compounded and net of {fee_rate:.2%} per side."
    )
    print(f"{'Param':<12} {'Value':<10} {'Return':>10} {'Hit rate':>10} {'Trades':>8} {'Bars':>6}")
    print("-" * 60)
    for param, value, res in rows:
        hit = "n/a" if res["hit_rate"] is None else f"{res['hit_rate']:.2%}"
        ret = "n/a" if res["total_return"] is None else f"{res['total_return']:.2%}"
        print(f"{param:<12} {str(value):<10} {ret:>10} {hit:>10} {res['trades']:>8} {res['bars']:>6}")


def _run(klines, symbol, interval, params, fee_rate, regime_filter, eval_start):
    result = compute_ewma_regime(symbol, interval, klines=klines, **params)
    return backtest_strategy(
        result,
        regime_filter,
        fee_rate=fee_rate,
        eval_start=eval_start,
    )


if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "15"
    klines = fetch_klines(symbol, interval, limit=1000)
    rows = sensitivity_rows(klines, symbol, interval)
    print_sensitivity(rows, symbol, interval, DEFAULT_FEE_RATE)
