"""Forward-return profile of the EWMA regime label.

The label on bar t - 1 is scored against the return of bar t. The return that
entered the label is not the return being measured.
"""

from ewma_regime import (
    annualized_sharpe,
    compute_ewma_regime,
    iter_forward_pairs,
    mean_and_std,
)

REGIMES = {0: "Low", 1: "Medium", 2: "High"}


def forward_regime_stats(result, eval_start=None):
    """Next-bar return distribution conditional on the already-known label."""
    grouped = {0: [], 1: [], 2: []}
    paired = []
    for _, label, forward in iter_forward_pairs(result, eval_start):
        grouped[label].append(forward)
        paired.append(forward)
    stats = {label: _summarize(grouped[label], result.get("interval")) for label in REGIMES}
    stats["all"] = _summarize(paired, result.get("interval"))
    return stats


def print_backtest(symbol, interval, stats):
    print(f"\n=== Forward EWMA regime test: {symbol} {interval} ===")
    print("Each row is the return of the bar after the label was known. Warmup bars are excluded.")
    print(
        f"{'Regime':<8} {'Bars':>6} {'Mean':>12} {'Std':>12} "
        f"{'Hit rate':>10} {'Ann. Sharpe':>12}"
    )
    order = [("all", "All")] + [(label, REGIMES[label]) for label in (0, 1, 2)]
    for key, name in order:
        row = stats[key]
        print(
            f"{name:<8} {row['count']:>6} {_format_num(row['mean'], 12, '.6f')} "
            f"{_format_num(row['std'], 12, '.6f')} {_format_pct(row['hit_rate']):>10} "
            f"{_format_num(row['sharpe'], 12, '.3f')}"
        )


def _summarize(values, interval):
    mean, std = mean_and_std(values)
    hit_rate = None
    if values:
        hit_rate = sum(1 for value in values if value > 0) / len(values)
    sharpe = None
    if interval is not None and len(values) >= 2:
        sharpe = annualized_sharpe(values, interval)
    return {
        "count": len(values),
        "mean": mean,
        "std": std,
        "hit_rate": hit_rate,
        "sharpe": sharpe,
    }


def _format_num(value, width, spec):
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{width}{spec}}"


def _format_pct(value):
    return "n/a" if value is None else f"{value:.2%}"


if __name__ == "__main__":
    samples = [("BTCUSDT", "15"), ("ETHUSDT", "15"), ("BTCUSDT", "60")]
    for symbol, interval in samples:
        result = compute_ewma_regime(symbol, interval, limit=1000)
        print_backtest(symbol, interval, forward_regime_stats(result))
