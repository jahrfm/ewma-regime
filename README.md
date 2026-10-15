# EWMA Volatility Regime Indicator

This repository implements the Exponentially Weighted Moving Average (EWMA) volatility regime approach described in the YouTube video about market regimes. The goal is to provide an objective volatility label (Low/Medium/High) that can inform trading strategies across any timeframe and symbol.

## Overview

The EWMA volatility approach creates a "risk thermometer" that adapts smoothly to changing market conditions:
- **Low Volatility**: Calm markets where prices wiggle gently
- **Medium Volatility**: Normal market conditions  
- **High Volatility**: Sharp moves, slides, panics, and bounces that make no sense

Unlike regime-switching models (like Markov models) that assume discrete states, EWMA provides a continuous volatility estimate that we then discretize into regimes using percentile thresholds.

## How It Works

1. **Fetch price data** (OHLCV) from any source (default: Bybit API)
2. **Calculate simple returns**: \( r_t = \frac{P_t - P_{t-1}}{P_{t-1}} \)
3. **Compute EWMA variance**: 
   \[
   \sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_t^2
   \]
   where \(\lambda\) controls the memory (higher = longer memory)
4. **Convert to volatility**: \( \sigma_t = \sqrt{\sigma_t^2} \)
5. **Label regimes** based on where current volatility falls in the lookback window:
   - Below low_threshold percentile → Low (0)
   - Above high_threshold percentile → High (2)  
   - Otherwise → Medium (1)

## Files in This Repository

- `ewma_regime.py` - Core implementation with functions to fetch data and compute regimes
- `backtest_ewma.py` - Basic backtesting showing regime characteristics
- `backtest_strategy.py` - Strategy backtests using regime filters
- `param_sensitivity.py` - Parameter sensitivity analysis
- `ewma_param_test.py` - Comprehensive parameter testing across multiple symbols/timeframes

## Usage

### Basic Example
```python
from ewma_regime import compute_ewma_regime

# Get EWMA regime labels for BTCUSDT 15-minute data
result = compute_ewma_regime(
    symbol="BTCUSDT",
    interval="15",  # 1m, 5m, 15m, 30m, 60m, 240m, D, W, M
    limit=200       # Number of periods to fetch
)

# result contains:
# - returns: list of returns
# - sigma: list of volatility estimates  
# - labels: list of regime labels (0=Low, 1=Medium, 2=High)
# - timestamps, closes, etc.

print(f"Current regime: {result['labels'][-1]}")  # Most recent label
```

### With Custom Parameters
```python
result = compute_ewma_regime(
    symbol="ETHUSDT",
    interval="60",  # 1-hour
    lambda_param=0.90,    # Memory parameter (0.8-0.99)
    lookback=15,          # Lookback window for percentiles
    low_thresh=30,        # Low threshold percentile
    high_thresh=70        # High threshold percentile
)
```

## Backtesting Results

Our testing across BTCUSDT, ETHUSDT, and SOLUSDT showed that the EWMA regime filter can add value:

### Key Findings
1. **Regime Filtering Works**: Being long only during Low/Medium volatility (and flat during High) often outperformed buy-and-hold
2. **Parameter Sensitivity**: 
   - Lower lambda (0.80-0.90) = more responsive regime detection
   - Higher lambda (0.95-0.99) = more stable regimes
   - Lookback windows of 15-30 periods worked well
   - Thresholds around 30/70 or 35/65 provided good balance
3. **Symbol Differences**: 
   - SOLUSDT favored lower lambda (more responsive)
   - BTCUSDT favored higher lambda (more stable)
   - ETHUSDT worked well with middle-ground parameters

### Example Strategy Performance (BTCUSDT 15m, 200 periods)
| Strategy | Total Return | Win Rate | Trades |
|----------|--------------|----------|---------|
| Buy & Hold | -1.26% | 50.25% | 0 |
| **Long Low/Med** (flat in High) | **+2.17%** | 41.71% | 13 |
| Long Low Only | +0.68% | 21.11% | 19 |
| Long Med Only | +1.49% | 20.60% | 28 |

## Integration with Trading Strategies

The regime label can be used as a filter for any trading strategy:

```python
# Example: Only take moving average crossover signals in Low/Med volatility
if regime_label in [0, 1]:  # Low or Medium volatility
    if ma_fast > ma_slow:
        # Enter long
    elif ma_fast < ma_slow:
        # Enter short (or exit long)
else:
    # High volatility - stay flat or reduce position size
```

### Position Sizing Example
```python
# Reduce position size in high volatility
base_size = 1000  # USD
volatility_multiplier = {
    0: 1.0,   # Low volatility - full size
    1: 0.7,   # Medium volatility - 70% size  
    2: 0.3    # High volatility - 30% size
}
position_size = base_size * volatility_multiplier[current_regime]
```

## Parameter Tuning Guidelines

### For Maximum Returns (Aggressive)
- Lower lambda (0.80-0.90) for faster adaptation
- Shorter lookback (10-25) for responsive thresholds
- Consider narrower thresholds (35/65) to capture more extreme regimes

### For Best Risk-Adjusted Returns (Conservative)  
- Higher lambda (0.95-0.99) for stable regime detection
- Medium lookback (25-40) for robust percentiles
- Wider thresholds (25/75 or 30/70) to avoid whipsaws

### For Universal Starting Point
```python
# Good default parameters that worked reasonably well across tests
compute_ewma_regime(
    symbol="YOUR_SYMBOL",
    interval="YOUR_TIMEFRAME",
    lambda_param=0.94,
    lookback=30,
    low_thresh=30,
    high_thresh=70
)
```

## Implementation Details

The implementation uses only standard library + requests for API calls, making it highly portable:
- No numpy/pandas dependencies
- Pure Python percentile calculation
- Easy to integrate into existing trading systems
- Works with any OHLCV data source (just modify the fetch function)

## Extending to Other Data Sources

To use with different data sources, modify the `fetch_klines` function in `ewma_regime.py`:
```python
def fetch_klines(symbol, interval, limit=200):
    # Replace with your data source logic
    # Should return list of klines in chronological order:
    # [timestamp, open, high, low, close, volume, turnover]
    # Timestamp in milliseconds
    pass
```

## License

MIT License - feel free to use, modify, and distribute.

## Disclaimer

This is for educational purposes only. Past performance does not guarantee future results. Always conduct your own research and consider paper trading before deploying any strategy with real capital.
