import requests
import math

def fetch_klines(symbol, interval, limit=200):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data.get("retCode") != 0:
        raise Exception(f"API error: {data}")
    klines = data["result"]["list"]
    klines = list(reversed(klines))
    return klines

def compute_ewma_regime(symbol, interval, limit=200, lambda_param=0.94, lookback=30,
                        low_thresh=30, high_thresh=70):
    klines = fetch_klines(symbol, interval, limit)
    closes = [float(k[4]) for k in klines]
    returns = []
    for i in range(1, len(closes)):
        ret = (closes[i] - closes[i-1]) / closes[i-1]
        returns.append(ret)
    if len(returns) == 0:
        raise ValueError("No returns")
    sigma2 = [returns[0]**2]
    for i in range(1, len(returns)):
        sigma2_t = lambda_param * sigma2[i-1] + (1 - lambda_param) * (returns[i]**2)
        sigma2.append(sigma2_t)
    sigma = [math.sqrt(v) for v in sigma2]
    def percentile(lst, p):
        if not lst:
            return 0.0
        sorted_lst = sorted(lst)
        k = (len(sorted_lst) - 1) * p / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_lst[int(k)]
        d0 = sorted_lst[int(f)] * (c - k)
        d1 = sorted_lst[int(c)] * (k - f)
        return d0 + d1
    labels = []
    for i in range(len(sigma)):
        if i < lookback:
            labels.append(1)
            continue
        window = sigma[i-lookback:i]
        current = sigma[i]
        low_threshold = percentile(window, low_thresh)
        high_threshold = percentile(window, high_thresh)
        if current < low_threshold:
            labels.append(0)
        elif current > high_threshold:
            labels.append(2)
        else:
            labels.append(1)
    result = {
        "symbol": symbol,
        "interval": interval,
        "timestamps": [k[0] for k in klines],
        "closes": closes,
        "returns": returns,
        "sigma": sigma,
        "labels": labels,
        "lambda": lambda_param,
        "lookback": lookback,
        "low_thresh": low_thresh,
        "high_thresh": high_thresh
    }
    return result

def backtest_strategy(result, regime_filter):
    """
    regime_filter: list of allowed regimes for long positions
    e.g., [0,1] = long in low/medium, flat in high
          [0]   = long only in low
          [1,2] = long in medium/high
    Returns: total return, win rate, number of trades
    """
    returns = result["returns"]
    labels = result["labels"]
    
    # We'll take a position at the close of period i based on label of return i-1?
    # Actually, we have label for return i (which is from close i to i+1)
    # Let's say we know the label at the open of period i (based on previous return)
    # For simplicity, we'll align: position for return i is based on label i-1
    # So we shift labels forward by 1, with first position being neutral
    
    if len(labels) != len(returns):
        # This shouldn't happen with our implementation
        raise ValueError("Labels and returns length mismatch")
    
    # Create position series: 1 = long, 0 = flat
    # Position at time i is based on label at time i-1 (previous period's regime)
    positions = [0] * len(returns)  # start flat
    for i in range(1, len(labels)):
        if labels[i-1] in regime_filter:
            positions[i] = 1
    
    # Calculate strategy returns
    strategy_returns = []
    for pos, ret in zip(positions, returns):
        strategy_returns.append(pos * ret)
    
    # Metrics
    total_return = sum(strategy_returns)
    # Compound return approximation (for small returns, sum is fine)
    # Win rate
    wins = sum(1 for r in strategy_returns if r > 0)
    win_rate = wins / len(strategy_returns) if len(strategy_returns) > 0 else 0
    # Number of trades (position changes)
    trades = sum(1 for i in range(1, len(positions)) if positions[i] != positions[i-1])
    
    return {
        "total_return": total_return,
        "win_rate": win_rate,
        "trades": trades,
        "strategy_returns": strategy_returns,
        "positions": positions
    }

def print_strategy_results(symbol, interval, results_dict):
    print(f"\n=== Strategy Backtest: {symbol} {interval} ===")
    print(f"{'Strategy':<25} {'Total Return':<15} {'Win Rate':<12} {'Trades':<8}")
    for name, res in results_dict.items():
        print(f"{name:<25} {res['total_return']:<15.6f} {res['win_rate']:<12.2%} {res['trades']:<8}")

if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "15"
    limit = 200
    
    result = compute_ewma_regime(symbol, interval, limit)
    
    # Test different strategies
    strategies = {
        "Buy & Hold": None,  # special case
        "Long Low/Med": [0,1],
        "Long Low Only": [0],
        "Long Med/High": [1,2],
        "Long High Only": [2],
        "Reverse (Long High)": [2],  # same as above but we'll interpret differently
        "Long Low/High": [0,2],
        "Long Med Only": [1]
    }
    
    results = {}
    # Buy and hold
    bh_return = sum(result["returns"])
    results["Buy & Hold"] = {
        "total_return": bh_return,
        "win_rate": sum(1 for r in result["returns"] if r > 0) / len(result["returns"]),
        "trades": 0  # no trades
    }
    
    for name, regime_filter in strategies.items():
        if regime_filter is None:
            continue  # already handled Buy & Hold
        res = backtest_strategy(result, regime_filter)
        results[name] = res
    
    print_strategy_results(symbol, interval, results)
