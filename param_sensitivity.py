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
    returns = result["returns"]
    labels = result["labels"]
    if len(labels) != len(returns):
        raise ValueError("Labels and returns length mismatch")
    positions = [0] * len(returns)
    for i in range(1, len(labels)):
        if labels[i-1] in regime_filter:
            positions[i] = 1
    strategy_returns = []
    for pos, ret in zip(positions, returns):
        strategy_returns.append(pos * ret)
    total_return = sum(strategy_returns)
    win_rate = sum(1 for r in strategy_returns if r > 0) / len(strategy_returns) if len(strategy_returns) > 0 else 0
    trades = sum(1 for i in range(1, len(positions)) if positions[i] != positions[i-1])
    return {
        "total_return": total_return,
        "win_rate": win_rate,
        "trades": trades
    }

def test_params(symbol="BTCUSDT", interval="15", limit=200):
    base_params = {
        "lambda_param": 0.94,
        "lookback": 30,
        "low_thresh": 30,
        "high_thresh": 70
    }
    results = []
    
    # Test lambda values (memory)
    for lam in [0.8, 0.9, 0.94, 0.97, 0.99]:
        params = base_params.copy()
        params["lambda_param"] = lam
        result = compute_ewma_regime(symbol, interval, limit, **params)
        res = backtest_strategy(result, [0,1])  # Long Low/Med
        results.append(("lambda", lam, res["total_return"], res["win_rate"], res["trades"]))
    
    # Test lookback values
    for lb in [10, 20, 30, 50, 70]:
        params = base_params.copy()
        params["lookback"] = lb
        result = compute_ewma_regime(symbol, interval, limit, **params)
        res = backtest_strategy(result, [0,1])
        results.append(("lookback", lb, res["total_return"], res["win_rate"], res["trades"]))
    
    # Test threshold values (make more/less sensitive)
    for low, high in [(20,80), (25,75), (30,70), (35,65), (40,60)]:
        params = base_params.copy()
        params["low_thresh"] = low
        params["high_thresh"] = high
        result = compute_ewma_regime(symbol, interval, limit, **params)
        res = backtest_strategy(result, [0,1])
        results.append(("thresholds", f"{low}/{high}", res["total_return"], res["win_rate"], res["trades"]))
    
    return results

if __name__ == "__main__":
    print("Parameter Sensitivity Test: BTCUSDT 15m, Long Low/Med Strategy")
    print(f"{'Param':<12} {'Value':<10} {'Return':<10} {'Win Rate':<10} {'Trades':<8}")
    print("-" * 55)
    results = test_params()
    for param, value, ret, wr, trades in results:
        print(f"{param:<12} {str(value):<10} {ret:<10.6f} {wr:<10.2%} {trades:<8}")
