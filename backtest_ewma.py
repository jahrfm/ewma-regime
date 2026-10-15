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
    # Align: labels length = len(returns)
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

def backtest_regime(result):
    returns = result["returns"]
    labels = result["labels"]
    # Group returns by label
    ret_by_label = {0: [], 1: [], 2: []}
    for r, l in zip(returns, labels):
        ret_by_label[l].append(r)
    stats = {}
    for l in [0,1,2]:
        lst = ret_by_label[l]
        if len(lst) == 0:
            stats[l] = {"count":0, "mean":0.0, "std":0.0, "sharpe":0.0, "hit_rate":0.0}
            continue
        mean = sum(lst)/len(lst)
        variance = sum((x-mean)**2 for x in lst)/len(lst)
        std = math.sqrt(variance) if variance>0 else 0.0
        sharpe = mean/std if std>0 else 0.0
        hit = sum(1 for x in lst if x>0)/len(lst)
        stats[l] = {"count":len(lst), "mean":mean, "std":std, "sharpe":sharpe, "hit_rate":hit}
    return stats, ret_by_label

def print_backtest(symbol, interval, stats):
    print(f"\n=== Backtest EWMA Regime: {symbol} {interval} ===")
    print(f"{'Regime':<8} {'Count':<8} {'Mean Ret':<12} {'Std':<12} {'Sharpe':<10} {'Hit Rate':<10}")
    for l in [0,1,2]:
        s = stats[l]
        label_name = {0:"Low",1:"Medium",2:"High"}[l]
        print(f"{label_name:<8} {s['count']:<8} {s['mean']:<12.6f} {s['std']:<12.6f} {s['sharpe']:<10.4f} {s['hit_rate']:<10.2%}")

if __name__ == "__main__":
    # Test BTCUSDT 15m
    result = compute_ewma_regime("BTCUSDT", "15", limit=200)
    stats, _ = backtest_regime(result)
    print_backtest("BTCUSDT", "15", stats)
    # Also test ETHUSDT 15m
    result2 = compute_ewma_regime("ETHUSDT", "15", limit=200)
    stats2, _ = backtest_regime(result2)
    print_backtest("ETHUSDT", "15", stats2)
    # Test BTCUSDT 1h
    result3 = compute_ewma_regime("BTCUSDT", "60", limit=200)
    stats3, _ = backtest_regime(result3)
    print_backtest("BTCUSDT", "60", stats3)
