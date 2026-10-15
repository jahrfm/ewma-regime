import requests
import math
import time

def fetch_klines(symbol, interval, limit=200):
    """
    Fetch klines from Bybit.
    Returns list of klines in chronological order (oldest first).
    Each kline: [start, open, high, low, close, volume, turnover]
    """
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
    # Bybit returns in descending order (latest first). Reverse to chronological.
    klines = list(reversed(klines))
    return klines

def compute_ewma_regime(symbol, interval, limit=200, lambda_param=0.94, lookback=30,
                        low_thresh=30, high_thresh=70):
    """
    Compute EWMA volatility and regime labels.
    Returns a dictionary with:
        - timestamps: list of start times (ms)
        - closes: list of closing prices
        - returns: list of returns (length = len(closes)-1)
        - sigma: list of volatility estimates (same length as returns)
        - labels: list of regime labels (0=low, 1=medium, 2=high) for each return (same length as returns)
    """
    klines = fetch_klines(symbol, interval, limit)
    # Extract closing prices
    closes = [float(k[4]) for k in klines]
    # Calculate simple returns
    returns = []
    for i in range(1, len(closes)):
        ret = (closes[i] - closes[i-1]) / closes[i-1]
        returns.append(ret)
    # EWMA variance
    if len(returns) == 0:
        raise ValueError("No returns computed (need at least 2 klines)")
    sigma2 = [returns[0]**2]  # initial variance
    for i in range(1, len(returns)):
        sigma2_t = lambda_param * sigma2[i-1] + (1 - lambda_param) * (returns[i]**2)
        sigma2.append(sigma2_t)
    # Volatility estimate (sigma)
    sigma = [math.sqrt(v) for v in sigma2]
    # Function to compute percentile of a list without numpy
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
    # Labels based on sigma percentile over a lookback window
    labels = []
    for i in range(len(sigma)):
        if i < lookback:
            # Not enough history, label as medium
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
    # Align with klines: the first kline (index 0) has no return, so we can set its label to 1 (medium) or NaN.
    # We'll return labels for returns only, and the user can align as needed.
    result = {
        "timestamps": [k[0] for k in klines],
        "closes": closes,
        "returns": returns,
        "sigma": sigma,
        "labels": labels,  # same length as returns
        "symbol": symbol,
        "interval": interval,
        "lambda": lambda_param,
        "lookback": lookback,
        "low_thresh": low_thresh,
        "high_thresh": high_thresh
    }
    return result

def print_last_n(result, n=10):
    """Print the last n periods with label."""
    returns = result["returns"]
    sigma = result["sigma"]
    labels = result["labels"]
    closes = result["closes"]
    timestamps = result["timestamps"]
    # We have len(returns) = len(closes)-1
    # We'll print from the end
    start_idx = max(0, len(returns) - n)
    print(f"Last {n} periods for {result['symbol']} {result['interval']}m:")
    print("Time (ms)      Close      Return      Sigma      Label (0=low,1=medium,2=high)")
    for i in range(start_idx, len(returns)):
        # The return at index i corresponds to the change from closes[i] to closes[i+1]
        # We'll associate it with the closing price at i+1 and timestamp at i+1
        idx_close = i + 1
        print(f"{timestamps[idx_close]:13} {closes[idx_close]:10.2f} {returns[i]:10.5f} {sigma[i]:10.5f} {labels[i]}")

if __name__ == "__main__":
    # Example usage
    result = compute_ewma_regime("BTCUSDT", "15", limit=200)
    print_last_n(result, 10)
