import math
import time
from datetime import datetime, timezone

import requests

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
MAX_KLINES_PER_REQUEST = 1000
MINUTE_INTERVALS = {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720"}
CALENDAR_INTERVALS = {"D", "W", "M"}
# Bybit retCode for "too many visits".
RATE_LIMIT_CODE = 10006


def bars_per_year(interval):
    """Crypto trades every day, so a year is 365.25 calendar days."""
    interval = _normalize_interval(interval)
    if interval == "D":
        return 365.25
    if interval == "W":
        return 365.25 / 7.0
    if interval == "M":
        return 12.0
    minutes = int(interval)
    return 365.25 * 24.0 * 60.0 / minutes


def mean_and_std(values):
    """Sample mean and sample standard deviation (n - 1).

    Returns (mean, std). std is None when fewer than two observations exist.
    """
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n < 2:
        return mean, None
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    scale = max(abs(mean), max(abs(value) for value in values))
    # A nearly constant sample leaves a few ulps of variance. That is not dispersion.
    if variance <= (scale * 1e-12) ** 2:
        return mean, 0.0
    return mean, math.sqrt(variance)


def annualized_sharpe(values, interval):
    """Mean / sample std, scaled by sqrt(bars per year). None if std is undefined."""
    mean, std = mean_and_std(values)
    if mean is None or std is None or std == 0:
        return None
    return (mean / std) * math.sqrt(bars_per_year(interval))


def fetch_klines(symbol, interval, limit=200, category="linear", timeout=15):
    """
    Fetch klines from Bybit in chronological order (oldest first).

    Each kline is [start, open, high, low, close, volume, turnover], with
    start in milliseconds. The in-progress candle is dropped so the last
    return is a finished bar. limit may exceed 1000; pages are stitched on
    start time.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be an integer >= 1")
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("symbol must be a non-empty string")
    interval = _normalize_interval(interval)
    if category not in {"linear", "inverse", "spot"}:
        raise ValueError("category must be 'linear', 'inverse', or 'spot'")

    def fetch_page(end_ms, page_limit):
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": page_limit,
        }
        if end_ms is not None:
            params["end"] = end_ms
        data = _get_json(BYBIT_KLINE_URL, params, timeout)
        return data["result"]["list"]

    klines = _collect_pages(fetch_page, limit)
    klines = drop_incomplete_candle(klines, interval)
    if len(klines) < 2:
        raise ValueError(
            f"need at least 2 completed klines for {symbol} {interval}, got {len(klines)}"
        )
    return klines


def compute_ewma_regime(symbol="BTCUSDT", interval="15", limit=200, lambda_param=0.94,
                        lookback=30, low_thresh=30, high_thresh=70, klines=None,
                        category="linear"):
    """
    Compute EWMA volatility and a causal regime label.

    returns[i] is the simple return from closes[i] to closes[i + 1], so it is
    known only at the close of bar i + 1. sigma[i] uses returns[i]. labels[i]
    compares sigma[i] with the previous `lookback` volatilities and is therefore
    known at the same time as returns[i].

    labels[i] is None until a full lookback exists. It is never filled with
    Medium. A test of the label must pair labels[i - 1] with returns[i]: the
    label is already known, and the return has not happened yet.
    """
    _validate_params(lambda_param, lookback, low_thresh, high_thresh)
    if klines is None:
        klines = fetch_klines(symbol, interval, limit, category=category)
    else:
        klines = sorted(klines, key=lambda row: int(row[0]))
        if interval is not None:
            interval = _normalize_interval(interval)

    timestamps, closes = _parse_klines(klines)
    series = compute_ewma_from_closes(
        closes,
        lambda_param=lambda_param,
        lookback=lookback,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
    )
    series.update({
        "timestamps": timestamps,
        "symbol": symbol,
        "interval": str(interval) if interval is not None else None,
        "lambda": lambda_param,
        "lookback": lookback,
        "low_thresh": low_thresh,
        "high_thresh": high_thresh,
    })
    return series


def compute_ewma_from_closes(closes, lambda_param=0.94, lookback=30,
                             low_thresh=30, high_thresh=70):
    """EWMA regime from a chronological sequence of positive closes."""
    _validate_params(lambda_param, lookback, low_thresh, high_thresh)
    closes = _parse_closes(closes)
    returns = [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes))
    ]
    sigma2 = [returns[0] ** 2]
    for i in range(1, len(returns)):
        sigma2.append(
            lambda_param * sigma2[i - 1] + (1.0 - lambda_param) * (returns[i] ** 2)
        )
    sigma = [math.sqrt(v) for v in sigma2]
    labels = []
    for i, current in enumerate(sigma):
        if i < lookback:
            labels.append(None)
            continue
        window = sigma[i - lookback:i]
        low_threshold = percentile(window, low_thresh)
        high_threshold = percentile(window, high_thresh)
        if current < low_threshold:
            labels.append(0)
        elif current > high_threshold:
            labels.append(2)
        else:
            labels.append(1)
    return {
        "closes": closes,
        "returns": returns,
        "sigma": sigma,
        "labels": labels,
        "lambda": lambda_param,
        "lookback": lookback,
        "low_thresh": low_thresh,
        "high_thresh": high_thresh,
    }


def iter_forward_pairs(result, eval_start=None):
    """Yield (index, prior_label, forward_return) for a causal test of the label.

    prior_label is labels[index - 1], which was fixed before returns[index]
    was realized. Warmup labels (None) are skipped. eval_start, when set,
    drops earlier forward bars so several lookbacks can share one window.
    """
    returns = result["returns"]
    labels = result["labels"]
    if len(returns) != len(labels):
        raise ValueError("returns and labels must be the same length")
    start = 1 if eval_start is None else eval_start
    if isinstance(start, bool) or not isinstance(start, int) or start < 1:
        raise ValueError("eval_start must be an integer >= 1")
    for i in range(start, len(returns)):
        label = labels[i - 1]
        if label is None:
            continue
        if label not in (0, 1, 2):
            raise ValueError(f"label at index {i - 1} must be 0, 1, 2, or None")
        yield i, label, returns[i]


def percentile(lst, p):
    """Linear-interpolation percentile, matching the usual numpy default."""
    if not lst:
        raise ValueError("percentile of an empty sample is undefined")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 100:
        raise ValueError("percentile must be between 0 and 100")
    sorted_lst = sorted(lst)
    if p == 100:
        return sorted_lst[-1]
    k = (len(sorted_lst) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_lst[int(k)]
    return sorted_lst[int(f)] * (c - k) + sorted_lst[int(c)] * (k - f)


def drop_incomplete_candle(klines, interval, now_ms=None):
    """Drop the newest candle when its close time is still in the future."""
    if not klines:
        return klines
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    interval = _normalize_interval(interval)
    completed = list(klines)
    start = int(completed[-1][0])
    if start + _candle_duration_ms(start, interval) > now_ms:
        completed.pop()
    return completed


def print_last_n(result, n=10):
    """Print the last n return bars with the label known on that same bar."""
    returns = result["returns"]
    sigma = result["sigma"]
    labels = result["labels"]
    closes = result["closes"]
    timestamps = result["timestamps"]
    start_idx = max(0, len(returns) - n)
    interval = result.get("interval")
    print(f"Last {min(n, len(returns))} bars for {result.get('symbol')} {interval}:")
    print("Label is contemporaneous with the return. None means the lookback is not full yet.")
    print(f"{'Time (ms)':<15} {'Close':>10} {'Return':>10} {'Sigma':>10} {'Label':>6}")
    for i in range(start_idx, len(returns)):
        label = "-" if labels[i] is None else str(labels[i])
        print(
            f"{timestamps[i + 1]:<15} {closes[i + 1]:10.2f} "
            f"{returns[i]:10.5f} {sigma[i]:10.5f} {label:>6}"
        )


def _collect_pages(fetch_page, limit):
    """Stitch newest-first pages into a chronological list of at most `limit` rows."""
    remaining = limit
    end_ms = None
    collected = []
    seen = set()
    while remaining > 0:
        page_limit = min(MAX_KLINES_PER_REQUEST, remaining)
        batch = fetch_page(end_ms, page_limit)
        if not batch:
            break
        fresh = []
        for row in batch:
            ts = int(row[0])
            if ts in seen:
                continue
            seen.add(ts)
            fresh.append(row)
        if not fresh:
            break
        collected.extend(fresh)
        remaining -= len(fresh)
        oldest = min(int(row[0]) for row in fresh)
        if end_ms is not None and oldest >= end_ms:
            raise RuntimeError("Bybit pagination did not move backward in time")
        end_ms = oldest - 1
        if len(batch) < page_limit:
            break
        if remaining > 0:
            time.sleep(0.05)
    collected.sort(key=lambda row: int(row[0]))
    if len(collected) > limit:
        collected = collected[-limit:]
    return collected


def _get_json(url, params, timeout, attempts=3):
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"Bybit returned a non-JSON body (HTTP {response.status_code})"
                ) from exc
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 == attempts:
                break
            time.sleep(0.5 * (attempt + 1))
            continue
        ret_code = data.get("retCode")
        if ret_code == 0:
            result = data.get("result") or {}
            if "list" not in result:
                raise RuntimeError(f"Bybit response has no kline list: {data}")
            return data
        last_error = RuntimeError(f"API error: {data}")
        if ret_code != RATE_LIMIT_CODE or attempt + 1 == attempts:
            raise last_error
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Bybit request failed after {attempts} attempts") from last_error


def _normalize_interval(interval):
    text = str(interval).strip()
    if text not in MINUTE_INTERVALS and text not in CALENDAR_INTERVALS:
        allowed = ", ".join(sorted(MINUTE_INTERVALS, key=int) + sorted(CALENDAR_INTERVALS))
        raise ValueError(f"interval must be one of: {allowed}")
    return text


def _candle_duration_ms(start_ms, interval):
    if interval == "D":
        return 86_400_000
    if interval == "W":
        return 7 * 86_400_000
    if interval == "M":
        start = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
        year = start.year + (1 if start.month == 12 else 0)
        month = 1 if start.month == 12 else start.month + 1
        end = datetime(year, month, 1, tzinfo=timezone.utc)
        return int(end.timestamp() * 1000) - start_ms
    return int(interval) * 60_000


def _validate_params(lambda_param, lookback, low_thresh, high_thresh):
    if isinstance(lambda_param, bool) or not isinstance(lambda_param, (int, float)):
        raise ValueError("lambda_param must be a number")
    if not math.isfinite(lambda_param) or not 0 <= lambda_param < 1:
        raise ValueError("lambda_param must be in [0, 1) so variance stays non-negative and updates")
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ValueError("lookback must be an integer >= 1")
    for name, value in (("low_thresh", low_thresh), ("high_thresh", high_thresh)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number from 0 to 100")
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"{name} must be between 0 and 100")
    if low_thresh >= high_thresh:
        raise ValueError("low_thresh must be less than high_thresh")


def _parse_closes(closes):
    parsed = []
    for i, price in enumerate(closes):
        try:
            price = float(price)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"close at index {i} is not a number") from exc
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"close at index {i} must be finite and > 0")
        parsed.append(price)
    if len(parsed) < 2:
        raise ValueError("need at least 2 closes to compute a return")
    return parsed


def _parse_klines(klines):
    if not klines:
        raise ValueError("no klines")
    timestamps = []
    closes = []
    for i, row in enumerate(klines):
        if len(row) < 5:
            raise ValueError(f"kline {i} has no close price")
        try:
            timestamps.append(int(row[0]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"kline {i} has no integer start time") from exc
        closes.append(row[4])
    return timestamps, _parse_closes(closes)


if __name__ == "__main__":
    result = compute_ewma_regime("BTCUSDT", "15", limit=200)
    print_last_n(result, 10)
