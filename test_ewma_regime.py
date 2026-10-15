import unittest

from backtest_ewma import forward_regime_stats
from backtest_strategy import backtest_strategy, buy_and_hold
from ewma_regime import (
    _collect_pages,
    compute_ewma_from_closes,
    drop_incomplete_candle,
    iter_forward_pairs,
    percentile,
)
from param_sensitivity import LOOKBACK_VALUES, sensitivity_rows


def prices_from_returns(returns, start=100.0):
    prices = [start]
    for value in returns:
        prices.append(prices[-1] * (1.0 + value))
    return prices


class LabelTests(unittest.TestCase):
    def test_warmup_labels_are_missing_not_medium(self):
        result = compute_ewma_from_closes(prices_from_returns([0.001] * 40), lookback=30)
        self.assertEqual(result["labels"][:30], [None] * 30)
        self.assertTrue(all(label in (0, 1, 2) for label in result["labels"][30:]))

    def test_flat_volatility_stays_medium(self):
        result = compute_ewma_from_closes([100.0] * 50, lookback=10)
        self.assertEqual(result["labels"][10:], [1] * (len(result["labels"]) - 10))

    def test_volatility_jump_is_labeled_high_on_that_bar(self):
        closes = [100.0] * 31 + [200.0]
        result = compute_ewma_from_closes(closes, lookback=20, lambda_param=0.9)
        self.assertEqual(result["labels"][-2], 1)
        self.assertEqual(result["labels"][-1], 2)

    def test_volatility_crush_is_labeled_low(self):
        returns = [0.02, -0.02] * 40 + [0.0001, -0.0001] * 5
        result = compute_ewma_from_closes(prices_from_returns(returns), lookback=20, lambda_param=0.8)
        self.assertEqual(result["labels"][-1], 0)

    def test_future_price_does_not_change_earlier_labels(self):
        returns = [0.001 if i % 2 == 0 else -0.002 for i in range(80)]
        original = compute_ewma_from_closes(prices_from_returns(returns), lookback=15)
        bumped = prices_from_returns(returns)
        bumped[-1] *= 1.5
        changed = compute_ewma_from_closes(bumped, lookback=15)
        self.assertEqual(original["labels"][:-1], changed["labels"][:-1])
        self.assertEqual(original["sigma"][:-1], changed["sigma"][:-1])

    def test_spike_is_not_scored_with_the_label_it_creates(self):
        closes = [100.0] * 41 + [200.0, 200.0]
        result = compute_ewma_from_closes(closes, lookback=20)
        self.assertEqual(result["labels"][40], 2)
        tail = [pair for pair in iter_forward_pairs(result) if pair[0] >= 40]
        self.assertEqual(tail, [(40, 1, 1.0), (41, 2, 0.0)])
        stats = forward_regime_stats(result)
        self.assertEqual(stats[2]["count"], 1)
        self.assertEqual(stats[2]["mean"], 0.0)
        self.assertIn(1.0, _bucket_values(result, 1))

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            compute_ewma_from_closes([100.0, 0.0])
        with self.assertRaises(ValueError):
            compute_ewma_from_closes([100.0, 101.0], lambda_param=1)
        with self.assertRaises(ValueError):
            compute_ewma_from_closes([100.0, 101.0], lambda_param=-0.1)
        with self.assertRaises(ValueError):
            compute_ewma_from_closes([100.0, 101.0], low_thresh=70, high_thresh=30)
        with self.assertRaises(ValueError):
            compute_ewma_from_closes([100.0])


class ForwardTestTests(unittest.TestCase):
    def test_metrics_use_only_the_following_bar(self):
        result = {
            "returns": [0.10, -0.05, 0.02],
            "labels": [0, 1, 2],
            "interval": "D",
            "lookback": 1,
        }
        gross = backtest_strategy(result, [0], fee_rate=0.0)
        # Label 0 is known before the -5% bar. The +10% bar has no prior label.
        self.assertAlmostEqual(gross["total_return"], -0.05)
        self.assertEqual(gross["hit_rate"], 0.0)
        self.assertEqual(gross["trades"], 1)
        self.assertAlmostEqual(gross["exposure"], 0.5)
        self.assertEqual(gross["bars"], 2)

        net = backtest_strategy(result, [0], fee_rate=0.01)
        # Enter, earn -5%, then exit flat. No terminal fee because the last position is flat.
        self.assertAlmostEqual(net["total_return"], 0.99 * 0.95 * 0.99 - 1.0)
        self.assertEqual(net["hit_rate"], 0.0)

        held = buy_and_hold(result)
        self.assertAlmostEqual(held["total_return"], 0.95 * 1.02 - 1.0)
        self.assertAlmostEqual(held["exposure"], 1.0)
        self.assertEqual(held["trades"], 0)

    def test_terminal_exit_fee_is_charged_when_still_long(self):
        result = {
            "returns": [0.0, 0.10],
            "labels": [1, 1],
            "interval": "D",
        }
        net = backtest_strategy(result, [1], fee_rate=0.01)
        self.assertAlmostEqual(net["total_return"], 0.99 * 1.10 * 0.99 - 1.0)

    def test_warmup_is_not_traded(self):
        returns = [0.01] * 25
        result = compute_ewma_from_closes(prices_from_returns(returns), lookback=20)
        result["interval"] = "D"
        metrics = backtest_strategy(result, [0, 1, 2], fee_rate=0.0)
        self.assertEqual(metrics["bars"], len(returns) - 21)

    def test_empty_window_is_not_a_zero_return(self):
        result = {"returns": [0.01, 0.01], "labels": [None, None], "interval": "D"}
        metrics = backtest_strategy(result, [1], fee_rate=0.0)
        self.assertIsNone(metrics["total_return"])
        self.assertEqual(metrics["bars"], 0)

    def test_percentile_interpolates(self):
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 50), 2.5)

    def test_constant_forwards_do_not_explode_sharpe(self):
        result = {
            "returns": [0.0, 0.01, 0.01, 0.01],
            "labels": [1, 1, 1, 1],
            "interval": "15",
        }
        stats = forward_regime_stats(result)
        self.assertEqual(stats[1]["std"], 0.0)
        self.assertIsNone(stats[1]["sharpe"])


class SensitivityTests(unittest.TestCase):
    def test_rows_share_one_sample(self):
        returns = []
        for i in range(120):
            scale = 0.002 if i < 60 else 0.01
            returns.append(scale if i % 2 == 0 else -scale)
        closes = prices_from_returns(returns)
        klines = [[i * 60_000, 0, 0, 0, closes[i], 0, 0] for i in range(len(closes))]
        rows = sensitivity_rows(klines, fee_rate=0.0)
        lambda_bars = [res["bars"] for name, _, res in rows if name == "lambda"]
        threshold_bars = [res["bars"] for name, _, res in rows if name == "thresholds"]
        lookback_bars = [res["bars"] for name, _, res in rows if name == "lookback"]
        self.assertGreater(lambda_bars[0], 0)
        self.assertEqual(len(set(lambda_bars)), 1)
        self.assertEqual(len(set(threshold_bars)), 1)
        self.assertEqual(lambda_bars, threshold_bars)
        self.assertEqual(len(set(lookback_bars)), 1)
        self.assertLess(lookback_bars[0], lambda_bars[0])
        self.assertEqual(
            lookback_bars[0],
            len(returns) - (max(LOOKBACK_VALUES) + 1),
        )


class FetchTests(unittest.TestCase):
    def test_pages_are_stitched_oldest_first_without_overlap(self):
        def fetch_page(end_ms, page_limit):
            if end_ms is None:
                self.assertEqual(page_limit, 1000)
                return [[(2000 - i) * 1000, "new"] for i in range(page_limit)]
            self.assertEqual(page_limit, 2)
            self.assertEqual(end_ms, 1001 * 1000 - 1)
            return [[end_ms, "just-older"], [end_ms - 1000, "older"]]

        stitched = _collect_pages(fetch_page, limit=1002)
        stamps = [row[0] for row in stitched]
        self.assertEqual(stamps, sorted(stamps))
        self.assertEqual(len(stamps), len(set(stamps)))
        self.assertEqual(len(stamps), 1002)
        self.assertEqual(stamps[0], 1001 * 1000 - 1 - 1000)
        self.assertEqual(stamps[-1], 2000 * 1000)

    def test_incomplete_candle_is_dropped(self):
        klines = [[0, "1"], [60_000, "2"]]
        kept = drop_incomplete_candle(klines, "1", now_ms=90_000)
        self.assertEqual(kept, [[0, "1"]])
        done = drop_incomplete_candle(klines, "1", now_ms=120_000)
        self.assertEqual(done, klines)


def _bucket_values(result, label):
    return [forward for _, prior, forward in iter_forward_pairs(result) if prior == label]


if __name__ == "__main__":
    unittest.main()
