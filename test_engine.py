import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import data
from engine import (simulate, decide, penalty, barrier_pnl, cl_expiry, front_expiry_days,
                    market_open, eia_events, HORIZON_HOURS)

OPEN = datetime(2026, 10, 28, 15, 0, tzinfo=timezone.utc)      # Wednesday 11:00 New York
CLOSED = datetime(2026, 10, 31, 15, 0, tzinfo=timezone.utc)    # Saturday


def scenario(seed=1, news=0., now=OPEN):
    inputs, history, intraday, _ = data.demo(seed)
    for k in inputs:
        inputs[k]["date"] = now
    inputs["news"]["value"] = news
    return inputs, history, intraday


def run(inputs, history, intraday, now=OPEN, **kw):
    expiry = front_expiry_days(now)
    sim = simulate(inputs, history, 76., expiry, HORIZON_HOURS,
                   events=eia_events(now, HORIZON_HOURS), intraday=intraday)
    return sim, decide(inputs, history, sim, now, expiry, **kw)


class ModelTests(unittest.TestCase):
    def test_repeatability_and_inventory_bounds(self):
        inputs, history, intraday = scenario()
        a = simulate(inputs, history, hours=5, paths=500)
        b = simulate(inputs, history, hours=5, paths=500)
        self.assertTrue(np.array_equal(a["samples"], b["samples"]))
        self.assertTrue(np.isfinite(a["samples"]).all())
        self.assertGreaterEqual(a["inventory"].min(), 0)
        self.assertLessEqual(a["inventory"].max(), 76)

    def test_capacity_penalty(self):
        self.assertGreater(penalty(75.9, 76, 20), penalty(20, 76, 20))
        self.assertGreater(penalty(75.9, 76, 1), penalty(75.9, 76, 180))

    def test_negative_prices_allowed(self):
        inputs, history, _ = scenario()
        inputs["price"]["value"] = -20.
        sim = simulate(inputs, history, hours=1, paths=500)
        self.assertLess(sim["samples"].min(), 0)

    def test_missing_optional_factors_do_not_crash(self):
        inputs, history, intraday = scenario()
        for key in ("freight", "curve", "rate", "dollar", "inventory", "crack",
                    "utilization", "surprise", "news"):
            inputs[key]["value"] = None
        sim, result = run(inputs, history, intraday)
        self.assertIn(result["action"], ("LONG", "SHORT", "WAIT"))

    def test_news_moves_the_forecast(self):
        inputs, history, intraday = scenario(3, news=1.)
        bull = simulate(inputs, history, intraday=intraday)["samples"][-1].mean()
        inputs["news"]["value"] = -1.
        bear = simulate(inputs, history, intraday=intraday)["samples"][-1].mean()
        self.assertGreater(bull, bear)


class CalendarTests(unittest.TestCase):
    def test_cl_expiry_rule(self):
        self.assertEqual(cl_expiry(2026, 10).isoformat(), "2026-10-20")   # 25th is a Sunday
        self.assertEqual(cl_expiry(2026, 11).isoformat(), "2026-11-20")   # 25th is a Wednesday
        self.assertEqual(front_expiry_days(datetime(2026, 10, 27, 10, tzinfo=timezone.utc)), 24.)
        self.assertEqual(front_expiry_days(datetime(2026, 12, 28, 10, tzinfo=timezone.utc)),
                         (cl_expiry(2027, 1) - datetime(2026, 12, 28).date()).days)

    def test_market_hours(self):
        self.assertTrue(market_open(OPEN))
        self.assertFalse(market_open(CLOSED))
        self.assertFalse(market_open(datetime(2026, 10, 28, 21, 30, tzinfo=timezone.utc)))  # 17:30 NY break


class DecisionTests(unittest.TestCase):
    def test_barrier_accounting(self):
        spot = 100.
        # path A hits TP first, B hits SL first, C hits neither and ends +0.2, D hits SL then TP
        samples = np.array([[100, 100, 100, 100],
                            [101.1, 98.9, 100.1, 98.9],
                            [100.0, 100.0, 100.2, 101.2]], dtype=np.float32)
        pnl, first_tp, first_sl = barrier_pnl(samples, spot, 1, tp=1., sl=1., shift=0., cost=0.)
        np.testing.assert_allclose(pnl, [1., -1., .2, -1.], atol=1e-5)
        short, _, _ = barrier_pnl(samples, spot, -1, tp=1., sl=1., shift=0., cost=0.)
        np.testing.assert_allclose(short, [-1., 1., -.2, 1.], atol=1e-5)

    def test_wait_when_market_closed_or_quote_stale(self):
        inputs, history, intraday = scenario(2, news=1., now=CLOSED)
        _, result = run(inputs, history, intraday, now=CLOSED)
        self.assertEqual(result["action"], "WAIT")
        inputs, history, intraday = scenario(2, news=1.)
        inputs["price"]["date"] = OPEN - timedelta(hours=2)
        _, result = run(inputs, history, intraday)
        self.assertEqual(result["action"], "WAIT")
        self.assertIsNone(result["tp"])

    def test_wait_without_history_or_quote(self):
        inputs, history, intraday = scenario()
        _, result = run(inputs, history[:20], intraday)
        self.assertEqual(result["action"], "WAIT")
        inputs["price"]["value"] = None
        sim = None
        self.assertEqual(decide(inputs, history, sim, OPEN, 20.)["action"], "WAIT")

    def test_levels_are_ordered_and_sized_sensibly(self):
        seen = set()
        for seed in range(60):
            news = (-1) ** seed * (seed % 5) / 4
            inputs, history, intraday = scenario(seed, news=news)
            sim, r = run(inputs, history, intraday)
            seen.add(r["action"])
            spot = inputs["price"]["value"]
            if r["action"] == "LONG":
                self.assertGreater(r["tp"], spot)
                self.assertLess(r["sl"], spot)
            elif r["action"] == "SHORT":
                self.assertLess(r["tp"], spot)
                self.assertGreater(r["sl"], spot)
            else:
                self.assertIsNone(r["tp"])
                self.assertIsNone(r["sl"])
            if r["tp"] is not None:
                self.assertLess(abs(r["tp"] - spot), 3 * sim["sigma"] * np.sqrt(HORIZON_HOURS / (252 * 23)) + .2)
        self.assertIn("WAIT", seen)
        self.assertTrue({"LONG", "SHORT"} & seen)

    def test_strong_bullish_drivers_flip_direction(self):
        # oil-like mean-reverting history with the quote far below its 60-day median
        rng = np.random.default_rng(7)
        history = list(70 + rng.normal(0, 1.2, 400))
        inputs, _, intraday = scenario(7, news=1.)
        inputs["price"]["value"] = 62.
        inputs["rate"]["value"] = 0.2
        inputs["dollar"]["value"] = -0.2
        up = run(inputs, history, intraday)[1]
        inputs["price"]["value"] = 78.
        inputs["news"]["value"] = -1.
        inputs["rate"]["value"] = 3.0
        inputs["dollar"]["value"] = 0.2
        down = run(inputs, history, intraday)[1]
        self.assertEqual(up["action"], "LONG")
        self.assertEqual(down["action"], "SHORT")


class NewsTests(unittest.TestCase):
    def test_headline_signs(self):
        self.assertGreater(data.score_headline("Oil surges as OPEC+ extends output cuts"), 0)
        self.assertGreater(data.score_headline("Crude inventories fall by 5 million barrels"), 0)
        self.assertLess(data.score_headline("Oil tumbles on demand fears and supply glut"), 0)
        self.assertLess(data.score_headline("Crude stocks build more than expected"), 0)
        self.assertEqual(data.score_headline("Company announces new office"), 0)

    def test_aggregate_news_is_bounded_recent_and_deduplicated(self):
        now = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
        bull = [("Oil surges on supply disruption", now - timedelta(hours=1)),
                ("Crude rallies as inventories draw down", now - timedelta(hours=2)),
                ("OPEC+ extends production cuts", now - timedelta(hours=3))]
        self.assertGreater(data.news_score(bull, now), 0)
        self.assertEqual(data.news_score(bull + bull, now), data.news_score(bull, now))
        old = [(t, s - timedelta(hours=60)) for t, s in bull]
        self.assertEqual(data.news_score(old, now), 0.)
        bear = [("Oil plunges on glut", now - timedelta(hours=1))] * 3 + \
               [("Crude tumbles as demand fears grow", now - timedelta(hours=1)),
                ("Prices slump on recession worries", now - timedelta(hours=1))]
        s = data.news_score(bear, now)
        self.assertTrue(-1 <= s < 0)

    def test_rss_parsing(self):
        xml = ("<rss><channel><item><title>Oil jumps</title>"
               "<pubDate>Wed, 28 Oct 2026 10:00:00 GMT</pubDate></item></channel></rss>")
        items = data.parse_rss(xml)
        self.assertEqual(items[0][0], "Oil jumps")
        self.assertEqual(items[0][1].tzinfo, timezone.utc)


if __name__ == "__main__":
    unittest.main()
