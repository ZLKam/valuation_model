import datetime
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd

from valuation.option_scoring import (
    COMBO_PUT_PROFILE_MAP,
    COMBO_SCORE_WEIGHTS,
    SCORE_WEIGHTS,
    BullishComboPreferences,
    ShortPutPreferences,
    black_scholes_metrics,
    calculate_iv_percentile,
    calculate_iv_rank,
    score_premium_funded_bullish_pairs,
    score_short_put_chain,
)
from valuation.options import (
    QUOTE_BASIS_AUTO,
    QUOTE_BASIS_PREVIOUS_SESSION,
    OptionsAnalyzer,
    _select_underlying_quote,
)


def quote(symbol, strike, bid=0.40, ask=0.42, iv=0.25, dte=35, oi=1_000, volume=100):
    return {
        "contractSymbol": symbol,
        "expiration": "2099-01-01",
        "dte": dte,
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "openInterest": oi,
        "volume": volume,
        "impliedVolatility": iv,
        "ivUsed": iv,
    }


class UnderlyingQuoteSelectionTests(unittest.TestCase):
    class FakeTicker:
        fast_info = {"last_price": None}

        def history(self, **kwargs):
            return pd.DataFrame()

    def test_closed_market_uses_fresh_post_market_quote_not_previous_close(self):
        selected = _select_underlying_quote(
            self.FakeTicker(),
            {
                "marketState": "CLOSED",
                "regularMarketPrice": 695.33,
                "regularMarketTime": 1_784_318_400,
                "postMarketPrice": 693.76,
                "postMarketTime": 1_784_332_800,
                "previousClose": 705.94,
            },
        )

        self.assertEqual(selected["price"], 693.76)
        self.assertEqual(selected["source"], "postMarketPrice")
        self.assertFalse(selected["used_previous_close"])
        self.assertIsNotNone(selected["timestamp"])

    def test_regular_session_does_not_select_an_extended_hours_field(self):
        selected = _select_underlying_quote(
            self.FakeTicker(),
            {
                "marketState": "REGULAR",
                "regularMarketPrice": 100.25,
                "regularMarketTime": 2_000,
                "postMarketPrice": 99.00,
                "postMarketTime": 3_000,
                "previousClose": 98.50,
            },
        )

        self.assertEqual(selected["price"], 100.25)
        self.assertEqual(selected["source"], "regularMarketPrice")

    def test_previous_close_is_last_resort_and_is_flagged(self):
        selected = _select_underlying_quote(
            self.FakeTicker(),
            {"marketState": "CLOSED", "previousClose": 88.0},
        )

        self.assertEqual(selected["price"], 88.0)
        self.assertEqual(selected["source"], "previousClose")
        self.assertTrue(selected["used_previous_close"])

    def test_timestamped_intraday_trade_beats_untimestamped_quote_field(self):
        class IntradayTicker:
            fast_info = {"last_price": None}

            def history(self, **kwargs):
                index = pd.DatetimeIndex(["2026-07-17T19:59:00-04:00"])
                return pd.DataFrame({"Close": [101.25]}, index=index)

        selected = _select_underlying_quote(
            IntradayTicker(),
            {"marketState": "REGULAR", "currentPrice": 99.0, "previousClose": 98.0},
        )

        self.assertEqual(selected["price"], 101.25)
        self.assertEqual(selected["source"], "latest_1m_close")
        self.assertIsNotNone(selected["timestamp"])


class OptionMathTests(unittest.TestCase):
    def test_put_greeks_and_probability_have_expected_signs(self):
        metrics = black_scholes_metrics(100.0, 90.0, 35 / 365.25, 0.04, 0.25, 0.01, "put")
        self.assertLess(metrics["delta"], 0.0)
        self.assertGreater(metrics["gamma"], 0.0)
        self.assertLess(metrics["theta_per_day"], 0.0)
        self.assertGreater(metrics["vega_per_vol_point"], 0.0)
        self.assertGreater(metrics["probability_itm"], 0.0)
        self.assertLess(metrics["probability_itm"], 0.20)

    def test_iv_rank_and_percentile_are_distinct_and_bounded(self):
        history = [0.10, 0.15, 0.20, 0.40]
        self.assertAlmostEqual(calculate_iv_rank(0.25, history), 0.50)
        self.assertAlmostEqual(calculate_iv_percentile(0.25, history), 0.75)
        self.assertIsNone(calculate_iv_rank(0.20, [0.20, 0.20]))


class ShortPutScoringTests(unittest.TestCase):
    def setUp(self):
        self.preferences = ShortPutPreferences.for_profile(
            "lowest_risk",
            max_assignment_probability=0.25,
            min_open_interest=0,
            max_bid_ask_spread_pct=0.50,
        )

    def test_profile_weights_are_complete_and_sum_to_one(self):
        expected_components = {
            "risk_target_fit",
            "breakeven_safety",
            "liquidity",
            "greek_risk",
            "dte_theta",
            "premium_efficiency",
            "volatility_context",
        }
        for weights in SCORE_WEIGHTS.values():
            self.assertEqual(set(weights), expected_components)
            self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_modes_supply_distinct_default_assignment_ceilings(self):
        expected = {
            "lowest_risk": (0.15, 35),
            "balanced": (0.28, 38),
            "income_focused": (0.50, 30),
        }

        for profile, (ceiling, target_dte) in expected.items():
            with self.subTest(profile=profile):
                preferences = ShortPutPreferences.for_profile(profile)
                self.assertEqual(preferences.max_assignment_probability, ceiling)
                self.assertEqual((preferences.min_dte, preferences.max_dte), (30, 45))
                self.assertEqual(preferences.target_dte, target_dte)

        self.assertEqual(
            COMBO_PUT_PROFILE_MAP,
            {
                "downside_aware": "lowest_risk",
                "balanced": "balanced",
                "upside_focused": "income_focused",
            },
        )

    def test_custom_dte_window_preserves_or_clamps_profile_target(self):
        balanced = ShortPutPreferences.for_profile("balanced")

        self.assertEqual(balanced.with_dte_window(30, 45).target_dte, 38)
        self.assertEqual(balanced.with_dte_window(7, 20).target_dte, 20)
        self.assertEqual(balanced.with_dte_window(45, 60).target_dte, 45)

    def test_lower_risk_contract_ranks_first_when_quotes_match(self):
        result = score_short_put_chain(
            [quote("LOW_RISK", 85.0), quote("HIGHER_RISK", 90.0)],
            100.0,
            0.20,
            self.preferences,
            iv_context={"iv_rank": 0.70, "source": "test"},
        )
        self.assertEqual(result["recommendations"][0]["contract_symbol"], "LOW_RISK")
        self.assertLess(
            result["recommendations"][0]["estimated_assignment_probability"],
            result["recommendations"][1]["estimated_assignment_probability"],
        )

    def test_hard_risk_and_liquidity_limits_are_enforced(self):
        strict = self.preferences.with_overrides(
            max_assignment_probability=0.10,
            min_open_interest=500,
            max_bid_ask_spread_pct=0.10,
        )
        result = score_short_put_chain(
            [
                quote("ASSIGNMENT", 95.0, oi=1_000),
                quote("NO_OI", 85.0, oi=10),
                quote("WIDE", 85.0, bid=0.20, ask=0.40, oi=1_000),
                quote("PASS", 85.0, oi=1_000),
            ],
            100.0,
            0.20,
            strict,
        )
        self.assertEqual([item["contract_symbol"] for item in result["recommendations"]], ["PASS"])
        self.assertEqual(result["rejected_counts"]["assignment estimate above limit"], 1)
        self.assertEqual(result["rejected_counts"]["open interest below minimum"], 1)
        self.assertEqual(result["rejected_counts"]["bid/ask spread too wide"], 1)

    def test_seller_economics_use_executable_bid_and_short_greek_signs(self):
        result = score_short_put_chain(
            [quote("BID", 85.0, bid=0.40, ask=0.60)],
            100.0,
            0.20,
            self.preferences,
        )["recommendations"][0]
        self.assertEqual(result["premium_per_contract"], 40.0)
        self.assertEqual(result["break_even"], 84.60)
        self.assertGreater(result["short_position_delta"], 0.0)
        self.assertGreater(result["short_position_theta_per_day"], 0.0)
        self.assertLess(result["short_position_gamma"], 0.0)
        self.assertLess(result["short_position_vega_per_vol_point"], 0.0)

    def test_cash_secured_limit_filters_contract(self):
        limited = self.preferences.with_overrides(max_cash_secured=8_700.0)
        result = score_short_put_chain(
            [quote("FITS", 85.0), quote("TOO_LARGE", 90.0)],
            100.0,
            0.20,
            limited,
        )
        self.assertEqual([item["contract_symbol"] for item in result["recommendations"]], ["FITS"])
        self.assertEqual(result["rejected_counts"]["cash required exceeds limit"], 1)

    def test_income_profile_rejects_tiny_delta_and_targets_material_income_risk(self):
        preferences = ShortPutPreferences.for_profile(
            "income_focused",
            max_assignment_probability=0.55,
            min_open_interest=0,
            max_bid_ask_spread_pct=0.50,
            limit=10,
        )
        result = score_short_put_chain(
            [
                quote("TINY_DELTA", 90.0, bid=0.50, ask=0.52),
                quote("TARGET_DELTA", 98.0, bid=3.00, ask=3.05),
            ],
            100.0,
            0.20,
            preferences,
        )

        self.assertEqual([item["contract_symbol"] for item in result["recommendations"]], ["TARGET_DELTA"])
        self.assertGreaterEqual(result["recommendations"][0]["short_position_delta"], 0.25)
        self.assertLessEqual(result["recommendations"][0]["short_position_delta"], 0.50)
        self.assertEqual(result["rejected_counts"]["short delta outside profile range"], 1)

    def test_all_csp_profiles_enforce_distinct_delta_objectives(self):
        contracts = [
            quote("LOWEST_TARGET", 90.0, bid=2.00, ask=2.10),
            quote("BALANCED_TARGET", 95.0, bid=2.00, ask=2.10),
            quote("INCOME_TARGET", 99.0, bid=2.00, ask=2.10),
        ]
        expected = {
            "lowest_risk": ("LOWEST_TARGET", 0.01, 0.18, 0.08),
            "balanced": ("BALANCED_TARGET", 0.10, 0.35, 0.22),
            "income_focused": ("INCOME_TARGET", 0.25, 0.50, 0.40),
        }

        for profile, (symbol, low, high, target) in expected.items():
            with self.subTest(profile=profile):
                preferences = ShortPutPreferences.for_profile(
                    profile,
                    max_assignment_probability=0.60,
                    min_open_interest=0,
                    max_bid_ask_spread_pct=0.50,
                    limit=10,
                )
                result = score_short_put_chain(
                    contracts,
                    100.0,
                    0.20,
                    preferences,
                )
                recommendation = result["recommendations"][0]
                self.assertEqual(recommendation["contract_symbol"], symbol)
                self.assertGreaterEqual(recommendation["short_position_delta"], low)
                self.assertLessEqual(recommendation["short_position_delta"], high)
                self.assertEqual(result["constraints"]["short_delta"], [low, high])
                self.assertEqual(result["constraints"]["target_short_delta"], target)


class BullishComboScoringTests(unittest.TestCase):
    def setUp(self):
        put_preferences = ShortPutPreferences.for_profile(
            "balanced",
            max_assignment_probability=0.25,
            min_open_interest=0,
            max_bid_ask_spread_pct=0.50,
            min_short_delta=0.01,
            target_short_delta=0.10,
            max_short_delta=0.35,
            limit=20,
        )
        self.put_candidates = score_short_put_chain(
            [quote("PUT90", 90.0, bid=2.00, ask=2.10, iv=0.25)],
            100.0,
            0.20,
            put_preferences,
        )["recommendations"]
        self.combo_preferences = BullishComboPreferences.for_profile(
            "balanced",
            min_call_delta=0.10,
            max_call_delta=0.60,
            min_call_open_interest=0,
            max_call_spread_pct=0.50,
            min_premium_utilization=0.0,
            max_extra_debit=0.0,
            limit=10,
        )

    @staticmethod
    def call_quote(symbol, strike, bid, ask, expiration="2099-01-01", iv=0.25):
        return {
            "contractSymbol": symbol,
            "expiration": expiration,
            "dte": 35,
            "strike": strike,
            "bid": bid,
            "ask": ask,
            "openInterest": 1_000,
            "volume": 100,
            "impliedVolatility": iv,
            "ivUsed": iv,
        }

    def test_combo_weights_are_complete_and_sum_to_one(self):
        expected = {
            "downside_safety",
            "upside_participation",
            "funding_efficiency",
            "liquidity",
            "net_greeks",
            "iv_skew",
            "dte_fit",
        }
        for weights in COMBO_SCORE_WEIGHTS.values():
            self.assertEqual(set(weights), expected)
            self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_all_combo_modes_inherit_the_intended_put_delta_profile(self):
        contracts = [
            quote("LOWEST_TARGET", 90.0, bid=2.00, ask=2.10),
            quote("BALANCED_TARGET", 95.0, bid=2.00, ask=2.10),
            quote("INCOME_TARGET", 99.0, bid=2.00, ask=2.10),
        ]
        call = self.call_quote("CALL105", 105.0, 1.70, 1.80)
        wrong_expiry_call = self.call_quote(
            "WRONG_EXPIRY",
            105.0,
            1.70,
            1.80,
            expiration="2099-02-01",
        )
        modes = {
            "downside_aware": ("lowest_risk", "LOWEST_TARGET", 0.01, 0.18),
            "balanced": ("balanced", "BALANCED_TARGET", 0.10, 0.35),
            "upside_focused": ("income_focused", "INCOME_TARGET", 0.25, 0.50),
        }

        for combo_profile, (put_profile, symbol, low, high) in modes.items():
            with self.subTest(combo_profile=combo_profile):
                put_preferences = ShortPutPreferences.for_profile(
                    put_profile,
                    max_assignment_probability=0.60,
                    min_open_interest=0,
                    max_bid_ask_spread_pct=0.50,
                    limit=10,
                )
                puts = score_short_put_chain(
                    contracts,
                    100.0,
                    0.20,
                    put_preferences,
                )["recommendations"]
                combo_preferences = BullishComboPreferences.for_profile(
                    combo_profile,
                    min_call_open_interest=0,
                    max_call_spread_pct=0.50,
                    min_premium_utilization=0.0,
                    max_extra_debit=0.0,
                    limit=10,
                )
                result = score_premium_funded_bullish_pairs(
                    puts,
                    [call, wrong_expiry_call],
                    100.0,
                    0.20,
                    combo_preferences,
                )
                recommendation = result["recommendations"][0]
                self.assertEqual(recommendation["put_contract_symbol"], symbol)
                self.assertEqual(recommendation["call_contract_symbol"], "CALL105")
                self.assertEqual(recommendation["expiration"], "2099-01-01")
                self.assertGreaterEqual(recommendation["short_put_delta"], low)
                self.assertLessEqual(recommendation["short_put_delta"], high)
                self.assertGreaterEqual(recommendation["net_credit"], 0.0)

    def test_put_bid_funds_call_ask_and_payoff_is_correct(self):
        result = score_premium_funded_bullish_pairs(
            self.put_candidates,
            [self.call_quote("CALL105", 105.0, 1.70, 1.80)],
            100.0,
            0.20,
            self.combo_preferences,
        )["recommendations"][0]
        self.assertEqual(result["put_credit"], 200.0)
        self.assertEqual(result["call_cost"], 180.0)
        self.assertAlmostEqual(result["net_credit"], 20.0)
        self.assertAlmostEqual(result["lower_break_even"], 89.80)
        self.assertAlmostEqual(result["max_loss"], 8_980.0)
        self.assertAlmostEqual(result["profit_at_up_10_pct"], 520.0)
        self.assertAlmostEqual(result["profit_at_down_10_pct"], 20.0)
        self.assertGreater(result["net_delta"], 0.0)

    def test_call_cost_above_put_credit_is_rejected_without_extra_budget(self):
        result = score_premium_funded_bullish_pairs(
            self.put_candidates,
            [self.call_quote("TOO_EXPENSIVE", 103.0, 2.00, 2.20)],
            100.0,
            0.20,
            self.combo_preferences,
        )
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["rejected_counts"]["call ask exceeds put-credit budget"], 1)

    def test_higher_delta_call_ranks_first_when_cost_and_liquidity_match(self):
        result = score_premium_funded_bullish_pairs(
            self.put_candidates,
            [
                self.call_quote("NEARER", 103.0, 1.70, 1.80),
                self.call_quote("FARTHER", 108.0, 1.70, 1.80),
            ],
            100.0,
            0.20,
            self.combo_preferences,
        )
        self.assertEqual(result["recommendations"][0]["call_contract_symbol"], "NEARER")
        self.assertGreater(
            result["recommendations"][0]["long_call_delta"],
            result["recommendations"][1]["long_call_delta"],
        )

    def test_optional_extra_cash_is_capped_and_reported_as_net_debit(self):
        preferences = self.combo_preferences.with_overrides(
            max_extra_debit=50.0,
            min_premium_utilization=0.90,
        )
        result = score_premium_funded_bullish_pairs(
            self.put_candidates,
            [self.call_quote("EXTRA", 103.0, 2.30, 2.40)],
            100.0,
            0.20,
            preferences,
        )["recommendations"][0]
        self.assertAlmostEqual(result["additional_debit"], 40.0)
        self.assertAlmostEqual(result["net_debit"], 40.0)
        self.assertAlmostEqual(result["premium_utilization"], 1.0)
        self.assertAlmostEqual(result["upper_break_even"], 103.40)


class OptionsAnalyzerTests(unittest.TestCase):
    def test_scans_multiple_expirations_and_attaches_proxy_iv_rank(self):
        today = datetime.date.today()
        expirations = [(today + datetime.timedelta(days=days)).isoformat() for days in (28, 35)]

        class FakeTicker:
            info = {"currentPrice": 100.0, "dividendYield": 0.01}
            fast_info = {"last_price": 100.0}
            options = expirations

            def history(self, period="3mo", auto_adjust=True):
                if period == "1y":
                    return pd.DataFrame({"Close": [15.0 + index * 0.1 for index in range(30)]})
                return pd.DataFrame({"Close": [90.0 + index * 0.2 for index in range(60)]})

            def option_chain(self, expiration):
                dte = (datetime.date.fromisoformat(expiration) - today).days
                frame = pd.DataFrame(
                    [
                        {
                            "contractSymbol": f"QQQ{dte}P85",
                            "strike": 85.0,
                            "lastPrice": 0.40,
                            "bid": 0.40,
                            "ask": 0.42,
                            "impliedVolatility": 0.25,
                            "volume": 100,
                            "openInterest": 1_000,
                            "inTheMoney": False,
                        }
                    ]
                )
                return types.SimpleNamespace(calls=pd.DataFrame(), puts=frame)

        fake_underlying = FakeTicker()
        fake_proxy = FakeTicker()
        with patch("valuation.options.yf.Ticker", side_effect=lambda symbol: fake_proxy if symbol == "^VXN" else fake_underlying):
            preferences = ShortPutPreferences.for_profile(
                "lowest_risk",
                min_dte=21,
                max_dte=45,
                target_dte=35,
                max_assignment_probability=0.25,
                min_open_interest=0,
                max_bid_ask_spread_pct=0.50,
            )
            result = OptionsAnalyzer().recommend_short_puts("QQQ", preferences)

        self.assertEqual(result["input_count"], 2)
        self.assertEqual(len(result["scanned_expirations"]), 2)
        self.assertTrue(result["iv_context"]["is_proxy"])
        self.assertIsNotNone(result["iv_context"]["iv_rank"])
        self.assertEqual(result["recommendations"][0]["rank"], 1)

    def test_joint_scanner_builds_same_expiration_funded_pair(self):
        today = datetime.date.today()
        expiration = (today + datetime.timedelta(days=35)).isoformat()

        class FakeTicker:
            info = {"currentPrice": 100.0, "dividendYield": 0.01}
            fast_info = {"last_price": 100.0}
            options = [expiration]

            def history(self, period="3mo", auto_adjust=True):
                if period == "1y":
                    return pd.DataFrame({"Close": [15.0 + index * 0.1 for index in range(30)]})
                return pd.DataFrame({"Close": [90.0 + index * 0.2 for index in range(60)]})

            def option_chain(self, selected_expiration):
                common = {
                    "lastPrice": 1.90,
                    "impliedVolatility": 0.25,
                    "volume": 100,
                    "openInterest": 1_000,
                    "inTheMoney": False,
                }
                puts = pd.DataFrame(
                    [{**common, "contractSymbol": "QQQP90", "strike": 90.0, "bid": 2.00, "ask": 2.10}]
                )
                calls = pd.DataFrame(
                    [{**common, "contractSymbol": "QQQC105", "strike": 105.0, "bid": 1.70, "ask": 1.80}]
                )
                return types.SimpleNamespace(calls=calls, puts=puts)

        fake_underlying = FakeTicker()
        fake_proxy = FakeTicker()
        put_preferences = ShortPutPreferences.for_profile(
            "balanced",
            min_dte=21,
            max_dte=45,
            target_dte=35,
            max_assignment_probability=0.25,
            min_open_interest=0,
            max_bid_ask_spread_pct=0.50,
            min_short_delta=0.01,
            target_short_delta=0.10,
            max_short_delta=0.35,
        )
        combo_preferences = BullishComboPreferences.for_profile(
            "balanced",
            min_call_delta=0.10,
            max_call_delta=0.60,
            min_call_open_interest=0,
            max_call_spread_pct=0.50,
            min_premium_utilization=0.0,
            max_extra_debit=0.0,
        )
        with patch(
            "valuation.options.yf.Ticker",
            side_effect=lambda symbol: fake_proxy if symbol == "^VXN" else fake_underlying,
        ):
            result = OptionsAnalyzer().recommend_premium_funded_bullish_combo(
                "QQQ", put_preferences, combo_preferences
            )

        self.assertEqual(result["strategy"], "premium_funded_bullish_risk_reversal")
        self.assertEqual(result["eligible_short_put_count"], 1)
        self.assertEqual(result["recommendations"][0]["put_contract_symbol"], "QQQP90")
        self.assertEqual(result["recommendations"][0]["call_contract_symbol"], "QQQC105")
        self.assertGreaterEqual(result["recommendations"][0]["net_credit"], 0.0)


class OptionSnapshotTests(unittest.TestCase):
    capture_date = datetime.date(2026, 7, 21)
    expiration = (capture_date + datetime.timedelta(days=35)).isoformat()

    class RegularTicker:
        info = {
            "marketState": "REGULAR",
            "regularMarketPrice": 100.0,
            "regularMarketTime": int(
                datetime.datetime(2026, 7, 21, 19, 45, tzinfo=datetime.timezone.utc).timestamp()
            ),
            "dividendYield": 0.01,
        }
        fast_info = {"last_price": 100.0}
        options = []

        def history(self, period="3mo", auto_adjust=True, **kwargs):
            if period == "1y":
                return pd.DataFrame({"Close": [15.0 + index * 0.1 for index in range(30)]})
            return pd.DataFrame({"Close": [90.0 + index * 0.2 for index in range(60)]})

        def option_chain(self, selected_expiration):
            common = {
                "lastPrice": 1.90,
                "impliedVolatility": 0.25,
                "volume": 100,
                "openInterest": 1_000,
                "inTheMoney": False,
                "lastTradeDate": pd.Timestamp("2026-07-21T15:44:00-04:00"),
            }
            puts = pd.DataFrame(
                [{**common, "contractSymbol": "QQQP90", "strike": 90.0, "bid": 2.00, "ask": 2.10}]
            )
            calls = pd.DataFrame(
                [{**common, "contractSymbol": "QQQC105", "strike": 105.0, "bid": 1.70, "ask": 1.80}]
            )
            return types.SimpleNamespace(calls=calls, puts=puts)

    class ProxyTicker(RegularTicker):
        pass

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.analyzer = OptionsAnalyzer(snapshot_dir=self.temporary_directory.name)
        self.put_preferences = ShortPutPreferences.for_profile(
            "balanced",
            min_dte=30,
            max_dte=45,
            target_dte=35,
            max_assignment_probability=0.40,
            min_open_interest=0,
            max_bid_ask_spread_pct=0.50,
            min_short_delta=0.01,
            target_short_delta=0.10,
            max_short_delta=0.45,
        )

    def capture_snapshot(self):
        regular_ticker = self.RegularTicker()
        regular_ticker.options = [self.expiration]
        proxy_ticker = self.ProxyTicker()
        with (
            patch("valuation.options._market_date", return_value=self.capture_date),
            patch(
                "valuation.options.yf.Ticker",
                side_effect=lambda symbol: proxy_ticker if symbol == "^VXN" else regular_ticker,
            ),
        ):
            result = self.analyzer.recommend_short_puts(
                "QQQ", self.put_preferences, r_rate=0.046, quote_basis="live"
            )
        self.assertTrue(result["snapshot_saved"])
        return result

    def test_regular_session_scan_saves_an_aligned_two_sided_chain(self):
        result = self.capture_snapshot()
        snapshot = self.analyzer.snapshot_store.load_latest("QQQ")

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["market_date"], self.capture_date.isoformat())
        self.assertEqual(snapshot["spot"], 100.0)
        self.assertEqual(snapshot["risk_free_rate"], 0.046)
        self.assertEqual(snapshot["marketable_puts"], 1)
        self.assertEqual(snapshot["marketable_calls"], 1)
        self.assertEqual(snapshot["marketable_otm_puts"], 1)
        self.assertEqual(snapshot["marketable_otm_calls"], 1)
        stored_put = snapshot["chains"][self.expiration]["puts"][0]
        self.assertEqual(stored_put["contractSymbol"], "QQQP90")
        self.assertNotIn("lastTradeDate", stored_put)
        self.assertNotIn("gamma", stored_put)
        self.assertEqual(result["quote_basis_used"], "live")
        self.assertEqual(result["marketable_otm_put_count"], 1)

    def test_regular_session_does_not_save_chain_with_only_marketable_itm_puts(self):
        class ItmOnlyTicker(self.RegularTicker):
            def option_chain(self, selected_expiration):
                common = {
                    "lastPrice": 1.0,
                    "impliedVolatility": 0.25,
                    "volume": 100,
                    "openInterest": 1_000,
                    "inTheMoney": False,
                }
                puts = pd.DataFrame(
                    [
                        {**common, "contractSymbol": "QQQP110", "strike": 110.0, "bid": 10.0, "ask": 10.2},
                        {**common, "contractSymbol": "QQQP90", "strike": 90.0, "bid": 0.0, "ask": 0.0},
                    ]
                )
                calls = pd.DataFrame(
                    [{**common, "contractSymbol": "QQQC105", "strike": 105.0, "bid": 1.7, "ask": 1.8}]
                )
                return types.SimpleNamespace(calls=calls, puts=puts)

        regular_ticker = ItmOnlyTicker()
        regular_ticker.options = [self.expiration]
        proxy_ticker = self.ProxyTicker()
        with (
            patch("valuation.options._market_date", return_value=self.capture_date),
            patch(
                "valuation.options.yf.Ticker",
                side_effect=lambda symbol: proxy_ticker if symbol == "^VXN" else regular_ticker,
            ),
        ):
            result = self.analyzer.recommend_short_puts(
                "QQQ", self.put_preferences, r_rate=0.046, quote_basis="live"
            )

        self.assertFalse(result["snapshot_saved"])
        self.assertEqual(result["snapshot_save_status"], "no_marketable_otm_puts")
        self.assertEqual(result["marketable_put_count"], 1)
        self.assertEqual(result["marketable_otm_put_count"], 0)
        self.assertEqual(result["data_readiness"], "no_marketable_otm_puts")
        self.assertIsNone(self.analyzer.snapshot_store.load_latest("QQQ"))

    def test_existing_bad_snapshot_is_flagged_during_replay(self):
        bad_snapshot = {
            "symbol": "QQQ",
            "market_date": self.capture_date.isoformat(),
            "captured_at": "2026-07-21T19:45:00+00:00",
            "market_state": "REGULAR",
            "spot": 100.0,
            "historical_volatility": 0.20,
            "dividend_yield": 0.01,
            "risk_free_rate": 0.046,
            "iv_context": {"iv_rank": 0.50, "source": "test snapshot"},
            "listed_expirations": [self.expiration],
            "chains": {
                self.expiration: {
                    "puts": [
                        quote("ITM", 110.0, bid=10.0, ask=10.2),
                        quote("OTM_ZERO", 90.0, bid=0.0, ask=0.0),
                    ],
                    "calls": [],
                }
            },
        }
        self.analyzer.snapshot_store.save(bad_snapshot)

        replay = self.analyzer.recommend_short_puts(
            "QQQ", self.put_preferences, quote_basis=QUOTE_BASIS_PREVIOUS_SESSION
        )

        self.assertEqual(replay["marketable_put_count"], 1)
        self.assertEqual(replay["marketable_otm_put_count"], 0)
        self.assertEqual(replay["data_readiness"], "no_marketable_otm_puts")
        self.assertEqual(replay["recommendations"], [])

    def test_previous_session_replay_uses_snapshot_date_and_never_fetches_live_chain(self):
        live_result = self.capture_snapshot()
        later_date = self.capture_date + datetime.timedelta(days=5)
        with (
            patch("valuation.options._market_date", return_value=later_date),
            patch("valuation.options.yf.Ticker", side_effect=AssertionError("replay must not fetch live data")),
        ):
            replay = self.analyzer.recommend_short_puts(
                "QQQ",
                self.put_preferences,
                r_rate=0.01,
                quote_basis=QUOTE_BASIS_PREVIOUS_SESSION,
            )

        self.assertTrue(replay["is_snapshot"])
        self.assertEqual(replay["quote_basis_used"], QUOTE_BASIS_PREVIOUS_SESSION)
        self.assertEqual(replay["market_date"], self.capture_date.isoformat())
        self.assertEqual(replay["recommendations"][0]["dte"], 35)
        self.assertEqual(replay["risk_free_rate"], 0.046)
        self.assertEqual(replay["spot"], live_result["spot"])
        self.assertIn("not currently executable", replay["spot_warning"])

    def test_replay_ignores_listed_expirations_that_were_not_captured(self):
        self.capture_snapshot()
        snapshot = self.analyzer.snapshot_store.load_latest("QQQ")
        snapshot["listed_expirations"].append((self.capture_date + datetime.timedelta(days=40)).isoformat())
        self.analyzer.snapshot_store.save(snapshot)

        with patch("valuation.options.yf.Ticker", side_effect=AssertionError("replay must remain offline")):
            replay = self.analyzer.recommend_short_puts(
                "QQQ", self.put_preferences, quote_basis=QUOTE_BASIS_PREVIOUS_SESSION
            )

        self.assertNotIn("error", replay)
        self.assertEqual(replay["scanned_expirations"], [self.expiration])
        self.assertEqual(replay["recommendations"][0]["contract_symbol"], "QQQP90")

    def test_auto_off_hours_replays_snapshot_for_both_put_and_call_legs(self):
        self.capture_snapshot()

        class PremarketTicker:
            info = {"marketState": "PRE", "preMarketPrice": 103.0}

            @property
            def options(self):
                raise AssertionError("Auto snapshot mode must not fetch a live chain")

        combo_preferences = BullishComboPreferences.for_profile(
            "balanced",
            min_call_delta=0.10,
            max_call_delta=0.60,
            min_call_open_interest=0,
            max_call_spread_pct=0.50,
            min_premium_utilization=0.0,
            max_extra_debit=0.0,
        )
        with patch("valuation.options.yf.Ticker", return_value=PremarketTicker()):
            result = self.analyzer.recommend_premium_funded_bullish_combo(
                "QQQ",
                self.put_preferences,
                combo_preferences,
                quote_basis=QUOTE_BASIS_AUTO,
            )

        self.assertTrue(result["is_snapshot"])
        self.assertEqual(result["spot"], 100.0)
        self.assertEqual(result["recommendations"][0]["put_contract_symbol"], "QQQP90")
        self.assertEqual(result["recommendations"][0]["call_contract_symbol"], "QQQC105")

    def test_missing_snapshot_explains_how_to_create_one(self):
        with patch("valuation.options.yf.Ticker", side_effect=AssertionError("explicit replay is offline")):
            result = self.analyzer.recommend_short_puts(
                "NVDA", self.put_preferences, quote_basis=QUOTE_BASIS_PREVIOUS_SESSION
            )

        self.assertIn("No saved regular-session option snapshot", result["error"])
        self.assertIn("Run one live scan", result["error"])

    def test_live_basis_refuses_to_mix_premarket_spot_with_an_old_chain(self):
        class PremarketTicker:
            info = {"marketState": "PRE", "preMarketPrice": 103.0}

            @property
            def options(self):
                raise AssertionError("Off-hours live mode must stop before loading the chain")

        with patch("valuation.options.yf.Ticker", return_value=PremarketTicker()):
            result = self.analyzer.recommend_short_puts(
                "QQQ", self.put_preferences, quote_basis="live"
            )

        self.assertIn("Current-session option quotes are unavailable", result["error"])
        self.assertFalse(result["is_snapshot"])


if __name__ == "__main__":
    unittest.main()
