import unittest

import numpy as np
import pandas as pd

from valuation.assumptions import AssumptionEngine, classify_company, is_market_leader
from valuation.dcf import DCFModel, DCF_API_VERSION
from valuation.engine import ValuationEngine, _build_dcf_model
from valuation.fundamentals import EarningsPowerModel, ResidualIncomeModel
from valuation.multiples import MultiplesModel
from valuation.wacc import WACCCalculator


def sample_financials(fcf_values=None):
    dates = pd.to_datetime(["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"])
    return pd.DataFrame(
        {
            "revenue": [80.0, 90.0, 100.0, 112.0],
            "fcf": fcf_values or [8.0, 9.5, 11.0, 13.0],
            "net_income": [6.0, 7.0, 8.0, 9.5],
            "ebitda": [12.0, 14.0, 16.0, 18.0],
            "equity": [30.0, 35.0, 41.0, 48.0],
            "total_cash": [5.0, 6.0, 8.0, 10.0],
            "total_debt": [12.0, 11.0, 10.0, 9.0],
            "interest_expense": [0.8, 0.75, 0.7, 0.65],
            "pretax_income": [7.5, 8.5, 10.0, 12.0],
            "tax_provision": [1.5, 1.7, 2.0, 2.4],
        },
        index=dates,
    )


class AssumptionTests(unittest.TestCase):
    def test_profile_selection(self):
        self.assertEqual(classify_company({"sector": "Financial Services", "industry": "Banks"}), "FINANCIAL")
        self.assertEqual(
            classify_company({"sector": "Technology", "industry": "Software", "revenueGrowth": 0.25}),
            "HIGH_GROWTH",
        )

    def test_extreme_rebound_is_treated_as_cyclical(self):
        info = {
            "sector": "Technology",
            "industry": "Semiconductors",
            "revenueGrowth": 3.4,
            "earningsGrowth": 12.0,
        }
        self.assertEqual(classify_company(info), "CYCLICAL")

    def test_market_leader_is_detected_from_observable_characteristics(self):
        info = {
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 3e12,
            "numberOfAnalystOpinions": 35,
        }
        self.assertTrue(is_market_leader(info))
        self.assertFalse(is_market_leader({**info, "marketCap": 20e9}))

    def test_automatic_inputs_are_bounded_and_explainable(self):
        info = {"sector": "Technology", "industry": "Software", "revenueGrowth": 0.18, "earningsGrowth": 0.20}
        assumptions = AssumptionEngine().derive(info, sample_financials(), 0.04, 0.095)
        self.assertGreater(assumptions.initial_growth, 0.08)
        self.assertLessEqual(assumptions.initial_growth, 0.35)
        self.assertGreater(assumptions.target_fcf_margin, 0.0)
        self.assertLess(assumptions.terminal_growth, 0.095)
        self.assertIn("growth", assumptions.sources)


class DCFTests(unittest.TestCase):
    def setUp(self):
        self.financials = sample_financials()
        self.wacc = {"wacc": 0.095, "tax_rate": 0.21, "total_debt": 9.0, "total_cash": 10.0}

    def test_multistage_scenarios_are_ordered(self):
        result = DCFModel().calculate(
            self.financials, self.wacc, 0.10, 0.022, shares_outstanding=10.0,
            current_price=12.0, target_fcf_margin=0.12,
        )
        self.assertNotIn("error", result)
        self.assertLess(result["bear"]["target_price"], result["base"]["target_price"])
        self.assertLess(result["base"]["target_price"], result["bull"]["target_price"])
        self.assertEqual(len(result["base"]["projected_fcfs"]), 10)
        self.assertGreater(result["base"]["terminal_value_share"], 0.0)

    def test_engine_and_dcf_api_contract_match(self):
        self.assertEqual(DCF_API_VERSION, 2)
        self.assertIsInstance(_build_dcf_model(), DCFModel)

    def test_negative_steady_state_cash_flow_is_not_fabricated(self):
        result = DCFModel().calculate(
            sample_financials([-8.0, -7.0, -6.0, -5.0]), self.wacc, 0.12, 0.02,
            shares_outstanding=10.0, current_price=12.0, target_fcf_margin=-0.05,
        )
        self.assertIn("error", result)
        self.assertIn("not been established", result["error"])

    def test_reverse_dcf_recovers_market_growth(self):
        model = DCFModel()
        result = model.calculate(
            self.financials, self.wacc, 0.08, 0.02, shares_outstanding=10.0,
            current_price=10.0, target_fcf_margin=0.12, include_sensitivity=False,
        )
        market_price = result["base"]["target_price"]
        implied = model.implied_growth_rate(
            self.financials, self.wacc, 0.02, 10.0, market_price, 0.12,
        )
        self.assertIsNotNone(implied)
        self.assertAlmostEqual(implied, 0.08, places=3)


class ComplementaryModelTests(unittest.TestCase):
    def test_earnings_power_uses_normalized_history(self):
        result = EarningsPowerModel().calculate(sample_financials(), shares=10.0, cost_of_equity=0.10)
        self.assertLess(result["bear"], result["base"])
        self.assertLess(result["base"], result["bull"])

    def test_residual_income_for_financials(self):
        result = ResidualIncomeModel().calculate(
            sample_financials(), {"sharesOutstanding": 10.0, "dividendRate": 0.2, "trailingEps": 0.9},
            cost_of_equity=0.10, terminal_growth=0.02,
        )
        self.assertNotIn("error", result)
        self.assertGreater(result["base"], 0.0)

    def test_ensemble_confidence_and_verdict(self):
        methods = [
            {"name": "A", "bear": 80.0, "base": 105.0, "bull": 130.0, "weight": 0.6},
            {"name": "B", "bear": 75.0, "base": 110.0, "bull": 140.0, "weight": 0.4},
        ]
        result = ValuationEngine._combine(methods, 100.0, 4, {"confidence": 0.8})
        self.assertAlmostEqual(result["base"], 107.0)
        self.assertEqual(result["verdict"], "Near fair value")
        self.assertGreater(result["confidence_score"], 50)

    def test_zero_weight_method_cannot_remove_configured_method(self):
        methods = [
            {"name": "Cash-flow value", "bear": 15.0, "base": 50.0, "bull": 118.0, "confidence": 0.8},
            {"name": "Trading range", "bear": 136.0, "base": 192.0, "bull": 395.0, "confidence": 0.6},
            {"name": "Earnings power", "bear": 3.0, "base": 5.0, "bull": 7.0, "confidence": 0.8},
        ]
        usable, _ = ValuationEngine._prepare_methods(methods, "HIGH_GROWTH", 255.0)
        self.assertEqual({method["name"] for method in usable}, {"Cash-flow value", "Trading range"})

    def test_single_low_confidence_method_is_marked_directional(self):
        methods = [
            {"name": "Trading range", "bear": 80.0, "base": 100.0, "bull": 130.0, "confidence": 0.4},
        ]
        usable, risks = ValuationEngine._prepare_methods(methods, "STANDARD", 100.0)
        self.assertEqual(len(usable), 1)
        self.assertTrue(any("directional" in risk for risk in risks))

    def test_sixfold_method_disagreement_is_withheld(self):
        methods = [
            {"name": "Trading range", "bear": 900.0, "base": 1500.0, "bull": 1800.0, "confidence": 0.5},
            {"name": "Earnings power", "bear": 20.0, "base": 30.0, "bull": 40.0, "confidence": 0.8},
        ]
        usable, risks = ValuationEngine._prepare_methods(methods, "CYCLICAL", 900.0)
        self.assertEqual(usable, [])
        self.assertTrue(any("sixfold" in risk for risk in risks))

    def test_institutional_outlook_is_discounted_to_present_value(self):
        info = {
            "targetLowPrice": 250.0,
            "targetMedianPrice": 320.0,
            "targetHighPrice": 400.0,
            "numberOfAnalystOpinions": 40,
        }
        result = ValuationEngine._institutional_outlook(info, 0.10)
        self.assertAlmostEqual(result["base"], 320.0 / 1.10)
        self.assertEqual(result["coverage"], 40)

    def test_market_leader_weights_include_institutional_outlook(self):
        methods = [
            {"name": "Cash-flow value", "bear": 100.0, "base": 160.0, "bull": 250.0, "confidence": 0.9},
            {"name": "Trading range", "bear": 180.0, "base": 260.0, "bull": 340.0, "confidence": 0.8},
            {"name": "Institutional outlook", "bear": 200.0, "base": 295.0, "bull": 370.0, "confidence": 0.7},
        ]
        usable, _ = ValuationEngine._prepare_methods(methods, "STANDARD", 325.0, market_leader=True)
        weights = {method["name"]: method["weight"] for method in usable}
        self.assertGreater(weights["Institutional outlook"], 0.50)


class MultiplesTests(unittest.TestCase):
    def test_currency_mismatch_disables_enterprise_value_conversions(self):
        info = {
            "currency": "USD",
            "financialCurrency": "TWD",
            "sharesOutstanding": 10.0,
            "forwardEps": 5.0,
            "totalRevenue": 1000.0,
        }
        inputs = MultiplesModel._target_inputs(info, pd.DataFrame())
        self.assertTrue(inputs["currency_mismatch"])
        self.assertTrue(np.isnan(MultiplesModel._implied_price("ev_revenue", 5.0, inputs)))
        self.assertEqual(MultiplesModel._implied_price("pe_forward", 20.0, inputs), 100.0)


class WACCTests(unittest.TestCase):
    def test_wacc_uses_reported_debt_cost_and_bounds(self):
        info = {"beta": 1.1, "marketCap": 100.0}
        result = WACCCalculator(0.04, 0.055).calculate(info, sample_financials())
        self.assertGreaterEqual(result["wacc"], 0.06)
        self.assertLessEqual(result["wacc"], 0.20)
        self.assertGreater(result["cost_of_debt"], 0.0)
        self.assertAlmostEqual(result["weight_equity"] + result["weight_debt"], 1.0)


if __name__ == "__main__":
    unittest.main()
