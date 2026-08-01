import json
import os
import tempfile
import unittest
from unittest.mock import patch

from valuation import positions


class PositionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.portfolio_file = os.path.join(self.temp_dir.name, "portfolio.json")
        self.file_patch = patch.object(positions, "PORTFOLIO_FILE", self.portfolio_file)
        self.file_patch.start()

    def tearDown(self):
        self.file_patch.stop()
        self.temp_dir.cleanup()

    def test_legacy_margin_schema_migrates(self):
        with open(self.portfolio_file, "w", encoding="utf-8") as handle:
            json.dump({"total_margin": 250000.0, "positions": []}, handle)
        portfolio = positions.load_portfolio()
        self.assertEqual(portfolio["account_value"], 250000.0)
        self.assertNotIn("total_margin", portfolio)
        self.assertEqual(portfolio["risk_per_trade"], 0.01)

    def test_position_size_respects_risk_allocation_and_cash(self):
        result = positions.PositionManager.calculate_position_size(
            entry_price=100.0,
            stop_price=90.0,
            account_value=100000.0,
            risk_per_trade=0.01,
            max_position_pct=0.15,
            available_cash=100000.0,
        )
        self.assertEqual(result["suggested_shares"], 100)
        self.assertEqual(result["max_loss_at_stop"], 1000.0)
        self.assertEqual(result["limiting_factor"], "risk budget")

    def test_add_update_and_diagnostics(self):
        positions.PositionManager.update_settings(100000.0, 0.01, 0.15, 0.35)
        positions.PositionManager.add_or_update_position(
            "aapl", 50, 100.0, "Core", 92.0, sector="Technology",
        )
        positions.PositionManager.add_or_update_position(
            "AAPL", 25, 120.0, "Core", 95.0, sector="Technology",
        )
        portfolio = positions.load_portfolio()
        self.assertEqual(len(portfolio["positions"]), 1)
        self.assertEqual(portfolio["positions"][0]["shares"], 75.0)
        self.assertAlmostEqual(portfolio["positions"][0]["avg_price"], 106.6666667, places=5)

        diagnostics = positions.PositionManager.get_portfolio_diagnostics(portfolio, {"AAPL": 130.0})
        self.assertEqual(diagnostics["total_market_value"], 9750.0)
        self.assertGreater(diagnostics["unrealized_pnl"], 0.0)
        self.assertGreater(diagnostics["portfolio_heat"], 0.0)

    def test_invalid_stop_is_rejected(self):
        with self.assertRaises(ValueError):
            positions.PositionManager.calculate_position_size(100.0, 105.0, 100000.0, 0.01, 0.15)

    def test_session_portfolio_update_is_isolated_and_not_written(self):
        original = positions.normalize_portfolio({
            "account_value": 50000.0,
            "risk_per_trade": 0.01,
            "max_position_pct": 0.15,
            "max_sector_pct": 0.35,
            "positions": [],
        })

        updated = positions.PositionManager.add_or_update_position(
            "MSFT",
            10,
            400.0,
            "Core",
            360.0,
            sector="Technology",
            portfolio=original,
            persist=False,
        )

        self.assertEqual(original["positions"], [])
        self.assertEqual(updated["positions"][0]["ticker"], "MSFT")
        self.assertFalse(os.path.exists(self.portfolio_file))

    def test_imported_portfolio_requires_a_json_object(self):
        with self.assertRaises(ValueError):
            positions.normalize_portfolio([])


if __name__ == "__main__":
    unittest.main()
