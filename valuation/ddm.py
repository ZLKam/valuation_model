"""Dividend valuation used only when cash distributions are meaningful."""

from __future__ import annotations

import numpy as np
import pandas as pd


class DividendDiscountModel:
    def calculate(
        self,
        ticker_info: dict,
        cost_of_equity: float,
        growth_rate: float,
        terminal_growth_rate: float = 0.02,
    ) -> dict:
        dividend_rate = ticker_info.get("dividendRate")
        price = ticker_info.get("currentPrice") or ticker_info.get("regularMarketPrice") or ticker_info.get("previousClose")
        if dividend_rate is None or pd.isna(dividend_rate) or dividend_rate <= 0:
            dividend_yield = ticker_info.get("dividendYield")
            dividend_rate = float(dividend_yield * price) if dividend_yield and price else 0.0

        results = {
            "gordon_growth_price": np.nan,
            "two_stage_ddm_price": np.nan,
            "bear": np.nan,
            "base": np.nan,
            "bull": np.nan,
            "dividend_rate": float(dividend_rate or 0.0),
            "cost_of_equity": float(cost_of_equity),
        }
        if not dividend_rate or dividend_rate <= 0:
            return results

        required_return = float(np.clip(cost_of_equity, 0.06, 0.22))
        stable_growth = min(float(terminal_growth_rate), required_return - 0.025)
        initial_growth = float(np.clip(growth_rate, -0.10, 0.12))

        if required_return > stable_growth:
            results["gordon_growth_price"] = dividend_rate * (1.0 + stable_growth) / (required_return - stable_growth)

        def two_stage(growth_shift: float, rate_shift: float) -> float:
            rate = float(np.clip(required_return + rate_shift, 0.06, 0.24))
            terminal = min(stable_growth + growth_shift * 0.15, rate - 0.025)
            start_growth = float(np.clip(initial_growth + growth_shift, -0.15, 0.15))
            growth_path = np.linspace(start_growth, terminal, 7)
            dividend = float(dividend_rate)
            present_value = 0.0
            for year, growth in enumerate(growth_path, start=1):
                dividend *= 1.0 + float(growth)
                present_value += dividend / ((1.0 + rate) ** year)
            terminal_value = dividend * (1.0 + terminal) / (rate - terminal)
            return max(present_value + terminal_value / ((1.0 + rate) ** len(growth_path)), 0.0)

        results["bear"] = two_stage(-0.02, 0.0125)
        results["base"] = two_stage(0.00, 0.0000)
        results["bull"] = two_stage(0.02, -0.0075)
        results["two_stage_ddm_price"] = results["base"]
        return results
