"""Fundamental models that complement DCF and trading multiples."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _valid_number(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


class EarningsPowerModel:
    """Values normalized earnings without assuming perpetual high growth."""

    def calculate(self, financials: pd.DataFrame, shares: float, cost_of_equity: float) -> dict:
        if financials is None or financials.empty or not shares or shares <= 0:
            return {"error": "Earnings power requires financial history and diluted shares."}

        earnings = pd.to_numeric(financials.get("net_income"), errors="coerce").dropna().tail(5)
        if earnings.empty:
            return {"error": "Reported earnings history is unavailable."}

        # Median protects a normalized valuation from one-off boom/bust years.
        normalized_income = float(earnings.median())
        if normalized_income <= 0:
            return {"error": "Normalized earnings are not positive."}

        normalized_eps = normalized_income / float(shares)

        def scenario(earnings_factor: float, rate_shift: float) -> float:
            required_return = float(np.clip(cost_of_equity + rate_shift, 0.065, 0.22))
            return max(normalized_eps * earnings_factor / required_return, 0.0)

        return {
            "bear": scenario(0.82, 0.015),
            "base": scenario(1.00, 0.000),
            "bull": scenario(1.12, -0.0075),
            "normalized_income": normalized_income,
            "normalized_eps": normalized_eps,
            "years": int(len(earnings)),
        }


class ResidualIncomeModel:
    """Equity valuation for banks and insurers where enterprise DCF is weak."""

    def calculate(
        self,
        financials: pd.DataFrame,
        ticker_info: dict,
        cost_of_equity: float,
        terminal_growth: float,
    ) -> dict:
        shares = ticker_info.get("sharesOutstanding") or ticker_info.get("impliedSharesOutstanding")
        if not shares or shares <= 0 or financials is None or financials.empty:
            return {"error": "Residual income requires book value history and diluted shares."}

        frame = financials.copy()
        if "equity" not in frame or "net_income" not in frame:
            return {"error": "Book value or net income history is unavailable."}
        frame = frame[["equity", "net_income"]].apply(pd.to_numeric, errors="coerce").dropna()
        frame = frame[frame["equity"] > 0].tail(5)
        if frame.empty:
            return {"error": "Positive book value history is unavailable."}

        latest_book = float(frame["equity"].iloc[-1])
        book_per_share = latest_book / float(shares)
        roes = (frame["net_income"] / frame["equity"]).replace([np.inf, -np.inf], np.nan).dropna()
        normalized_roe = float(roes.median()) if not roes.empty else np.nan
        if not _valid_number(normalized_roe):
            return {"error": "A sustainable return on equity could not be estimated."}
        normalized_roe = float(np.clip(normalized_roe, -0.10, 0.35))

        dividend_rate = ticker_info.get("dividendRate") or 0.0
        trailing_eps = ticker_info.get("trailingEps") or 0.0
        payout = float(dividend_rate / trailing_eps) if trailing_eps and trailing_eps > 0 else 0.35
        retention = float(np.clip(1.0 - payout, 0.15, 0.75))

        def run(roe_shift: float, rate_shift: float) -> float:
            required = float(np.clip(cost_of_equity + rate_shift, 0.065, 0.20))
            start_roe = float(np.clip(normalized_roe + roe_shift, -0.10, 0.38))
            book = book_per_share
            pv_residual = 0.0
            last_residual = 0.0
            horizon = 8
            for year in range(1, horizon + 1):
                # Competitive returns fade toward the shareholder hurdle rate.
                fade = year / (horizon + 2.0)
                roe = start_roe + (required - start_roe) * fade
                earnings = book * roe
                residual = earnings - required * book
                pv_residual += residual / ((1.0 + required) ** year)
                last_residual = residual
                book += max(earnings, 0.0) * retention

            stable_growth = min(terminal_growth, required - 0.03)
            terminal = last_residual * (1.0 + stable_growth) / (required - stable_growth)
            value = book_per_share + pv_residual + terminal / ((1.0 + required) ** horizon)
            return max(float(value), 0.0)

        return {
            "bear": run(-0.025, 0.0125),
            "base": run(0.000, 0.0000),
            "bull": run(0.020, -0.0075),
            "book_value_per_share": book_per_share,
            "normalized_roe": normalized_roe,
            "retention_rate": retention,
        }
