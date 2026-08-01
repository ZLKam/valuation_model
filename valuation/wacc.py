"""Cost-of-capital estimation with practical data-quality guardrails."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


class WACCCalculator:
    def __init__(self, risk_free_rate: float = 0.0425, market_risk_premium: float = 0.045):
        self.risk_free_rate = float(np.clip(risk_free_rate, 0.0, 0.10))
        self.market_risk_premium = float(np.clip(market_risk_premium, 0.035, 0.085))

    def calculate(self, ticker_info: dict, financials_history: pd.DataFrame) -> dict:
        warnings: list[str] = []

        raw_beta = _safe(ticker_info.get("beta"), 1.0)
        if raw_beta <= 0:
            raw_beta = 1.0
            warnings.append("Beta was unavailable; a market beta of 1.0 was used.")
        beta = float(np.clip(raw_beta, 0.50, 2.50))

        market_cap = _safe(ticker_info.get("marketCap"))
        if market_cap <= 0:
            shares = _safe(ticker_info.get("sharesOutstanding") or ticker_info.get("impliedSharesOutstanding"))
            price = _safe(ticker_info.get("currentPrice") or ticker_info.get("regularMarketPrice") or ticker_info.get("previousClose"))
            market_cap = shares * price
        if market_cap <= 0:
            market_cap = 1e9
            warnings.append("Market capitalization was estimated from a conservative fallback.")

        # Smaller companies generally carry risks not captured by a single beta.
        if market_cap < 2e9:
            size_premium = 0.015
        elif market_cap < 10e9:
            size_premium = 0.0075
        else:
            size_premium = 0.0
        cost_of_equity = self.risk_free_rate + beta * self.market_risk_premium + size_premium

        latest_debt = _safe(ticker_info.get("totalDebt"))
        latest_cash = _safe(ticker_info.get("totalCash"))
        tax_rate = 0.21
        cost_of_debt = 0.0
        debt_source = "no debt"

        if financials_history is not None and not financials_history.empty:
            latest = financials_history.iloc[-1]
            latest_debt = _safe(latest.get("total_debt"), latest_debt)
            latest_cash = _safe(latest.get("total_cash"), latest_cash)

            if "pretax_income" in financials_history and "tax_provision" in financials_history:
                pretax = pd.to_numeric(financials_history["pretax_income"], errors="coerce")
                tax = pd.to_numeric(financials_history["tax_provision"], errors="coerce")
                effective = (tax / pretax).replace([np.inf, -np.inf], np.nan)
                effective = effective[(effective >= 0.05) & (effective <= 0.40)].dropna()
                if not effective.empty:
                    tax_rate = float(effective.tail(3).median())

            if latest_debt > 0 and "interest_expense" in financials_history and "total_debt" in financials_history:
                interest = pd.to_numeric(financials_history["interest_expense"], errors="coerce").abs()
                debts = pd.to_numeric(financials_history["total_debt"], errors="coerce")
                rates = (interest / debts).replace([np.inf, -np.inf], np.nan)
                rates = rates[(rates >= 0.005) & (rates <= 0.25)].dropna()
                if not rates.empty:
                    cost_of_debt = float(rates.tail(3).median())
                    debt_source = "reported interest expense"

        if latest_debt > 0 and cost_of_debt <= 0:
            leverage = latest_debt / max(market_cap, 1.0)
            credit_spread = 0.015 + min(leverage, 1.5) * 0.02
            cost_of_debt = self.risk_free_rate + credit_spread
            debt_source = "risk-free rate plus leverage-adjusted credit spread"
        if latest_debt > 0:
            cost_of_debt = float(np.clip(cost_of_debt, self.risk_free_rate + 0.0075, self.risk_free_rate + 0.08))

        total_capital = market_cap + max(latest_debt, 0.0)
        weight_equity = market_cap / total_capital if total_capital > 0 else 1.0
        weight_debt = max(latest_debt, 0.0) / total_capital if total_capital > 0 else 0.0
        after_tax_cost_of_debt = cost_of_debt * (1.0 - tax_rate)
        raw_wacc = weight_equity * cost_of_equity + weight_debt * after_tax_cost_of_debt
        wacc = float(np.clip(raw_wacc, 0.06, 0.20))
        if abs(wacc - raw_wacc) > 1e-9:
            warnings.append("The calculated discount rate was constrained to a prudent 6%-20% range.")

        return {
            "wacc": wacc,
            "wacc_raw": raw_wacc,
            "cost_of_equity": cost_of_equity,
            "cost_of_debt": cost_of_debt,
            "after_tax_cost_of_debt": after_tax_cost_of_debt,
            "weight_equity": weight_equity,
            "weight_debt": weight_debt,
            "tax_rate": tax_rate,
            "beta": beta,
            "raw_beta": raw_beta,
            "size_premium": size_premium,
            "market_cap": market_cap,
            "total_debt": latest_debt,
            "total_cash": latest_cash,
            "risk_free_rate": self.risk_free_rate,
            "market_risk_premium": self.market_risk_premium,
            "debt_cost_source": debt_source,
            "warnings": warnings,
        }
