"""Multi-stage discounted cash-flow valuation.

This model projects revenue and cash-flow margins separately, fades growth to a
stable rate, and refuses to manufacture a positive terminal value for a company
that has not demonstrated sustainable cash generation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Used by the orchestration layer to detect a stale module after Streamlit
# partially hot-reloads a changed model contract.
DCF_API_VERSION = 2


def _safe(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


class DCFModel:
    def calculate(
        self,
        financials_history: pd.DataFrame,
        wacc_details: dict,
        base_growth_rate: float,
        base_terminal_growth: float = 0.02,
        shares_outstanding: float | None = None,
        current_price: float | None = None,
        target_fcf_margin: float | None = None,
        projection_years: int = 10,
        include_sensitivity: bool = True,
    ) -> dict:
        if financials_history is None or financials_history.empty:
            return {"error": "No financial history is available for a cash-flow valuation."}
        if not shares_outstanding or _safe(shares_outstanding) <= 0:
            return {"error": "Diluted shares outstanding are unavailable."}
        if "revenue" not in financials_history or "fcf" not in financials_history:
            return {"error": "Revenue or free cash flow history is unavailable."}

        history = financials_history.copy()
        history["revenue"] = pd.to_numeric(history["revenue"], errors="coerce")
        history["fcf"] = pd.to_numeric(history["fcf"], errors="coerce")
        valid = history[(history["revenue"] > 0) & history["fcf"].notna()].copy()
        if valid.empty:
            return {"error": "There is no usable revenue and free cash flow history."}

        latest = history.iloc[-1]
        latest_revenue = _safe(latest.get("revenue"))
        latest_fcf = _safe(latest.get("fcf"))
        if latest_revenue <= 0:
            latest_revenue = _safe(valid["revenue"].iloc[-1])
        if latest_revenue <= 0:
            return {"error": "Latest annual revenue is unavailable."}

        margins = (valid["fcf"] / valid["revenue"]).replace([np.inf, -np.inf], np.nan).dropna().clip(-0.50, 0.60)
        current_margin = _safe(latest_fcf / latest_revenue)
        normalized_margin = _safe(margins.tail(4).median())
        if target_fcf_margin is None:
            target_fcf_margin = 0.60 * _safe(margins.tail(4).mean()) + 0.40 * normalized_margin
        target_fcf_margin = float(np.clip(_safe(target_fcf_margin), -0.25, 0.45))

        # No arbitrary 5% FCF proxy: a non-positive steady-state margin cannot
        # support a conventional Gordon-growth terminal value.
        if target_fcf_margin <= 0:
            return {
                "error": "Sustainable positive free cash flow has not been established.",
                "quality": {
                    "current_fcf_margin": current_margin,
                    "normalized_fcf_margin": normalized_margin,
                    "reason": "DCF excluded instead of assuming an unsupported positive margin.",
                },
            }

        latest_cash = _safe(latest.get("total_cash"), _safe(wacc_details.get("total_cash")))
        latest_debt = _safe(latest.get("total_debt"), _safe(wacc_details.get("total_debt")))
        wacc = float(np.clip(_safe(wacc_details.get("wacc"), 0.10), 0.055, 0.22))
        terminal_growth = min(float(base_terminal_growth), wacc - 0.025)
        projection_years = int(np.clip(projection_years, 5, 12))

        def run_scenario(
            initial_growth: float,
            stable_growth: float,
            discount_rate: float,
            steady_margin: float,
        ) -> dict:
            initial_growth = float(np.clip(initial_growth, -0.25, 0.40))
            discount_rate = float(np.clip(discount_rate, 0.055, 0.24))
            stable_growth = min(float(np.clip(stable_growth, -0.01, 0.035)), discount_rate - 0.025)
            steady_margin = float(np.clip(steady_margin, 0.005, 0.48))

            # Explicit revenue growth fades toward a mature operating rate one
            # percentage point above perpetual growth.
            exit_revenue_growth = max(stable_growth + 0.01, min(initial_growth, 0.055))
            growth_path = np.linspace(initial_growth, exit_revenue_growth, projection_years)
            margin_path = np.linspace(current_margin, steady_margin, projection_years)

            revenues: list[float] = []
            projected_fcfs: list[float] = []
            discount_factors: list[float] = []
            revenue = latest_revenue
            for year, (growth, margin) in enumerate(zip(growth_path, margin_path), start=1):
                revenue *= 1.0 + float(growth)
                fcf = revenue * float(margin)
                # Mid-year convention better reflects cash generated throughout the year.
                discount_factor = 1.0 / ((1.0 + discount_rate) ** (year - 0.5))
                revenues.append(revenue)
                projected_fcfs.append(fcf)
                discount_factors.append(discount_factor)

            pv_projected_fcfs = [fcf * factor for fcf, factor in zip(projected_fcfs, discount_factors)]
            terminal_fcf = projected_fcfs[-1] * (1.0 + stable_growth)
            terminal_value = terminal_fcf / (discount_rate - stable_growth)
            pv_terminal_value = terminal_value / ((1.0 + discount_rate) ** projection_years)
            enterprise_value = sum(pv_projected_fcfs) + pv_terminal_value
            equity_value = enterprise_value + latest_cash - latest_debt
            target_price = max(equity_value / float(shares_outstanding), 0.0)

            return {
                "growth_rate": initial_growth,
                "exit_revenue_growth": exit_revenue_growth,
                "terminal_growth": stable_growth,
                "discount_rate": discount_rate,
                "start_fcf_margin": current_margin,
                "target_fcf_margin": steady_margin,
                "projected_revenues": revenues,
                "projected_fcf_margins": list(map(float, margin_path)),
                "projected_fcfs": projected_fcfs,
                "discount_factors": discount_factors,
                "pv_projected_fcfs": pv_projected_fcfs,
                "sum_pv_fcfs": sum(pv_projected_fcfs),
                "terminal_value": terminal_value,
                "pv_terminal_value": pv_terminal_value,
                "terminal_value_share": pv_terminal_value / enterprise_value if enterprise_value > 0 else 0.0,
                "enterprise_value": enterprise_value,
                "equity_value": equity_value,
                "target_price": target_price,
            }

        base = run_scenario(base_growth_rate, terminal_growth, wacc, target_fcf_margin)
        growth_spread = max(0.025, abs(base_growth_rate) * 0.25)
        margin_spread = max(0.015, abs(target_fcf_margin) * 0.15)
        bear = run_scenario(
            base_growth_rate - growth_spread,
            terminal_growth - 0.004,
            wacc + 0.015,
            target_fcf_margin - margin_spread,
        )
        bull = run_scenario(
            base_growth_rate + growth_spread,
            terminal_growth + 0.003,
            wacc - 0.010,
            target_fcf_margin + margin_spread,
        )

        sensitivity_df = pd.DataFrame()
        if include_sensitivity:
            wacc_steps = [max(wacc + shift, 0.055) for shift in (-0.02, -0.01, 0.0, 0.01, 0.02)]
            terminal_steps = [max(terminal_growth + shift, 0.0) for shift in (-0.01, -0.005, 0.0, 0.005, 0.01)]
            data = {}
            for rate in wacc_steps:
                data[f"{rate * 100:.1f}%"] = [
                    run_scenario(base_growth_rate, min(growth, rate - 0.025), rate, target_fcf_margin)["target_price"]
                    for growth in terminal_steps
                ]
            sensitivity_df = pd.DataFrame(data, index=[f"{growth * 100:.1f}%" for growth in terminal_steps])
            sensitivity_df.index.name = "Terminal growth / WACC"

        warnings: list[str] = []
        if current_margin <= 0:
            warnings.append("The base case requires free cash flow to recover from a negative current margin.")
        if base["terminal_value_share"] > 0.75:
            warnings.append("More than 75% of enterprise value comes from the terminal value.")

        return {
            "base": base,
            "bear": bear,
            "bull": bull,
            "sensitivity": sensitivity_df.to_dict(),
            "sensitivity_columns": list(sensitivity_df.columns),
            "sensitivity_index": list(sensitivity_df.index),
            "quality": {
                "current_fcf_margin": current_margin,
                "normalized_fcf_margin": normalized_margin,
                "terminal_value_share": base["terminal_value_share"],
                "warnings": warnings,
            },
            "inputs": {
                "latest_rev": latest_revenue,
                "latest_fcf": latest_fcf,
                "latest_cash": latest_cash,
                "latest_debt": latest_debt,
                "shares_outstanding": float(shares_outstanding),
                "current_price": current_price,
                "projection_years": projection_years,
            },
        }

    def implied_growth_rate(
        self,
        financials_history: pd.DataFrame,
        wacc_details: dict,
        terminal_growth: float,
        shares_outstanding: float,
        current_price: float,
        target_fcf_margin: float,
        projection_years: int = 10,
    ) -> float | None:
        """Solve the initial growth rate embedded in the current share price."""
        low, high = -0.25, 0.50
        low_result = self.calculate(
            financials_history,
            wacc_details,
            base_growth_rate=low,
            base_terminal_growth=terminal_growth,
            shares_outstanding=shares_outstanding,
            current_price=current_price,
            target_fcf_margin=target_fcf_margin,
            projection_years=projection_years,
            include_sensitivity=False,
        )
        high_result = self.calculate(
            financials_history,
            wacc_details,
            base_growth_rate=high,
            base_terminal_growth=terminal_growth,
            shares_outstanding=shares_outstanding,
            current_price=current_price,
            target_fcf_margin=target_fcf_margin,
            projection_years=projection_years,
            include_sensitivity=False,
        )
        if "error" in low_result or "error" in high_result:
            return None
        low_value = low_result["base"]["target_price"]
        high_value = high_result["base"]["target_price"]
        if not (low_value <= current_price <= high_value):
            return None
        for _ in range(40):
            mid = (low + high) / 2.0
            result = self.calculate(
                financials_history,
                wacc_details,
                base_growth_rate=mid,
                base_terminal_growth=terminal_growth,
                shares_outstanding=shares_outstanding,
                current_price=current_price,
                target_fcf_margin=target_fcf_margin,
                projection_years=projection_years,
                include_sensitivity=False,
            )
            value = result["base"]["target_price"]
            if value < current_price:
                low = mid
            else:
                high = mid
        return float((low + high) / 2.0)
