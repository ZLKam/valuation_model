"""Application-facing valuation orchestration.

All model selection and automatic assumptions live here.  The UI consumes one
stable result object and does not ask users to tune finance-model internals.
"""

from __future__ import annotations

import datetime as dt
import importlib
import re
from typing import Any

import numpy as np
import pandas as pd

from valuation.assumptions import AssumptionEngine, PROFILE_LABELS, is_market_leader
from valuation.data_provider import DataProvider
import valuation.dcf as dcf_module
from valuation.ddm import DividendDiscountModel
from valuation.fundamentals import EarningsPowerModel, ResidualIncomeModel
from valuation.multiples import MultiplesModel
from valuation.wacc import WACCCalculator


MODEL_WEIGHTS = {
    "FINANCIAL": {"Residual income": 0.50, "Trading range": 0.35, "Dividend value": 0.15},
    "HIGH_GROWTH": {"Cash-flow value": 0.55, "Trading range": 0.45},
    "CYCLICAL": {"Cash-flow value": 0.45, "Earnings power": 0.20, "Trading range": 0.35},
    "MATURE": {"Cash-flow value": 0.50, "Earnings power": 0.15, "Trading range": 0.20, "Dividend value": 0.15},
    "STANDARD": {"Cash-flow value": 0.60, "Trading range": 0.40},
}

MARKET_LEADER_WEIGHTS = {
    "HIGH_GROWTH": {"Cash-flow value": 0.20, "Trading range": 0.25, "Institutional outlook": 0.55},
    "CYCLICAL": {
        "Cash-flow value": 0.15,
        "Earnings power": 0.10,
        "Trading range": 0.20,
        "Institutional outlook": 0.55,
    },
    "MATURE": {
        "Cash-flow value": 0.20,
        "Trading range": 0.20,
        "Dividend value": 0.05,
        "Institutional outlook": 0.55,
    },
    "STANDARD": {"Cash-flow value": 0.20, "Trading range": 0.25, "Institutional outlook": 0.55},
}


def _safe(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _valid_target(value: Any, current_price: float) -> bool:
    value = _safe(value, np.nan)
    return bool(np.isfinite(value) and value > 0 and value >= current_price * 0.03 and value <= current_price * 8.0)


def _build_dcf_model():
    """Return the current DCF implementation, repairing partial hot reloads."""
    module = dcf_module
    if getattr(module, "DCF_API_VERSION", 0) != 2:
        module = importlib.reload(module)
    return module.DCFModel()


class ValuationEngine:
    def __init__(self, data_provider: DataProvider | None = None):
        self.data_provider = data_provider or DataProvider()

    def resolve_symbol(self, query: str) -> tuple[str, str]:
        """Resolve either a ticker or a company-name query."""
        query = str(query or "").strip()
        if not query:
            raise ValueError("Enter a company name or ticker.")
        direct = query.upper().replace(".", "-")
        matches = self.data_provider.search_tickers(query)
        for match in matches:
            if str(match.get("symbol", "")).upper().replace(".", "-") == direct:
                return direct, str(match.get("name") or direct)
        if matches:
            match = matches[0]
            return str(match["symbol"]).upper(), str(match.get("name") or match["symbol"])
        if re.fullmatch(r"[A-Z0-9][A-Z0-9-]{0,9}", direct):
            # Search services occasionally throttle valid symbols; analysis is
            # still the authoritative validation step.
            return direct, direct
        raise ValueError(f"No listed US company matched '{query}'.")

    def analyze(self, ticker_symbol: str) -> dict:
        symbol = str(ticker_symbol).strip().upper().replace(".", "-")
        ticker = self.data_provider.get_stock_data(symbol)
        try:
            info = ticker.info or {}
        except Exception as exc:
            raise ValueError(f"Market data for {symbol} could not be retrieved.") from exc

        quote_type = str(info.get("quoteType", "EQUITY")).upper()
        if quote_type == "ETF":
            raise ValueError(
                "This is an ETF. A stock DCF is not appropriate for a fund; use holdings, NAV, fees, and tracking analysis instead."
            )
        if quote_type not in {"EQUITY", ""}:
            raise ValueError(f"{symbol} is not a supported listed company.")

        current_price = self._current_price(ticker, info)
        if current_price <= 0:
            raise ValueError(f"A reliable current price for {symbol} is unavailable.")

        financials = self.data_provider.extract_financials_history(ticker)
        if financials.empty:
            raise ValueError(f"Annual financial statements for {symbol} are unavailable.")

        shares = _safe(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))
        risk_free_rate = self.data_provider.get_risk_free_rate()
        market_risk_premium = self.data_provider.get_market_risk_premium()
        wacc = WACCCalculator(
            risk_free_rate=risk_free_rate,
            market_risk_premium=market_risk_premium,
        ).calculate(info, financials)
        assumptions = AssumptionEngine().derive(info, financials, risk_free_rate, wacc["wacc"])
        if assumptions.profile == "ETF":
            raise ValueError("Fund valuation is not supported by the company valuation engine.")

        peers = self.data_provider.get_peer_tickers(symbol, limit=7)
        market_leader = is_market_leader(info)
        methods: list[dict] = []
        model_details: dict[str, dict] = {}
        risks = list(assumptions.warnings) + list(wacc.get("warnings", []))
        quote_currency = str(info.get("currency") or "").upper()
        financial_currency = str(info.get("financialCurrency") or quote_currency).upper()
        currency_mismatch = bool(quote_currency and financial_currency and quote_currency != financial_currency)
        if currency_mismatch:
            risks.append(
                f"Financial statements are reported in {financial_currency} while the shares trade in "
                f"{quote_currency}; enterprise-value models were excluded to avoid a currency mismatch."
            )

        if assumptions.profile != "FINANCIAL" and not currency_mismatch:
            dcf_model = _build_dcf_model()
            dcf = dcf_model.calculate(
                financials,
                wacc,
                base_growth_rate=assumptions.initial_growth,
                base_terminal_growth=assumptions.terminal_growth,
                shares_outstanding=shares,
                current_price=current_price,
                target_fcf_margin=assumptions.target_fcf_margin,
                projection_years=assumptions.projection_years,
            )
            model_details["dcf"] = dcf
            if "error" not in dcf:
                terminal_share = dcf["base"].get("terminal_value_share", 0.0)
                dcf_confidence = float(np.clip(0.90 - max(terminal_share - 0.60, 0.0), 0.40, 0.90))
                if assumptions.current_fcf_margin <= 0:
                    dcf_confidence *= 0.70
                methods.append(self._method(
                    "Cash-flow value", dcf["bear"]["target_price"], dcf["base"]["target_price"],
                    dcf["bull"]["target_price"], dcf_confidence,
                    "Ten-year revenue and free-cash-flow margin fade",
                ))
                risks.extend(dcf.get("quality", {}).get("warnings", []))
                implied_growth = dcf_model.implied_growth_rate(
                    financials, wacc, assumptions.terminal_growth, shares, current_price,
                    assumptions.target_fcf_margin, assumptions.projection_years,
                )
            else:
                risks.append(dcf["error"])
                implied_growth = None
        else:
            implied_growth = None

        try:
            multiples = MultiplesModel().calculate(symbol, peers, info, financials)
        except Exception as exc:
            multiples = {"error": str(exc), "peers_table": [], "aggregate": {}}
        model_details["multiples"] = multiples
        relative = multiples.get("aggregate", {})
        relative_confidence = _safe(relative.get("confidence"))
        relative_dispersion = _safe(relative.get("dispersion"), 1.0)
        reliable_relative = relative_confidence >= 0.35 and relative_dispersion <= 0.75
        if reliable_relative and all(_valid_target(relative.get(case), current_price) for case in ("bear", "base", "bull")):
            methods.append(self._method(
                "Trading range", relative["bear"], relative["base"], relative["bull"],
                relative_confidence, "Robust peer medians, filtered for outliers and company size",
            ))
        else:
            risks.append("Comparable-company signals were too sparse or dispersed for a reliable trading range.")

        if assumptions.profile == "FINANCIAL" and not currency_mismatch:
            residual = ResidualIncomeModel().calculate(
                financials, info, wacc["cost_of_equity"], assumptions.terminal_growth,
            )
            model_details["residual_income"] = residual
            if "error" not in residual:
                methods.append(self._method(
                    "Residual income", residual["bear"], residual["base"], residual["bull"], 0.82,
                    "Book value plus returns earned above the shareholder hurdle rate",
                ))
            else:
                risks.append(residual["error"])
        elif assumptions.profile != "FINANCIAL" and not currency_mismatch:
            earnings_power = EarningsPowerModel().calculate(financials, shares, wacc["cost_of_equity"])
            model_details["earnings_power"] = earnings_power
            if "error" not in earnings_power:
                methods.append(self._method(
                    "Earnings power", earnings_power["bear"], earnings_power["base"],
                    earnings_power["bull"], min(0.50 + earnings_power["years"] * 0.07, 0.80),
                    "Median normalized earnings capitalized without high-growth assumptions",
                ))

        institutional = self._institutional_outlook(info, wacc["cost_of_equity"])
        model_details["institutional_outlook"] = institutional
        if market_leader and "error" not in institutional:
            methods.append(self._method(
                "Institutional outlook",
                institutional["bear"],
                institutional["base"],
                institutional["bull"],
                institutional["confidence"],
                "Current one-year analyst target distribution, discounted back to present value",
            ))
            risks.append(
                "Institutional targets capture the current market regime but can move with sentiment and are not independent intrinsic value."
            )

        dividend_yield = _safe(info.get("dividendYield"))
        if dividend_yield >= 0.012 or assumptions.profile in {"MATURE", "FINANCIAL"}:
            ddm = DividendDiscountModel().calculate(
                info, wacc["cost_of_equity"], assumptions.initial_growth, assumptions.terminal_growth,
            )
            model_details["ddm"] = ddm
            if all(_valid_target(ddm.get(case), current_price) for case in ("bear", "base", "bull")):
                methods.append(self._method(
                    "Dividend value", ddm["bear"], ddm["base"], ddm["bull"], 0.68,
                    "Seven-year dividend fade into sustainable long-run growth",
                ))

        methods, ensemble_risks = self._prepare_methods(
            methods, assumptions.profile, current_price, market_leader=market_leader,
        )
        risks.extend(ensemble_risks)
        if not methods:
            raise ValueError(
                "The available data produced only weak or conflicting signals, so no fair-value estimate "
                "is shown rather than implying false precision."
            )

        valuation = self._combine(methods, current_price, assumptions.data_years, relative)
        drivers = self._drivers(financials, assumptions, info, wacc, valuation, market_leader)
        risks = self._risks(risks, valuation, wacc, financials, relative)

        try:
            price_history = ticker.history(period="1y", auto_adjust=True)
        except Exception:
            price_history = pd.DataFrame()

        return {
            "symbol": symbol,
            "company_name": info.get("longName") or info.get("shortName") or symbol,
            "current_price": current_price,
            "currency": info.get("currency") or "USD",
            "exchange": info.get("exchange") or info.get("fullExchangeName") or "",
            "sector": info.get("sector") or "Unclassified",
            "industry": info.get("industry") or "",
            "profile": assumptions.profile,
            "profile_label": PROFILE_LABELS.get(assumptions.profile, assumptions.profile.title()),
            "market_regime": {
                "leadership_premium": market_leader,
                "label": "AI and technology leadership premium" if market_leader else "Fundamental company regime",
                "analyst_coverage": int(_safe(info.get("numberOfAnalystOpinions"))),
            },
            "info": info,
            "financials": financials,
            "price_history": price_history,
            "risk_free_rate": risk_free_rate,
            "market_risk_premium": market_risk_premium,
            "wacc": wacc,
            "assumptions": assumptions.to_dict(),
            "methods": methods,
            "models": model_details,
            "valuation": valuation,
            "implied_growth": implied_growth,
            "peer_tickers": peers,
            "drivers": drivers,
            "risks": risks,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }

    @staticmethod
    def _current_price(ticker, info: dict) -> float:
        for key in ("currentPrice", "regularMarketPrice", "previousClose"):
            price = _safe(info.get(key))
            if price > 0:
                return price
        try:
            price = _safe(ticker.fast_info.get("last_price"))
            if price > 0:
                return price
        except Exception:
            pass
        try:
            history = ticker.history(period="5d")
            return _safe(history["Close"].dropna().iloc[-1]) if not history.empty else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _method(name: str, bear: float, base: float, bull: float, confidence: float, note: str) -> dict:
        ordered = sorted([float(bear), float(base), float(bull)])
        return {
            "name": name,
            "bear": ordered[0],
            "base": ordered[1],
            "bull": ordered[2],
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "note": note,
        }

    @staticmethod
    def _institutional_outlook(info: dict, cost_of_equity: float) -> dict:
        """Convert a one-year analyst target distribution into present value."""
        coverage = int(_safe(info.get("numberOfAnalystOpinions")))
        low = _safe(info.get("targetLowPrice"), np.nan)
        base = _safe(info.get("targetMedianPrice") or info.get("targetMeanPrice"), np.nan)
        high = _safe(info.get("targetHighPrice"), np.nan)
        if coverage < 5 or not all(np.isfinite(value) and value > 0 for value in (low, base, high)):
            return {"error": "A sufficiently broad analyst target distribution is unavailable."}
        low, base, high = sorted([low, base, high])
        if high / low > 8.0:
            return {"error": "Analyst targets are too dispersed to use as a market-regime reference."}
        discount_factor = 1.0 + float(np.clip(cost_of_equity, 0.06, 0.20))
        spread = (high - low) / max(base, 1e-9)
        confidence = float(np.clip(0.55 + 0.25 * min(coverage / 40.0, 1.0) - 0.15 * min(spread, 1.0), 0.40, 0.80))
        return {
            "bear": low / discount_factor,
            "base": base / discount_factor,
            "bull": high / discount_factor,
            "forecast_bear": low,
            "forecast_base": base,
            "forecast_bull": high,
            "coverage": coverage,
            "discount_factor": discount_factor,
            "confidence": confidence,
        }

    @staticmethod
    def _prepare_methods(
        methods: list[dict], profile: str, price: float, market_leader: bool = False,
    ) -> tuple[list[dict], list[str]]:
        risks: list[str] = []
        usable = [method for method in methods if _valid_target(method["base"], price)]
        configured = (
            MARKET_LEADER_WEIGHTS.get(profile, MODEL_WEIGHTS.get(profile, MODEL_WEIGHTS["STANDARD"]))
            if market_leader
            else MODEL_WEIGHTS.get(profile, MODEL_WEIGHTS["STANDARD"])
        )
        # Remove intentionally unused methods before testing ensemble outliers;
        # otherwise a zero-weight method can incorrectly eject a valid model.
        usable = [method for method in usable if configured.get(method["name"], 0.0) > 0]
        if len(usable) >= 3 and not market_leader:
            median = float(np.median([method["base"] for method in usable]))
            filtered = [method for method in usable if 0.40 * median <= method["base"] <= 2.50 * median]
            if len(filtered) >= 2 and len(filtered) < len(usable):
                risks.append("One model was excluded as an extreme outlier to the other methods.")
                usable = filtered

        if len(usable) == 1 and usable[0]["confidence"] < 0.65:
            risks.append("Only one lower-confidence valuation method was available; treat its range as directional.")
        if len(usable) == 2 and not market_leader:
            bases = [method["base"] for method in usable]
            if max(bases) / max(min(bases), 1e-9) > 6.0:
                risks.append("The remaining valuation methods disagreed by more than sixfold, so no midpoint was produced.")
                return [], risks
        for method in usable:
            base_weight = configured.get(method["name"], 0.0)
            method["raw_weight"] = base_weight * (0.65 + 0.35 * method["confidence"])
        total = sum(method["raw_weight"] for method in usable)
        if total <= 0:
            for method in usable:
                method["raw_weight"] = 1.0
            total = float(len(usable))
        for method in usable:
            method["weight"] = method.pop("raw_weight") / total
        return usable, risks

    @staticmethod
    def _combine(methods: list[dict], current_price: float, data_years: int, relative: dict) -> dict:
        values = {
            case: sum(method[case] * method["weight"] for method in methods)
            for case in ("bear", "base", "bull")
        }
        base_values = np.asarray([method["base"] for method in methods], dtype=float)
        dispersion = float(np.std(base_values) / max(np.mean(base_values), 1e-9)) if len(base_values) > 1 else 0.45
        upside = values["base"] / current_price - 1.0

        if upside >= 0.25:
            verdict, tone = "Meaningful upside", "positive"
        elif upside >= 0.10:
            verdict, tone = "Modest upside", "positive"
        elif upside > -0.10:
            verdict, tone = "Near fair value", "neutral"
        elif upside > -0.25:
            verdict, tone = "Modest downside", "negative"
        else:
            verdict, tone = "Meaningful downside", "negative"

        history_score = min(data_years / 4.0, 1.0)
        method_score = min(len(methods) / 3.0, 1.0)
        agreement_score = 1.0 - min(dispersion / 0.60, 1.0)
        peer_score = _safe(relative.get("confidence"), 0.35)
        confidence_score = round(100 * (0.30 * history_score + 0.30 * method_score + 0.30 * agreement_score + 0.10 * peer_score))
        if len(methods) == 1:
            confidence_score = min(confidence_score, 49)
        elif len(methods) < 3:
            confidence_score = min(confidence_score, 74)
        if _safe(relative.get("dispersion"), 1.0) > 0.30:
            confidence_score = min(confidence_score, 69)
        if dispersion > 0.30:
            confidence_score = min(confidence_score, 69)
        elif dispersion > 0.20:
            confidence_score = min(confidence_score, 74)
        if confidence_score >= 75:
            confidence_label = "High"
        elif confidence_score >= 50:
            confidence_label = "Medium"
        else:
            confidence_label = "Low"

        return {
            **values,
            "upside": upside,
            "bear_return": values["bear"] / current_price - 1.0,
            "bull_return": values["bull"] / current_price - 1.0,
            "verdict": verdict,
            "tone": tone,
            "confidence_score": confidence_score,
            "confidence_label": confidence_label,
            "model_dispersion": dispersion,
            "method_count": len(methods),
        }

    @staticmethod
    def _drivers(
        financials: pd.DataFrame, assumptions, info: dict, wacc: dict, valuation: dict, market_leader: bool = False,
    ) -> list[str]:
        drivers = []
        growth = assumptions.historical_growth
        direction = "grew" if growth >= 0 else "contracted"
        latest_growth = _safe(info.get("revenueGrowth"), np.nan)
        latest_text = (
            f"; latest reported growth is {latest_growth * 100:.1f}%"
            if np.isfinite(latest_growth)
            else ""
        )
        drivers.append(
            f"Revenue {direction} at roughly {abs(growth) * 100:.1f}% a year across the available history"
            f"{latest_text}."
        )
        drivers.append(
            f"The model fades free-cash-flow margin from {assumptions.current_fcf_margin * 100:.1f}% "
            f"toward a normalized {assumptions.target_fcf_margin * 100:.1f}%."
        )
        if market_leader:
            coverage = int(_safe(info.get("numberOfAnalystOpinions")))
            target = _safe(info.get("targetMedianPrice") or info.get("targetMeanPrice"))
            if coverage and target:
                drivers.append(
                    f"The market-leadership layer includes {coverage} analyst views centered near "
                    f"{target:,.0f} per share over the next year."
                )
        net_cash = wacc.get("total_cash", 0.0) - wacc.get("total_debt", 0.0)
        market_cap = max(wacc.get("market_cap", 0.0), 1.0)
        balance_word = "net cash" if net_cash >= 0 else "net debt"
        drivers.append(f"The balance sheet carries {balance_word} equal to {abs(net_cash) / market_cap * 100:.1f}% of market value.")
        return drivers[:3]

    @staticmethod
    def _risks(risks: list[str], valuation: dict, wacc: dict, financials: pd.DataFrame, relative: dict) -> list[str]:
        if valuation.get("bull_return", 0.0) < -0.50:
            risks.append(
                "The market price is more than twice the model's upside case, implying substantial future "
                "growth or optionality that is not yet supported by reported cash flow."
            )
        if valuation["model_dispersion"] > 0.30:
            risks.append("Valuation methods disagree materially, so the range matters more than the midpoint.")
        if relative.get("confidence", 0.0) < 0.40:
            risks.append("The peer sample is limited or widely dispersed.")
        if wacc.get("weight_debt", 0.0) > 0.40:
            risks.append("Debt represents more than 40% of estimated invested capital.")
        unique = []
        for risk in risks:
            risk = str(risk).strip()
            if risk and risk not in unique:
                unique.append(risk)
        if not unique:
            unique.append("Market prices can diverge from fundamental value for long periods.")
        return unique[:4]
