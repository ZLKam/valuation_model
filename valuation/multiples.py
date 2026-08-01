"""Robust comparable-company valuation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from valuation.assumptions import classify_company, is_market_leader

logger = logging.getLogger(__name__)


METRICS = {
    "pe_trailing": ("trailingPE", "Trailing P/E", 2.0, 120.0),
    "pe_forward": ("forwardPE", "Forward P/E", 2.0, 100.0),
    "ps_ratio": ("priceToSalesTrailing12Months", "Price / Sales", 0.1, 40.0),
    "ev_ebitda": ("enterpriseToEbitda", "EV / EBITDA", 2.0, 60.0),
    "ev_revenue": ("enterpriseToRevenue", "EV / Revenue", 0.1, 40.0),
    "pb_ratio": ("priceToBook", "Price / Book", 0.1, 30.0),
}


PROFILE_WEIGHTS = {
    "FINANCIAL": {"pe_forward": 0.35, "pe_trailing": 0.20, "pb_ratio": 0.45},
    "HIGH_GROWTH": {"pe_forward": 0.40, "ev_revenue": 0.35, "ev_ebitda": 0.25},
    "CYCLICAL": {"ev_ebitda": 0.45, "pe_trailing": 0.25, "pb_ratio": 0.15, "ev_revenue": 0.15},
    "MATURE": {"pe_forward": 0.35, "pe_trailing": 0.25, "ev_ebitda": 0.25, "pb_ratio": 0.15},
    "STANDARD": {"pe_forward": 0.35, "ev_ebitda": 0.30, "pe_trailing": 0.20, "ev_revenue": 0.15},
}


def _safe(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


class MultiplesModel:
    def calculate(
        self,
        target_ticker_symbol: str,
        peer_tickers: list,
        target_info: dict,
        target_financials: pd.DataFrame,
    ) -> dict:
        symbol = target_ticker_symbol.strip().upper()
        profile = classify_company(target_info)
        inputs = self._target_inputs(target_info, target_financials)

        target_multiples = self._extract_multiples(target_info)
        peer_rows: list[dict] = []
        for peer_symbol in list(dict.fromkeys(peer_tickers or []))[:8]:
            peer_symbol = str(peer_symbol).strip().upper()
            if not peer_symbol or peer_symbol == symbol:
                continue
            try:
                peer_info = yf.Ticker(peer_symbol).info or {}
            except Exception as exc:
                logger.warning("Unable to retrieve peer %s: %s", peer_symbol, exc)
                continue
            row = {
                "symbol": peer_symbol,
                "name": peer_info.get("shortName") or peer_info.get("longName") or peer_symbol,
                "market_cap": _safe(peer_info.get("marketCap"), np.nan),
                "revenue_growth": _safe(peer_info.get("revenueGrowth"), np.nan),
                "profit_margin": _safe(peer_info.get("profitMargins"), np.nan),
                **self._extract_multiples(peer_info),
            }
            peer_rows.append(row)

        peer_frame = pd.DataFrame(peer_rows)
        # Size similarity prevents a mega-cap from being benchmarked primarily
        # against structurally different micro-caps (and vice versa).
        target_cap = _safe(target_info.get("marketCap"))
        if len(peer_frame) >= 5 and target_cap > 0:
            comparable = peer_frame[
                peer_frame["market_cap"].between(target_cap * 0.10, target_cap * 10.0, inclusive="both")
            ]
            if len(comparable) >= 3:
                peer_frame = comparable

        stats: dict[str, dict] = {}
        implied_prices: dict[str, float] = {}
        implied_low: dict[str, float] = {}
        implied_high: dict[str, float] = {}
        quality_adjustment = self._quality_adjustment(target_info, peer_frame)

        for metric, (_, _, floor, ceiling) in METRICS.items():
            values = pd.to_numeric(peer_frame.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            values = values[(values >= floor) & (values <= ceiling)]
            values = self._trim_outliers(values)
            if values.empty:
                summary = {"median": np.nan, "mean": np.nan, "min": np.nan, "max": np.nan, "q25": np.nan, "q75": np.nan, "count": 0}
            else:
                summary = {
                    "median": float(values.median()),
                    "mean": float(values.mean()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "count": int(len(values)),
                }
            stats[metric] = summary
            implied_prices[metric] = self._implied_price(metric, summary["median"], inputs, quality_adjustment)
            implied_low[metric] = self._implied_price(metric, summary["q25"], inputs, quality_adjustment)
            implied_high[metric] = self._implied_price(metric, summary["q75"], inputs, quality_adjustment)

        weights = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["STANDARD"])
        aggregate = {
            "bear": self._weighted_value(implied_low, weights),
            "base": self._weighted_value(implied_prices, weights),
            "bull": self._weighted_value(implied_high, weights),
        }
        used_metrics = [metric for metric in weights if np.isfinite(implied_prices.get(metric, np.nan)) and implied_prices[metric] > 0]
        peer_counts = [stats[metric]["count"] for metric in used_metrics]
        dispersion = self._dispersion([implied_prices[metric] for metric in used_metrics])
        metric_coverage = min(len(used_metrics) / max(min(len(weights), 3), 1), 1.0)
        aggregate.update({
            "confidence": float(np.clip(
                (max(peer_counts, default=0) / 5.0)
                * (1.0 - min(dispersion, 0.75))
                * (0.60 + 0.40 * metric_coverage),
                0.0,
                1.0,
            )),
            "metrics_used": used_metrics,
            "dispersion": dispersion,
            "quality_adjustment": quality_adjustment,
        })

        target_row = {
            "symbol": symbol,
            "name": target_info.get("shortName") or target_info.get("longName") or symbol,
            "market_cap": target_cap or np.nan,
            **target_multiples,
        }
        display_columns = ["symbol", "name", "market_cap", *METRICS.keys()]
        display = pd.DataFrame([target_row, *peer_frame.to_dict(orient="records")])
        display = display.reindex(columns=display_columns)

        return {
            "peers_table": display.to_dict(orient="records"),
            "target_multiples": target_multiples,
            "stats": stats,
            "implied_prices": implied_prices,
            "implied_prices_low": implied_low,
            "implied_prices_high": implied_high,
            "aggregate": aggregate,
            "metrics_def": {key: {"info_key": values[0], "label": values[1]} for key, values in METRICS.items()},
            "profile": profile,
        }

    @staticmethod
    def _extract_multiples(info: dict) -> dict:
        result = {}
        for metric, (info_key, _, floor, ceiling) in METRICS.items():
            value = _safe(info.get(info_key), np.nan)
            result[metric] = value if np.isfinite(value) and floor <= value <= ceiling else np.nan
        return result

    @staticmethod
    def _target_inputs(info: dict, financials: pd.DataFrame) -> dict:
        shares = _safe(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))
        values = {
            "shares": shares,
            "currency_mismatch": bool(
                info.get("currency")
                and info.get("financialCurrency")
                and str(info.get("currency")).upper() != str(info.get("financialCurrency")).upper()
            ),
            "trailing_eps": _safe(info.get("trailingEps"), np.nan),
            "forward_eps": _safe(info.get("forwardEps"), np.nan),
            "book_value_per_share": _safe(info.get("bookValue"), np.nan),
            "revenue": _safe(info.get("totalRevenue")),
            "ebitda": _safe(info.get("ebitda")),
            "cash": _safe(info.get("totalCash")),
            "debt": _safe(info.get("totalDebt")),
        }
        if financials is not None and not financials.empty:
            latest = financials.iloc[-1]
            values["revenue"] = _safe(latest.get("revenue"), values["revenue"])
            values["ebitda"] = _safe(latest.get("ebitda"), values["ebitda"])
            values["cash"] = _safe(latest.get("total_cash"), values["cash"])
            values["debt"] = _safe(latest.get("total_debt"), values["debt"])
            if not np.isfinite(values["trailing_eps"]) and shares > 0:
                values["trailing_eps"] = _safe(latest.get("net_income")) / shares
            if not np.isfinite(values["book_value_per_share"]) and shares > 0:
                values["book_value_per_share"] = _safe(latest.get("equity")) / shares
        return values

    @staticmethod
    def _trim_outliers(values: pd.Series) -> pd.Series:
        if len(values) < 5:
            return values
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            return values
        trimmed = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
        return trimmed if len(trimmed) >= 3 else values

    @staticmethod
    def _implied_price(metric: str, multiple: float, inputs: dict, quality_adjustment: float = 1.0) -> float:
        if not np.isfinite(multiple) or multiple <= 0 or inputs["shares"] <= 0:
            return np.nan
        # Yahoo's per-share earnings for ADRs are generally quote-currency
        # figures, while statement totals and book value may remain in the
        # reporting currency.  Use only earnings multiples when they differ.
        if inputs.get("currency_mismatch") and metric not in {"pe_trailing", "pe_forward"}:
            return np.nan
        multiple *= quality_adjustment
        shares = inputs["shares"]
        if metric == "pe_trailing" and inputs["trailing_eps"] > 0:
            value = multiple * inputs["trailing_eps"]
        elif metric == "pe_forward" and inputs["forward_eps"] > 0:
            value = multiple * inputs["forward_eps"]
        elif metric == "ps_ratio" and inputs["revenue"] > 0:
            value = multiple * inputs["revenue"] / shares
        elif metric == "ev_revenue" and inputs["revenue"] > 0:
            value = (multiple * inputs["revenue"] + inputs["cash"] - inputs["debt"]) / shares
        elif metric == "ev_ebitda" and inputs["ebitda"] > 0:
            value = (multiple * inputs["ebitda"] + inputs["cash"] - inputs["debt"]) / shares
        elif metric == "pb_ratio" and inputs["book_value_per_share"] > 0:
            value = multiple * inputs["book_value_per_share"]
        else:
            return np.nan
        return max(float(value), 0.0)

    @staticmethod
    def _quality_adjustment(target_info: dict, peer_frame: pd.DataFrame) -> float:
        """Adjust peer multiples for material growth and margin differences.

        The adjustment is deliberately capped: it recognizes superior economics
        without allowing a theme or narrative to overwhelm observable peers.
        """
        if peer_frame.empty:
            return 1.0
        peer_growth = pd.to_numeric(peer_frame.get("revenue_growth"), errors="coerce").dropna()
        peer_margin = pd.to_numeric(peer_frame.get("profit_margin"), errors="coerce").dropna()
        target_growth = _safe(target_info.get("revenueGrowth"), np.nan)
        target_margin = _safe(target_info.get("profitMargins"), np.nan)
        adjustment = 1.0
        if np.isfinite(target_growth) and not peer_growth.empty:
            adjustment += 1.50 * (target_growth - float(peer_growth.median()))
        if np.isfinite(target_margin) and not peer_margin.empty:
            adjustment += 1.00 * (target_margin - float(peer_margin.median()))
        # Dominant platforms tend to retain a durable scarcity/ecosystem
        # premium that a simple growth-and-margin comparison misses.  We still
        # allow a discount, but not the full commodity-company discount.
        floor = 0.95 if is_market_leader(target_info) else 0.85
        return float(np.clip(adjustment, floor, 1.35))

    @staticmethod
    def _weighted_value(values: dict[str, float], weights: dict[str, float]) -> float:
        valid = [(values.get(metric, np.nan), weight) for metric, weight in weights.items()]
        valid = [(value, weight) for value, weight in valid if np.isfinite(value) and value > 0]
        total_weight = sum(weight for _, weight in valid)
        if total_weight <= 0:
            return np.nan
        return float(sum(value * weight for value, weight in valid) / total_weight)

    @staticmethod
    def _dispersion(values: list[float]) -> float:
        clean = np.asarray([value for value in values if np.isfinite(value) and value > 0], dtype=float)
        if len(clean) < 2:
            return 0.60 if len(clean) == 1 else 1.0
        return float(np.std(clean) / max(np.mean(clean), 1e-9))
