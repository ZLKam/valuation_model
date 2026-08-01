"""Automatic, explainable inputs for the valuation engine.

The UI deliberately keeps these inputs out of the primary workflow.  They are
derived from the company's own history and exposed only in the methodology
view so a user can understand the result without having to build the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


PROFILE_LABELS = {
    "FINANCIAL": "Financial institution",
    "HIGH_GROWTH": "High-growth company",
    "CYCLICAL": "Cyclical company",
    "MATURE": "Mature cash generator",
    "STANDARD": "Established company",
    "ETF": "Exchange-traded fund",
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def classify_company(info: dict) -> str:
    """Classify a security so unsuitable models are not forced onto it."""
    if str(info.get("quoteType", "")).upper() == "ETF":
        return "ETF"

    sector = str(info.get("sector", "")).lower()
    industry = str(info.get("industry", "")).lower()
    dividend_yield = _number(info.get("dividendYield"))
    revenue_growth = _number(info.get("revenueGrowth"))
    earnings_growth = _number(info.get("earningsGrowth"))

    if "financial" in sector or any(word in industry for word in ("bank", "insurance", "credit")):
        return "FINANCIAL"
    # A rebound from a depressed base can make a cyclical business look like a
    # secular compounder.  Treat extreme reported growth as a cycle before the
    # normal high-growth rules are considered.
    if revenue_growth > 1.0 or earnings_growth > 3.0:
        return "CYCLICAL"
    if any(word in industry for word in ("semiconductor", "software", "internet", "biotechnology")) and revenue_growth >= 0.10:
        return "HIGH_GROWTH"
    if any(word in industry for word in ("oil", "gas", "mining", "metal", "steel", "chemical", "auto")):
        return "CYCLICAL"
    if dividend_yield >= 0.025 and revenue_growth <= 0.10:
        return "MATURE"
    return "STANDARD"


def is_market_leader(info: dict) -> bool:
    """Identify scaled technology platforms where market-regime value matters.

    This deliberately uses observable characteristics rather than a hard-coded
    ticker list, so the group can change as market leadership changes.
    """
    market_cap = _number(info.get("marketCap"))
    analyst_coverage = _number(info.get("numberOfAnalystOpinions"))
    sector = str(info.get("sector", "")).lower()
    industry = str(info.get("industry", "")).lower()
    relevant_sectors = {"technology", "communication services", "consumer cyclical"}
    theme_keywords = (
        "semiconductor", "software", "internet", "consumer electronics",
        "computer hardware", "communication equipment", "information technology",
        "auto manufacturer",
    )
    return bool(
        market_cap >= 250e9
        and analyst_coverage >= 10
        and sector in relevant_sectors
        and any(keyword in industry for keyword in theme_keywords)
    )


@dataclass(frozen=True)
class ValuationAssumptions:
    profile: str
    initial_growth: float
    terminal_growth: float
    historical_growth: float
    current_fcf_margin: float
    target_fcf_margin: float
    projection_years: int = 10
    data_years: int = 0
    sources: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def profile_label(self) -> str:
        return PROFILE_LABELS.get(self.profile, self.profile.title())

    def to_dict(self) -> dict:
        result = asdict(self)
        result["profile_label"] = self.profile_label
        result["warnings"] = list(self.warnings)
        return result


class AssumptionEngine:
    """Build restrained forward inputs from reported fundamentals."""

    _GROWTH_BOUNDS = {
        "HIGH_GROWTH": (-0.10, 0.35),
        "CYCLICAL": (-0.12, 0.18),
        "MATURE": (-0.08, 0.12),
        "FINANCIAL": (-0.08, 0.15),
        "STANDARD": (-0.10, 0.22),
        "ETF": (-0.10, 0.15),
    }

    def derive(
        self,
        info: dict,
        financials: pd.DataFrame,
        risk_free_rate: float,
        discount_rate: float,
    ) -> ValuationAssumptions:
        profile = classify_company(info)
        warnings: list[str] = []

        revenues = pd.Series(dtype=float)
        fcfs = pd.Series(dtype=float)
        if financials is not None and not financials.empty:
            if "revenue" in financials:
                revenues = pd.to_numeric(financials["revenue"], errors="coerce").dropna()
            if "fcf" in financials:
                fcfs = pd.to_numeric(financials["fcf"], errors="coerce")

        historical_growth = self._historical_growth(revenues)
        reported_growth = _number(info.get("revenueGrowth"), np.nan)
        earnings_growth = _number(info.get("earningsGrowth"), np.nan)

        candidates: list[tuple[float, float, str]] = []
        if np.isfinite(historical_growth):
            candidates.append((historical_growth, 0.55, "multi-year revenue trend"))
        if np.isfinite(reported_growth):
            candidates.append((reported_growth, 0.30, "latest reported revenue growth"))
        if np.isfinite(earnings_growth):
            # Earnings growth is useful directionally, but too noisy to dominate.
            candidates.append((earnings_growth, 0.15, "latest reported earnings growth"))

        if candidates:
            raw_growth = sum(value * weight for value, weight, _ in candidates) / sum(weight for _, weight, _ in candidates)
            growth_source = ", ".join(source for _, _, source in candidates)
        else:
            raw_growth = 0.04
            growth_source = "conservative fallback due to limited history"
            warnings.append("Forward growth has limited supporting history.")

        lower, upper = self._GROWTH_BOUNDS[profile]
        initial_growth = float(np.clip(raw_growth, lower, upper))

        current_margin, target_margin = self._fcf_margins(revenues, fcfs, profile)
        if target_margin <= 0 and profile not in {"FINANCIAL", "ETF"}:
            warnings.append("The company has not established sustainable positive free cash flow; DCF receives no weight.")
        if len(revenues) < 3:
            warnings.append("Fewer than three annual revenue observations reduce confidence.")

        # A perpetual growth rate should stay below both nominal economic growth
        # and the discount rate. This removes the common WACC-g singularity.
        macro_anchor = float(np.clip(0.020 + max(risk_free_rate - 0.03, 0.0) * 0.15, 0.018, 0.025))
        terminal_growth = min(macro_anchor, max(discount_rate - 0.035, 0.005))

        return ValuationAssumptions(
            profile=profile,
            initial_growth=initial_growth,
            terminal_growth=terminal_growth,
            historical_growth=historical_growth if np.isfinite(historical_growth) else initial_growth,
            current_fcf_margin=current_margin,
            target_fcf_margin=target_margin,
            data_years=int(len(revenues)),
            sources={
                "growth": growth_source,
                "margin": "company-reported free cash flow margins, weighted toward recent years",
                "terminal_growth": "long-run nominal growth guardrail",
            },
            warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _historical_growth(revenues: pd.Series) -> float:
        values = revenues[revenues > 0]
        if len(values) < 2:
            return np.nan
        yoy = values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        yoy = yoy[(yoy >= -0.60) & (yoy <= 1.50)]
        median_yoy = float(yoy.median()) if not yoy.empty else np.nan
        years = len(values) - 1
        cagr = float((values.iloc[-1] / values.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else np.nan
        if np.isfinite(cagr) and np.isfinite(median_yoy):
            return 0.65 * cagr + 0.35 * median_yoy
        return cagr if np.isfinite(cagr) else median_yoy

    @staticmethod
    def _fcf_margins(revenues: pd.Series, fcfs: pd.Series, profile: str) -> tuple[float, float]:
        if revenues.empty or fcfs.empty:
            return 0.0, 0.0
        aligned = pd.concat([revenues.rename("revenue"), fcfs.rename("fcf")], axis=1).dropna()
        aligned = aligned[aligned["revenue"] > 0]
        if aligned.empty:
            return 0.0, 0.0
        margins = (aligned["fcf"] / aligned["revenue"]).clip(-0.50, 0.60)
        current = float(margins.iloc[-1])
        recent = margins.tail(4)
        weights = np.arange(1, len(recent) + 1, dtype=float)
        weighted = float(np.average(recent, weights=weights))
        median = float(recent.median())
        normalized = 0.60 * weighted + 0.40 * median

        if profile == "CYCLICAL":
            target = median
        elif profile == "HIGH_GROWTH" and normalized > 0:
            target = max(normalized, current * 0.85)
        else:
            target = normalized
        return current, float(np.clip(target, -0.25, 0.45))
