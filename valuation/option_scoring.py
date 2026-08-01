"""Transparent scoring primitives for single-leg option strategies.

The first supported strategy is a cash-secured short put.  Market-data access
is deliberately kept outside this module so the scoring rules can be tested
with deterministic chains and later reused with a broker-grade quote source.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Mapping

from scipy.stats import norm


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "lowest_risk": {
        "min_dte": 30,
        "max_dte": 45,
        "target_dte": 35,
        "max_assignment_probability": 0.15,
        "min_open_interest": 250,
        "max_bid_ask_spread_pct": 0.15,
        "premium_yield_target": 0.10,
        "min_short_delta": 0.01,
        "target_short_delta": 0.08,
        "max_short_delta": 0.18,
    },
    "balanced": {
        "min_dte": 30,
        "max_dte": 45,
        "target_dte": 38,
        "max_assignment_probability": 0.28,
        "min_open_interest": 100,
        "max_bid_ask_spread_pct": 0.25,
        "premium_yield_target": 0.16,
        "min_short_delta": 0.10,
        "target_short_delta": 0.22,
        "max_short_delta": 0.35,
    },
    "income_focused": {
        "min_dte": 30,
        "max_dte": 45,
        "target_dte": 30,
        "max_assignment_probability": 0.50,
        "min_open_interest": 50,
        "max_bid_ask_spread_pct": 0.35,
        "premium_yield_target": 0.35,
        "min_short_delta": 0.25,
        "target_short_delta": 0.40,
        "max_short_delta": 0.50,
    },
}


PROFILE_ALIASES = {
    "safest": "lowest_risk",
    "low": "lowest_risk",
    "moderate": "balanced",
    "aggressive": "income_focused",
    "income": "income_focused",
}


SCORE_WEIGHTS: dict[str, dict[str, float]] = {
    "lowest_risk": {
        "risk_target_fit": 0.28,
        "breakeven_safety": 0.17,
        "liquidity": 0.20,
        "greek_risk": 0.12,
        "dte_theta": 0.10,
        "premium_efficiency": 0.07,
        "volatility_context": 0.06,
    },
    "balanced": {
        "risk_target_fit": 0.22,
        "breakeven_safety": 0.15,
        "liquidity": 0.18,
        "greek_risk": 0.10,
        "dte_theta": 0.12,
        "premium_efficiency": 0.13,
        "volatility_context": 0.10,
    },
    "income_focused": {
        "risk_target_fit": 0.18,
        "breakeven_safety": 0.06,
        "liquidity": 0.14,
        "greek_risk": 0.06,
        "dte_theta": 0.16,
        "premium_efficiency": 0.26,
        "volatility_context": 0.14,
    },
}


RISK_TARGET_SAFETY_BLEND = {
    "lowest_risk": 0.75,
    "balanced": 0.35,
    "income_focused": 0.10,
}


COMBO_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "downside_aware": {
        "min_call_delta": 0.15,
        "max_call_delta": 0.50,
        "min_call_open_interest": 250,
        "max_call_spread_pct": 0.15,
        "min_premium_utilization": 0.40,
    },
    "balanced": {
        "min_call_delta": 0.20,
        "max_call_delta": 0.55,
        "min_call_open_interest": 100,
        "max_call_spread_pct": 0.25,
        "min_premium_utilization": 0.50,
    },
    "upside_focused": {
        "min_call_delta": 0.25,
        "max_call_delta": 0.60,
        "min_call_open_interest": 50,
        "max_call_spread_pct": 0.35,
        "min_premium_utilization": 0.60,
    },
}


COMBO_PROFILE_ALIASES = {
    "lowest_risk": "downside_aware",
    "safest": "downside_aware",
    "moderate": "balanced",
    "aggressive": "upside_focused",
    "maximum_upside": "upside_focused",
}


COMBO_PUT_PROFILE_MAP = {
    "downside_aware": "lowest_risk",
    "balanced": "balanced",
    "upside_focused": "income_focused",
}


COMBO_SCORE_WEIGHTS: dict[str, dict[str, float]] = {
    "downside_aware": {
        "downside_safety": 0.32,
        "upside_participation": 0.23,
        "funding_efficiency": 0.15,
        "liquidity": 0.15,
        "net_greeks": 0.08,
        "iv_skew": 0.04,
        "dte_fit": 0.03,
    },
    "balanced": {
        "downside_safety": 0.24,
        "upside_participation": 0.32,
        "funding_efficiency": 0.15,
        "liquidity": 0.12,
        "net_greeks": 0.09,
        "iv_skew": 0.05,
        "dte_fit": 0.03,
    },
    "upside_focused": {
        "downside_safety": 0.18,
        "upside_participation": 0.40,
        "funding_efficiency": 0.14,
        "liquidity": 0.10,
        "net_greeks": 0.10,
        "iv_skew": 0.05,
        "dte_fit": 0.03,
    },
}


@dataclass(frozen=True)
class ShortPutPreferences:
    """User constraints and priorities for a cash-secured short put scan."""

    profile: str = "lowest_risk"
    min_dte: int = 30
    max_dte: int = 45
    target_dte: int = 35
    max_assignment_probability: float = 0.15
    min_open_interest: int = 250
    max_bid_ask_spread_pct: float = 0.15
    premium_yield_target: float = 0.10
    min_short_delta: float = 0.01
    target_short_delta: float = 0.08
    max_short_delta: float = 0.18
    min_bid: float = 0.05
    max_cash_secured: float | None = None
    limit: int = 5
    require_otm: bool = True

    def __post_init__(self) -> None:
        canonical = PROFILE_ALIASES.get(self.profile, self.profile)
        if canonical not in PROFILE_DEFAULTS:
            raise ValueError(f"Unsupported short-put profile: {self.profile}")
        object.__setattr__(self, "profile", canonical)
        if self.min_dte < 1 or self.max_dte < self.min_dte:
            raise ValueError("DTE window must be positive and ordered")
        if not self.min_dte <= self.target_dte <= self.max_dte:
            raise ValueError("Target DTE must fall inside the DTE window")
        if not 0.01 <= self.max_assignment_probability <= 0.95:
            raise ValueError("Maximum assignment probability must be between 1% and 95%")
        if self.min_open_interest < 0:
            raise ValueError("Minimum open interest cannot be negative")
        if not 0.01 <= self.max_bid_ask_spread_pct <= 2.0:
            raise ValueError("Maximum bid/ask spread must be between 1% and 200%")
        if not 0.0 <= self.min_short_delta < self.target_short_delta < self.max_short_delta <= 1.0:
            raise ValueError("Short-delta range and target must be positive and ordered")
        if self.min_bid < 0 or self.limit < 1:
            raise ValueError("Minimum bid and result limit must be positive")
        if self.max_cash_secured is not None and self.max_cash_secured <= 0:
            raise ValueError("Cash available must be positive when supplied")

    @classmethod
    def for_profile(cls, profile: str = "lowest_risk", **overrides: Any) -> "ShortPutPreferences":
        canonical = PROFILE_ALIASES.get(profile, profile)
        if canonical not in PROFILE_DEFAULTS:
            raise ValueError(f"Unsupported short-put profile: {profile}")
        values = {"profile": canonical, **PROFILE_DEFAULTS[canonical], **overrides}
        return cls(**values)

    def with_overrides(self, **overrides: Any) -> "ShortPutPreferences":
        return replace(self, **overrides)

    def with_dte_window(self, min_dte: int, max_dte: int) -> "ShortPutPreferences":
        """Apply a user DTE window while retaining the profile target when possible."""
        target_dte = min(max(self.target_dte, min_dte), max_dte)
        return replace(self, min_dte=min_dte, max_dte=max_dte, target_dte=target_dte)


@dataclass(frozen=True)
class BullishComboPreferences:
    """Selection rules for a short-put-funded long-call risk reversal."""

    profile: str = "balanced"
    min_call_delta: float = 0.20
    max_call_delta: float = 0.55
    min_call_open_interest: int = 100
    max_call_spread_pct: float = 0.25
    min_premium_utilization: float = 0.50
    max_extra_debit: float = 0.0
    limit: int = 5

    def __post_init__(self) -> None:
        canonical = COMBO_PROFILE_ALIASES.get(self.profile, self.profile)
        if canonical not in COMBO_PROFILE_DEFAULTS:
            raise ValueError(f"Unsupported bullish-combo profile: {self.profile}")
        object.__setattr__(self, "profile", canonical)
        if not 0.01 <= self.min_call_delta < self.max_call_delta <= 1.0:
            raise ValueError("Call delta range must be positive and ordered")
        if self.min_call_open_interest < 0:
            raise ValueError("Minimum call open interest cannot be negative")
        if not 0.01 <= self.max_call_spread_pct <= 2.0:
            raise ValueError("Maximum call bid/ask spread must be between 1% and 200%")
        if not 0.0 <= self.min_premium_utilization <= 1.0:
            raise ValueError("Minimum premium utilization must be between 0% and 100%")
        if self.max_extra_debit < 0:
            raise ValueError("Additional call budget cannot be negative")
        if self.limit < 1:
            raise ValueError("Result limit must be positive")

    @classmethod
    def for_profile(cls, profile: str = "balanced", **overrides: Any) -> "BullishComboPreferences":
        canonical = COMBO_PROFILE_ALIASES.get(profile, profile)
        if canonical not in COMBO_PROFILE_DEFAULTS:
            raise ValueError(f"Unsupported bullish-combo profile: {profile}")
        values = {"profile": canonical, **COMBO_PROFILE_DEFAULTS[canonical], **overrides}
        return cls(**values)

    def with_overrides(self, **overrides: Any) -> "BullishComboPreferences":
        return replace(self, **overrides)


def black_scholes_metrics(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: str = "put",
) -> dict[str, float]:
    """Return BSM delta/gamma/theta/vega and risk-neutral expiry ITM probability.

    Theta is per calendar day and vega is per one volatility percentage point.
    Values use the long-option sign convention; a short position has the
    opposite exposure.
    """

    if spot <= 0 or strike <= 0 or volatility <= 0:
        raise ValueError("Spot, strike, and volatility must be positive")
    if option_type not in {"put", "call"}:
        raise ValueError("option_type must be 'put' or 'call'")
    if time_years <= 0:
        is_itm = spot < strike if option_type == "put" else spot > strike
        if option_type == "put":
            delta = -1.0 if is_itm else 0.0
        else:
            delta = 1.0 if is_itm else 0.0
        return {
            "d1": 0.0,
            "d2": 0.0,
            "delta": delta,
            "gamma": 0.0,
            "theta_per_day": 0.0,
            "vega_per_vol_point": 0.0,
            "probability_itm": 1.0 if is_itm else 0.0,
        }

    sqrt_t = math.sqrt(time_years)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * time_years
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discount_r = math.exp(-rate * time_years)
    discount_q = math.exp(-dividend_yield * time_years)
    density = norm.pdf(d1)
    gamma = discount_q * density / (spot * volatility * sqrt_t)
    vega = spot * discount_q * density * sqrt_t / 100.0
    common_theta = -(spot * discount_q * density * volatility) / (2.0 * sqrt_t)

    if option_type == "put":
        delta = discount_q * (norm.cdf(d1) - 1.0)
        theta_year = (
            common_theta
            + rate * strike * discount_r * norm.cdf(-d2)
            - dividend_yield * spot * discount_q * norm.cdf(-d1)
        )
        probability_itm = norm.cdf(-d2)
    else:
        delta = discount_q * norm.cdf(d1)
        theta_year = (
            common_theta
            - rate * strike * discount_r * norm.cdf(d2)
            + dividend_yield * spot * discount_q * norm.cdf(d1)
        )
        probability_itm = norm.cdf(d2)

    return {
        "d1": float(d1),
        "d2": float(d2),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta_per_day": float(theta_year / 365.0),
        "vega_per_vol_point": float(vega),
        "probability_itm": float(probability_itm),
    }


def probability_below_price(
    spot: float,
    threshold: float,
    time_years: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Risk-neutral probability that the underlying finishes below a threshold."""

    return black_scholes_metrics(
        spot,
        threshold,
        time_years,
        rate,
        volatility,
        dividend_yield,
        option_type="put",
    )["probability_itm"]


def calculate_iv_rank(current_iv: float, history: Iterable[float]) -> float | None:
    """Calculate the standard range-based IV rank from decimal volatility values."""

    values = [_number(item, -1.0) for item in history]
    values = [item for item in values if item > 0]
    current = _number(current_iv, -1.0)
    if current <= 0 or len(values) < 2:
        return None
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return None
    return _clamp((current - low) / (high - low))


def calculate_iv_percentile(current_iv: float, history: Iterable[float]) -> float | None:
    """Return the share of historical observations at or below current IV."""

    values = [_number(item, -1.0) for item in history]
    values = [item for item in values if item > 0]
    current = _number(current_iv, -1.0)
    if current <= 0 or not values:
        return None
    return sum(item <= current for item in values) / len(values)


def _profile_weights(profile: str) -> Mapping[str, float]:
    return SCORE_WEIGHTS[PROFILE_ALIASES.get(profile, profile)]


def _score_contract(
    contract: Mapping[str, Any],
    spot: float,
    historical_volatility: float,
    preferences: ShortPutPreferences,
    rate: float,
    dividend_yield: float,
    iv_context: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    strike = _number(contract.get("strike"))
    bid = _number(contract.get("bid"))
    ask = _number(contract.get("ask"))
    dte = int(_number(contract.get("dte")))
    open_interest = int(_number(contract.get("openInterest")))
    volume = int(_number(contract.get("volume")))

    if strike <= 0 or bid < preferences.min_bid or ask <= 0 or ask < bid or dte <= 0:
        return None, "invalid or non-marketable quote"
    if not preferences.min_dte <= dte <= preferences.max_dte:
        return None, "outside DTE window"
    if preferences.require_otm and strike >= spot:
        return None, "not out of the money"

    midpoint = (bid + ask) / 2.0
    spread_pct = (ask - bid) / midpoint if midpoint > 0 else math.inf
    if spread_pct > preferences.max_bid_ask_spread_pct:
        return None, "bid/ask spread too wide"
    if open_interest < preferences.min_open_interest:
        return None, "open interest below minimum"

    cash_secured = strike * 100.0
    if preferences.max_cash_secured is not None and cash_secured > preferences.max_cash_secured:
        return None, "cash required exceeds limit"

    quoted_iv = _number(contract.get("impliedVolatility"), -1.0)
    used_iv = _number(contract.get("ivUsed"), -1.0)
    iv_is_fallback = quoted_iv < 0.02
    if used_iv < 0.02:
        used_iv = historical_volatility
        iv_is_fallback = True
    if used_iv <= 0:
        return None, "missing volatility"

    time_years = dte / 365.25
    greeks = black_scholes_metrics(
        spot, strike, time_years, rate, used_iv, dividend_yield, "put"
    )
    assignment_probability = greeks["probability_itm"]
    if assignment_probability > preferences.max_assignment_probability:
        return None, "assignment estimate above limit"
    short_delta = max(-greeks["delta"], 0.0)
    if not preferences.min_short_delta <= short_delta <= preferences.max_short_delta:
        return None, "short delta outside profile range"

    break_even = strike - bid
    loss_probability = probability_below_price(
        spot, break_even, time_years, rate, used_iv, dividend_yield
    )
    max_loss = max(break_even, 0.0) * 100.0
    annualized_return = (bid / strike) * (365.0 / dte)
    otm_cushion = (spot - strike) / spot
    breakeven_cushion = (spot - break_even) / spot
    expected_move_pct = used_iv * math.sqrt(time_years)

    assignment_safety_score = _clamp(
        1.0 - assignment_probability / preferences.max_assignment_probability
    )
    delta_target_width = max(
        (preferences.max_short_delta - preferences.min_short_delta) / 2.0,
        0.03,
    )
    delta_target_score = math.exp(
        -0.5 * ((short_delta - preferences.target_short_delta) / delta_target_width) ** 2
    )
    safety_blend = RISK_TARGET_SAFETY_BLEND[preferences.profile]
    risk_target_score = (
        safety_blend * assignment_safety_score
        + (1.0 - safety_blend) * delta_target_score
    )
    breakeven_score = _clamp(
        1.0 - loss_probability / preferences.max_assignment_probability
    )

    spread_score = _clamp(1.0 - spread_pct / preferences.max_bid_ask_spread_pct)
    oi_target = max(1_000, preferences.min_open_interest * 5)
    oi_score = _clamp(math.log1p(open_interest) / math.log1p(oi_target))
    volume_score = _clamp(math.log1p(volume) / math.log1p(250))
    liquidity_score = 0.60 * spread_score + 0.30 * oi_score + 0.10 * volume_score

    dte_width = max((preferences.max_dte - preferences.min_dte) / 2.0, 1.0)
    dte_score = math.exp(-0.5 * ((dte - preferences.target_dte) / dte_width) ** 2)
    seller_theta = max(-greeks["theta_per_day"], 0.0)
    annualized_theta_return = seller_theta * 365.0 / strike
    theta_score = _clamp(annualized_theta_return / 0.12)
    dte_theta_score = 0.45 * dte_score + 0.55 * theta_score

    gamma_delta_change_1pct = greeks["gamma"] * spot * 0.01
    gamma_score = 1.0 - _clamp(gamma_delta_change_1pct / 0.02)
    vega_five_point_move_pct = (greeks["vega_per_vol_point"] * 5.0) / max(bid, 0.01)
    vega_score = 1.0 - _clamp(vega_five_point_move_pct / 2.0)
    greek_risk_score = 0.60 * gamma_score + 0.40 * vega_score

    premium_score = _clamp(annualized_return / preferences.premium_yield_target)
    iv_to_hv = used_iv / historical_volatility if historical_volatility > 0 else 1.0
    iv_edge_score = _clamp((iv_to_hv - 0.80) / 0.80)
    iv_rank = iv_context.get("iv_rank")
    if iv_rank is None:
        volatility_score = iv_edge_score
    else:
        volatility_score = 0.65 * _clamp(_number(iv_rank)) + 0.35 * iv_edge_score

    components = {
        "risk_target_fit": risk_target_score * 100.0,
        "breakeven_safety": breakeven_score * 100.0,
        "liquidity": liquidity_score * 100.0,
        "greek_risk": greek_risk_score * 100.0,
        "dte_theta": dte_theta_score * 100.0,
        "premium_efficiency": premium_score * 100.0,
        "volatility_context": volatility_score * 100.0,
    }
    weights = _profile_weights(preferences.profile)
    raw_score = sum(components[name] * weight for name, weight in weights.items())
    data_quality_factor = 0.85 if iv_is_fallback else 1.0
    total_score = raw_score * data_quality_factor

    recommendation = {
        "contract_symbol": contract.get("contractSymbol", ""),
        "expiration": contract.get("expiration", ""),
        "dte": dte,
        "strike": strike,
        "spot": spot,
        "bid": bid,
        "ask": ask,
        "midpoint": midpoint,
        "bid_ask_spread_pct": spread_pct,
        "premium_per_contract": bid * 100.0,
        "cash_secured": cash_secured,
        "max_profit": bid * 100.0,
        "max_loss": max_loss,
        "break_even": break_even,
        "annualized_return_on_cash": annualized_return,
        "otm_cushion": otm_cushion,
        "breakeven_cushion": breakeven_cushion,
        "expected_move_pct": expected_move_pct,
        "estimated_assignment_probability": assignment_probability,
        "estimated_expiry_loss_probability": loss_probability,
        "estimated_probability_otm": 1.0 - assignment_probability,
        "implied_volatility": used_iv,
        "quoted_implied_volatility": quoted_iv if quoted_iv > 0 else None,
        "historical_volatility": historical_volatility,
        "iv_to_hv": iv_to_hv,
        "iv_rank": iv_rank,
        "iv_rank_source": iv_context.get("source", "Unavailable"),
        "iv_percentile": iv_context.get("iv_percentile"),
        "option_delta": greeks["delta"],
        "short_position_delta": short_delta,
        "target_short_delta": preferences.target_short_delta,
        "delta_target_gap": abs(short_delta - preferences.target_short_delta),
        "delta_target_score": delta_target_score * 100.0,
        "assignment_safety_score": assignment_safety_score * 100.0,
        "risk_target_safety_blend": safety_blend,
        "short_position_theta_per_day": -greeks["theta_per_day"],
        "short_position_gamma": -greeks["gamma"],
        "short_position_vega_per_vol_point": -greeks["vega_per_vol_point"],
        "open_interest": open_interest,
        "volume": volume,
        "score": round(total_score, 2),
        "score_components": {name: round(value, 2) for name, value in components.items()},
        "score_weights": dict(weights),
        "data_quality": "Historical-volatility fallback" if iv_is_fallback else "Quoted IV",
        "reason": (
            f"{short_delta:.2f} short delta versus {preferences.target_short_delta:.2f} profile target; "
            f"{(1.0 - assignment_probability):.0%} modeled chance of expiring OTM; "
            f"{breakeven_cushion:.1%} spot-to-breakeven buffer; "
            f"{annualized_return:.1%} annualized bid yield."
        ),
    }
    return recommendation, None


def score_short_put_chain(
    contracts: Iterable[Mapping[str, Any]],
    spot: float,
    historical_volatility: float,
    preferences: ShortPutPreferences | None = None,
    *,
    rate: float = 0.0425,
    dividend_yield: float = 0.0,
    iv_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Filter and rank put quotes for a cash-secured short-put strategy."""

    if spot <= 0:
        raise ValueError("Spot price must be positive")
    if historical_volatility <= 0:
        raise ValueError("Historical volatility must be positive")
    prefs = preferences or ShortPutPreferences.for_profile("lowest_risk")
    context = iv_context or {}
    candidates = list(contracts)
    rejected_counts: dict[str, int] = {}
    recommendations: list[dict[str, Any]] = []

    for contract in candidates:
        scored, rejection = _score_contract(
            contract,
            spot,
            historical_volatility,
            prefs,
            rate,
            dividend_yield,
            context,
        )
        if scored is not None:
            recommendations.append(scored)
        elif rejection:
            rejected_counts[rejection] = rejected_counts.get(rejection, 0) + 1

    recommendations.sort(
        key=lambda item: (
            item["score"],
            -item["delta_target_gap"],
            item["annualized_return_on_cash"],
            -item["estimated_assignment_probability"],
        ),
        reverse=True,
    )
    recommendations = recommendations[: prefs.limit]
    for rank, recommendation in enumerate(recommendations, start=1):
        recommendation["rank"] = rank

    return {
        "recommendations": recommendations,
        "eligible_count": len(recommendations),
        "eligible_before_limit": sum(
            1
            for contract in candidates
            if contract is not None
        ) - sum(rejected_counts.values()),
        "input_count": len(candidates),
        "rejected_counts": dict(sorted(rejected_counts.items(), key=lambda item: (-item[1], item[0]))),
        "profile": prefs.profile,
        "constraints": {
            "dte": [prefs.min_dte, prefs.max_dte],
            "target_dte": prefs.target_dte,
            "max_assignment_probability": prefs.max_assignment_probability,
            "min_open_interest": prefs.min_open_interest,
            "max_bid_ask_spread_pct": prefs.max_bid_ask_spread_pct,
            "max_cash_secured": prefs.max_cash_secured,
            "short_delta": [prefs.min_short_delta, prefs.max_short_delta],
            "target_short_delta": prefs.target_short_delta,
        },
    }


def score_premium_funded_bullish_pairs(
    put_candidates: Iterable[Mapping[str, Any]],
    call_contracts: Iterable[Mapping[str, Any]],
    spot: float,
    historical_volatility: float,
    preferences: BullishComboPreferences | None = None,
    *,
    rate: float = 0.0425,
    dividend_yield: float = 0.0,
) -> dict[str, Any]:
    """Rank same-expiry OTM short-put/long-call bullish risk reversals.

    The short put is valued at its bid and the long call at its ask.  With the
    default zero additional budget, only combinations whose call is fully paid
    for by the put credit are eligible.
    """

    if spot <= 0 or historical_volatility <= 0:
        raise ValueError("Spot price and historical volatility must be positive")
    prefs = preferences or BullishComboPreferences.for_profile("balanced")
    puts = list(put_candidates)
    calls = list(call_contracts)
    rejected_counts: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

    prepared_calls: dict[str, list[dict[str, Any]]] = {}
    for contract in calls:
        expiration = str(contract.get("expiration") or "")
        strike = _number(contract.get("strike"))
        bid = _number(contract.get("bid"))
        ask = _number(contract.get("ask"))
        dte = int(_number(contract.get("dte")))
        open_interest = int(_number(contract.get("openInterest")))
        volume = int(_number(contract.get("volume")))
        if not expiration or strike <= 0 or bid < 0 or ask <= 0 or ask < bid or dte <= 0:
            reject("invalid long-call quote")
            continue
        if strike <= spot:
            reject("call is not out of the money")
            continue
        midpoint = (bid + ask) / 2.0
        spread_pct = (ask - bid) / midpoint if midpoint > 0 else math.inf
        if spread_pct > prefs.max_call_spread_pct:
            reject("call bid/ask spread too wide")
            continue
        if open_interest < prefs.min_call_open_interest:
            reject("call open interest below minimum")
            continue

        quoted_iv = _number(contract.get("impliedVolatility"), -1.0)
        used_iv = _number(contract.get("ivUsed"), -1.0)
        iv_is_fallback = quoted_iv < 0.02
        if used_iv < 0.02:
            used_iv = historical_volatility
            iv_is_fallback = True
        if used_iv <= 0:
            reject("missing call volatility")
            continue
        metrics = black_scholes_metrics(
            spot,
            strike,
            dte / 365.25,
            rate,
            used_iv,
            dividend_yield,
            "call",
        )
        if not prefs.min_call_delta <= metrics["delta"] <= prefs.max_call_delta:
            reject("call delta outside target range")
            continue

        spread_score = _clamp(1.0 - spread_pct / prefs.max_call_spread_pct)
        oi_target = max(1_000, prefs.min_call_open_interest * 5)
        oi_score = _clamp(math.log1p(open_interest) / math.log1p(oi_target))
        volume_score = _clamp(math.log1p(volume) / math.log1p(250))
        liquidity_score = 0.60 * spread_score + 0.30 * oi_score + 0.10 * volume_score
        prepared_calls.setdefault(expiration, []).append(
            {
                "contract_symbol": contract.get("contractSymbol", ""),
                "expiration": expiration,
                "dte": dte,
                "strike": strike,
                "bid": bid,
                "ask": ask,
                "midpoint": midpoint,
                "spread_pct": spread_pct,
                "open_interest": open_interest,
                "volume": volume,
                "implied_volatility": used_iv,
                "quoted_implied_volatility": quoted_iv if quoted_iv > 0 else None,
                "iv_is_fallback": iv_is_fallback,
                "liquidity_score": liquidity_score,
                **metrics,
            }
        )

    weights = COMBO_SCORE_WEIGHTS[prefs.profile]
    recommendations: list[dict[str, Any]] = []
    pair_input_count = 0
    extra_budget_per_share = prefs.max_extra_debit / 100.0

    for put in puts:
        expiration = str(put.get("expiration") or "")
        matching_calls = prepared_calls.get(expiration, [])
        for call in matching_calls:
            pair_input_count += 1
            put_bid = _number(put.get("bid"))
            available_call_budget = put_bid + extra_budget_per_share
            if call["ask"] > available_call_budget + 1e-9:
                reject("call ask exceeds put-credit budget")
                continue
            premium_utilization = min(call["ask"] / put_bid, 1.0) if put_bid > 0 else 0.0
            if premium_utilization < prefs.min_premium_utilization:
                reject("too little put premium used for the call")
                continue

            put_strike = _number(put.get("strike"))
            put_credit = put_bid * 100.0
            call_cost = call["ask"] * 100.0
            net_credit_per_share = put_bid - call["ask"]
            net_credit = net_credit_per_share * 100.0
            additional_debit = max(-net_credit, 0.0)
            lower_break_even = put_strike - net_credit_per_share
            cash_secured = put_strike * 100.0
            capital_required = cash_secured + additional_debit
            max_loss = max(put_strike - net_credit_per_share, 0.0) * 100.0

            assignment_score = _number(
                put.get("assignment_safety_score"), 0.0
            ) / 100.0
            breakeven_score = _number(
                put.get("score_components", {}).get("breakeven_safety"), 0.0
            ) / 100.0
            downside_score = 0.60 * assignment_score + 0.40 * breakeven_score

            call_delta_score = _clamp(call["delta"] / 0.50)
            call_probability_score = _clamp(call["probability_itm"] / 0.40)
            expected_move_dollars = spot * call["implied_volatility"] * math.sqrt(call["dte"] / 365.25)
            call_distance_expected_moves = (
                (call["strike"] - spot) / expected_move_dollars
                if expected_move_dollars > 0
                else math.inf
            )
            call_reach_score = _clamp(1.0 - call_distance_expected_moves / 2.0)
            upside_score = (
                0.55 * call_delta_score
                + 0.25 * call_probability_score
                + 0.20 * call_reach_score
            )

            if prefs.max_extra_debit > 0:
                extra_budget_score = 1.0 - _clamp(additional_debit / prefs.max_extra_debit)
            else:
                extra_budget_score = 1.0 if additional_debit <= 1e-9 else 0.0
            funding_score = 0.75 * _clamp(premium_utilization) + 0.25 * extra_budget_score

            put_liquidity_score = _number(
                put.get("score_components", {}).get("liquidity"), 0.0
            ) / 100.0
            liquidity_score = 0.45 * put_liquidity_score + 0.55 * call["liquidity_score"]

            put_short_theta = _number(put.get("short_position_theta_per_day"))
            put_short_gamma = _number(put.get("short_position_gamma"))
            put_short_vega = _number(put.get("short_position_vega_per_vol_point"))
            net_delta = _number(put.get("short_position_delta")) + call["delta"]
            net_theta = put_short_theta + call["theta_per_day"]
            net_gamma = put_short_gamma + call["gamma"]
            net_vega = put_short_vega + call["vega_per_vol_point"]
            theta_denominator = abs(put_short_theta) + abs(call["theta_per_day"])
            gamma_denominator = abs(put_short_gamma) + abs(call["gamma"])
            vega_denominator = abs(put_short_vega) + abs(call["vega_per_vol_point"])
            theta_balance = _clamp(
                0.5 + 0.5 * net_theta / theta_denominator
            ) if theta_denominator > 0 else 0.5
            gamma_balance = _clamp(
                0.5 + 0.5 * net_gamma / gamma_denominator
            ) if gamma_denominator > 0 else 0.5
            vega_balance = _clamp(
                0.5 + 0.5 * net_vega / vega_denominator
            ) if vega_denominator > 0 else 0.5
            net_greeks_score = 0.45 * theta_balance + 0.35 * gamma_balance + 0.20 * vega_balance

            put_iv = _number(put.get("implied_volatility"), historical_volatility)
            iv_skew = put_iv - call["implied_volatility"]
            iv_skew_score = _clamp(0.5 + iv_skew / 0.20)
            dte_fit_score = _number(
                put.get("score_components", {}).get("dte_theta"), 50.0
            ) / 100.0

            components = {
                "downside_safety": downside_score * 100.0,
                "upside_participation": upside_score * 100.0,
                "funding_efficiency": funding_score * 100.0,
                "liquidity": liquidity_score * 100.0,
                "net_greeks": net_greeks_score * 100.0,
                "iv_skew": iv_skew_score * 100.0,
                "dte_fit": dte_fit_score * 100.0,
            }
            raw_score = sum(components[name] * weight for name, weight in weights.items())
            data_quality_factor = 0.90 if call["iv_is_fallback"] else 1.0
            total_score = raw_score * data_quality_factor

            def expiry_profit(expiry_spot: float) -> float:
                call_payoff = max(expiry_spot - call["strike"], 0.0)
                put_payoff = -max(put_strike - expiry_spot, 0.0)
                return (call_payoff + put_payoff + net_credit_per_share) * 100.0

            expected_move_pct = call["implied_volatility"] * math.sqrt(call["dte"] / 365.25)
            recommendation = {
                "put_contract_symbol": put.get("contract_symbol", ""),
                "call_contract_symbol": call["contract_symbol"],
                "expiration": expiration,
                "dte": call["dte"],
                "spot": spot,
                "put_strike": put_strike,
                "put_bid": put_bid,
                "put_credit": put_credit,
                "put_implied_volatility": put_iv,
                "put_open_interest": int(_number(put.get("open_interest"))),
                "put_spread_pct": _number(put.get("bid_ask_spread_pct")),
                "estimated_assignment_probability": _number(
                    put.get("estimated_assignment_probability")
                ),
                "short_put_delta": _number(put.get("short_position_delta")),
                "call_strike": call["strike"],
                "call_bid": call["bid"],
                "call_ask": call["ask"],
                "call_cost": call_cost,
                "call_implied_volatility": call["implied_volatility"],
                "call_open_interest": call["open_interest"],
                "call_spread_pct": call["spread_pct"],
                "long_call_delta": call["delta"],
                "long_call_probability_itm": call["probability_itm"],
                "call_distance_expected_moves": call_distance_expected_moves,
                "net_credit": net_credit,
                "net_debit": max(-net_credit, 0.0),
                "additional_debit": additional_debit,
                "premium_utilization": premium_utilization,
                "call_funded_by_put_pct": min(put_credit / call_cost, 1.0) if call_cost > 0 else 0.0,
                "cash_secured": cash_secured,
                "capital_required": capital_required,
                "lower_break_even": lower_break_even,
                "call_activation_price": call["strike"],
                "upper_break_even": call["strike"] - net_credit_per_share if net_credit < 0 else None,
                "max_profit": float("inf"),
                "max_loss": max_loss,
                "net_delta": net_delta,
                "net_theta_per_day": net_theta,
                "net_gamma": net_gamma,
                "net_vega_per_vol_point": net_vega,
                "iv_skew": iv_skew,
                "expected_move_pct": expected_move_pct,
                "profit_at_up_10_pct": expiry_profit(spot * 1.10),
                "profit_at_down_10_pct": expiry_profit(spot * 0.90),
                "profit_at_expected_move_up": expiry_profit(spot * (1.0 + expected_move_pct)),
                "score": round(total_score, 2),
                "score_components": {name: round(value, 2) for name, value in components.items()},
                "score_weights": dict(weights),
                "data_quality": "Historical-volatility fallback" if call["iv_is_fallback"] else "Quoted IV on both legs",
                "reason": (
                    f"Put credit funds {min(put_credit / call_cost, 1.0):.0%} of the call; "
                    f"call delta {call['delta']:.2f}; assignment proxy "
                    f"{_number(put.get('estimated_assignment_probability')):.1%}."
                ),
            }
            recommendations.append(recommendation)

    recommendations.sort(
        key=lambda item: (
            item["score"],
            item["long_call_delta"],
            -item["estimated_assignment_probability"],
        ),
        reverse=True,
    )
    eligible_before_limit = len(recommendations)
    recommendations = recommendations[: prefs.limit]
    for rank, recommendation in enumerate(recommendations, start=1):
        recommendation["rank"] = rank

    return {
        "recommendations": recommendations,
        "eligible_count": len(recommendations),
        "eligible_before_limit": eligible_before_limit,
        "put_input_count": len(puts),
        "call_input_count": len(calls),
        "prepared_call_count": sum(len(items) for items in prepared_calls.values()),
        "pair_input_count": pair_input_count,
        "rejected_counts": dict(sorted(rejected_counts.items(), key=lambda item: (-item[1], item[0]))),
        "profile": prefs.profile,
        "constraints": {
            "call_delta": [prefs.min_call_delta, prefs.max_call_delta],
            "min_call_open_interest": prefs.min_call_open_interest,
            "max_call_spread_pct": prefs.max_call_spread_pct,
            "min_premium_utilization": prefs.min_premium_utilization,
            "max_extra_debit": prefs.max_extra_debit,
            "same_expiration_required": True,
        },
    }
