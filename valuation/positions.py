"""Portfolio storage, position sizing, and portfolio-level risk diagnostics."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "portfolio.json")

DEFAULT_PORTFOLIO = {
    "account_value": 100000.0,
    "risk_per_trade": 0.01,
    "max_position_pct": 0.15,
    "max_sector_pct": 0.35,
    "positions": [],
}


def _safe(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def normalize_ticker(ticker_symbol: str) -> str:
    symbol = str(ticker_symbol or "").strip().upper().replace(".", "-")
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9-]{0,9}", symbol):
        raise ValueError("Enter a valid ticker symbol, such as AAPL or BRK-B.")
    return symbol


def _migrate(data: dict) -> dict:
    migrated = dict(DEFAULT_PORTFOLIO)
    migrated.update(data or {})
    if "account_value" not in (data or {}):
        migrated["account_value"] = _safe((data or {}).get("total_margin"), DEFAULT_PORTFOLIO["account_value"])
    migrated.pop("total_margin", None)
    migrated["account_value"] = max(_safe(migrated.get("account_value"), 100000.0), 0.0)
    migrated["risk_per_trade"] = float(np.clip(_safe(migrated.get("risk_per_trade"), 0.01), 0.001, 0.05))
    migrated["max_position_pct"] = float(np.clip(_safe(migrated.get("max_position_pct"), 0.15), 0.03, 0.50))
    migrated["max_sector_pct"] = float(np.clip(_safe(migrated.get("max_sector_pct"), 0.35), 0.10, 1.00))
    raw_positions = migrated.get("positions") or []
    if not isinstance(raw_positions, list):
        raw_positions = []
    migrated["positions"] = [dict(position) for position in raw_positions if isinstance(position, dict)]
    for position in migrated["positions"]:
        position["ticker"] = normalize_ticker(position.get("ticker", ""))
        position["role"] = position.get("role") or position.get("stage") or "Core"
        position["sector"] = position.get("sector") or "Other"
        position["shares"] = max(_safe(position.get("shares")), 0.0)
        position["avg_price"] = max(_safe(position.get("avg_price")), 0.0)
        position["stop_loss"] = max(_safe(position.get("stop_loss")), 0.0)
    return migrated


def normalize_portfolio(data: dict) -> dict:
    """Validate and normalize imported or session-scoped portfolio data."""
    if not isinstance(data, dict):
        raise ValueError("Portfolio data must be a JSON object.")
    return _migrate(data)


def load_portfolio() -> dict:
    if not os.path.exists(PORTFOLIO_FILE):
        return dict(DEFAULT_PORTFOLIO, positions=[])
    try:
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as handle:
            return _migrate(json.load(handle))
    except Exception as exc:
        logger.error("Unable to load portfolio: %s", exc)
        return dict(DEFAULT_PORTFOLIO, positions=[])


def save_portfolio(data: dict) -> bool:
    payload = _migrate(data)
    temporary = f"{PORTFOLIO_FILE}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, PORTFOLIO_FILE)
        return True
    except Exception as exc:
        logger.error("Unable to save portfolio: %s", exc)
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass
        return False


def get_ticker_sector(ticker_symbol: str) -> str:
    symbol = normalize_ticker(ticker_symbol)
    try:
        info = yf.Ticker(symbol).info or {}
        return str(info.get("sector") or info.get("quoteType") or "Other")
    except Exception as exc:
        logger.warning("Unable to retrieve sector for %s: %s", symbol, exc)
        return "Other"


def get_latest_price(ticker_symbol: str) -> float:
    symbol = normalize_ticker(ticker_symbol)
    ticker = yf.Ticker(symbol)
    try:
        fast = ticker.fast_info
        price = _safe(fast.get("last_price") if hasattr(fast, "get") else fast.last_price)
        if price > 0:
            return price
    except Exception:
        pass
    try:
        info = ticker.info or {}
        for key in ("currentPrice", "regularMarketPrice", "previousClose"):
            price = _safe(info.get(key))
            if price > 0:
                return price
    except Exception as exc:
        logger.warning("Unable to retrieve price for %s: %s", symbol, exc)
    return 0.0


class PositionManager:
    @staticmethod
    def _working_portfolio(portfolio: dict | None) -> dict:
        return load_portfolio() if portfolio is None else normalize_portfolio(portfolio)

    @staticmethod
    def _commit(portfolio: dict, persist: bool, error_message: str) -> dict:
        if persist and not save_portfolio(portfolio):
            raise OSError(error_message)
        return portfolio

    @staticmethod
    def update_settings(
        account_value: float,
        risk_per_trade: float,
        max_position_pct: float,
        max_sector_pct: float,
        portfolio: dict | None = None,
        persist: bool = True,
    ) -> dict:
        if account_value <= 0:
            raise ValueError("Account value must be greater than zero.")
        portfolio = PositionManager._working_portfolio(portfolio)
        portfolio.update({
            "account_value": float(account_value),
            "risk_per_trade": float(np.clip(risk_per_trade, 0.001, 0.05)),
            "max_position_pct": float(np.clip(max_position_pct, 0.03, 0.50)),
            "max_sector_pct": float(np.clip(max_sector_pct, 0.10, 1.00)),
        })
        return PositionManager._commit(portfolio, persist, "The portfolio settings could not be saved.")

    @staticmethod
    def update_margin(total_margin: float) -> dict:
        """Backward-compatible alias for the previous account setting."""
        portfolio = load_portfolio()
        return PositionManager.update_settings(
            total_margin,
            portfolio["risk_per_trade"],
            portfolio["max_position_pct"],
            portfolio["max_sector_pct"],
        )

    @staticmethod
    def add_or_update_position(
        ticker: str,
        shares: float,
        avg_price: float,
        stage: str = "Core",
        stop_loss: float = 0.0,
        sector: str | None = None,
        notes: str = "",
        portfolio: dict | None = None,
        persist: bool = True,
    ) -> dict:
        symbol = normalize_ticker(ticker)
        shares = _safe(shares)
        avg_price = _safe(avg_price)
        stop_loss = _safe(stop_loss)
        if shares <= 0 or avg_price <= 0:
            raise ValueError("Shares and purchase price must be greater than zero.")
        if stop_loss < 0 or (stop_loss and stop_loss >= avg_price):
            raise ValueError("The stop price must be below the purchase price.")

        portfolio = PositionManager._working_portfolio(portfolio)
        existing = next((item for item in portfolio["positions"] if item["ticker"] == symbol), None)
        resolved_sector = sector or (existing or {}).get("sector") or get_ticker_sector(symbol)
        role = str(stage or "Core").title()
        if existing:
            old_shares = _safe(existing.get("shares"))
            new_shares = old_shares + shares
            existing.update({
                "shares": new_shares,
                "avg_price": (old_shares * _safe(existing.get("avg_price")) + shares * avg_price) / new_shares,
                "role": role,
                "stage": role,
                "stop_loss": stop_loss,
                "sector": resolved_sector,
                "notes": notes or existing.get("notes", ""),
            })
        else:
            portfolio["positions"].append({
                "ticker": symbol,
                "sector": resolved_sector,
                "shares": shares,
                "avg_price": avg_price,
                "role": role,
                "stage": role,
                "stop_loss": stop_loss,
                "notes": notes,
            })
        return PositionManager._commit(portfolio, persist, "The position could not be saved.")

    @staticmethod
    def set_position(
        ticker: str,
        shares: float,
        avg_price: float,
        stop_loss: float,
        role: str,
        sector: str | None = None,
        notes: str = "",
        portfolio: dict | None = None,
        persist: bool = True,
    ) -> dict:
        """Set the absolute holding values, used by the edit workflow."""
        symbol = normalize_ticker(ticker)
        if shares <= 0 or avg_price <= 0:
            raise ValueError("Shares and average cost must be greater than zero.")
        if stop_loss < 0 or (stop_loss and stop_loss >= avg_price):
            raise ValueError("The stop price must be below average cost.")
        portfolio = PositionManager._working_portfolio(portfolio)
        existing = next((item for item in portfolio["positions"] if item["ticker"] == symbol), None)
        if existing is None:
            raise ValueError(f"{symbol} is not in the portfolio.")
        existing.update({
            "shares": float(shares),
            "avg_price": float(avg_price),
            "stop_loss": float(stop_loss),
            "role": str(role).title(),
            "stage": str(role).title(),
            "sector": sector or existing.get("sector") or "Other",
            "notes": notes,
        })
        return PositionManager._commit(portfolio, persist, "The position could not be updated.")

    @staticmethod
    def remove_position(
        ticker: str,
        portfolio: dict | None = None,
        persist: bool = True,
    ) -> dict:
        symbol = normalize_ticker(ticker)
        portfolio = PositionManager._working_portfolio(portfolio)
        portfolio["positions"] = [item for item in portfolio["positions"] if item["ticker"] != symbol]
        return PositionManager._commit(portfolio, persist, "The position could not be removed.")

    @staticmethod
    def calculate_position_size(
        entry_price: float,
        stop_price: float,
        account_value: float,
        risk_per_trade: float,
        max_position_pct: float,
        existing_value: float = 0.0,
        available_cash: float | None = None,
    ) -> dict:
        entry_price = _safe(entry_price)
        stop_price = _safe(stop_price)
        account_value = _safe(account_value)
        if entry_price <= 0 or account_value <= 0:
            raise ValueError("Entry price and account value must be greater than zero.")
        if stop_price <= 0 or stop_price >= entry_price:
            raise ValueError("Choose a stop price below the planned entry.")

        risk_per_trade = float(np.clip(risk_per_trade, 0.001, 0.05))
        max_position_pct = float(np.clip(max_position_pct, 0.03, 0.50))
        risk_budget = account_value * risk_per_trade
        risk_per_share = entry_price - stop_price
        allocation_room = max(account_value * max_position_pct - max(existing_value, 0.0), 0.0)
        cash_room = allocation_room if available_cash is None else max(float(available_cash), 0.0)

        by_risk = math.floor(risk_budget / risk_per_share)
        by_allocation = math.floor(allocation_room / entry_price)
        by_cash = math.floor(cash_room / entry_price)
        suggested_shares = max(min(by_risk, by_allocation, by_cash), 0)
        constraints = {"risk budget": by_risk, "position limit": by_allocation, "available cash": by_cash}
        limiting_factor = min(constraints, key=constraints.get)

        return {
            "suggested_shares": suggested_shares,
            "position_value": suggested_shares * entry_price,
            "position_pct": suggested_shares * entry_price / account_value,
            "risk_budget": risk_budget,
            "risk_per_share": risk_per_share,
            "max_loss_at_stop": suggested_shares * risk_per_share,
            "stop_distance_pct": risk_per_share / entry_price,
            "allocation_room": allocation_room,
            "limiting_factor": limiting_factor,
            "shares_by_risk": by_risk,
            "shares_by_allocation": by_allocation,
            "shares_by_cash": by_cash,
        }

    @staticmethod
    def calculate_staged_recommendation(
        ticker: str,
        current_price: float,
        total_margin: float,
        stage: str,
        current_positions: list,
    ) -> dict:
        """Backward-compatible wrapper around risk-based position sizing."""
        symbol = normalize_ticker(ticker)
        stage_key = str(stage).lower()
        target_pct = {"watcher": 0.05, "core": 0.15, "addon": 0.25}.get(stage_key, 0.10)
        stop_pct = {"watcher": 0.12, "core": 0.08, "addon": 0.06}.get(stage_key, 0.08)
        existing_value = sum(
            _safe(item.get("shares")) * current_price for item in current_positions if item.get("ticker") == symbol
        )
        sized = PositionManager.calculate_position_size(
            current_price,
            current_price * (1.0 - stop_pct),
            total_margin,
            0.01,
            target_pct,
            existing_value,
        )
        return {
            "stage": stage,
            "target_allocation_pct": target_pct,
            "target_value": total_margin * target_pct,
            "current_value": existing_value,
            "max_addition_value": sized["allocation_room"],
            "suggested_shares": sized["suggested_shares"],
            "suggested_stop_loss": current_price * (1.0 - stop_pct),
            "stop_loss_pct": stop_pct,
        }

    @staticmethod
    def get_portfolio_diagnostics(portfolio: dict, current_prices: dict) -> dict:
        portfolio = _migrate(portfolio)
        account_value = portfolio["account_value"]
        ticker_values: dict[str, float] = {}
        sector_values: dict[str, float] = {}
        total_market_value = 0.0
        total_cost_basis = 0.0
        stop_risk = 0.0
        rows = []

        for position in portfolio["positions"]:
            symbol = position["ticker"]
            shares = _safe(position.get("shares"))
            average = _safe(position.get("avg_price"))
            price = _safe(current_prices.get(symbol), average)
            market_value = shares * price
            cost_basis = shares * average
            stop = _safe(position.get("stop_loss"))
            risk = shares * max(price - stop, 0.0) if stop > 0 else 0.0
            total_market_value += market_value
            total_cost_basis += cost_basis
            stop_risk += risk
            ticker_values[symbol] = ticker_values.get(symbol, 0.0) + market_value
            sector = position.get("sector") or "Other"
            sector_values[sector] = sector_values.get(sector, 0.0) + market_value
            rows.append({
                **position,
                "current_price": price,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "pnl": market_value - cost_basis,
                "pnl_pct": (market_value / cost_basis - 1.0) if cost_basis > 0 else 0.0,
                "allocation_pct": market_value / account_value if account_value > 0 else 0.0,
                "risk_to_stop": risk,
            })

        ticker_allocations = {key: value / account_value for key, value in ticker_values.items()} if account_value else {}
        sector_allocations = {key: value / account_value for key, value in sector_values.items()} if account_value else {}
        warnings = []
        for symbol, allocation in ticker_allocations.items():
            if allocation > portfolio["max_position_pct"]:
                warnings.append({"level": "high", "message": f"{symbol} is {allocation:.1%} of the account, above the {portfolio['max_position_pct']:.0%} limit."})
        for sector, allocation in sector_allocations.items():
            if allocation > portfolio["max_sector_pct"]:
                warnings.append({"level": "high", "message": f"{sector} is {allocation:.1%} of the account, above the {portfolio['max_sector_pct']:.0%} sector limit."})
        if total_market_value > account_value:
            warnings.append({"level": "high", "message": "Holdings exceed account equity; the portfolio is using leverage."})
        if stop_risk / account_value > 0.06 if account_value else False:
            warnings.append({"level": "medium", "message": "Combined risk to recorded stops exceeds 6% of account equity."})

        return {
            "account_value": account_value,
            "total_market_value": total_market_value,
            "total_cost_basis": total_cost_basis,
            "unrealized_pnl": total_market_value - total_cost_basis,
            "unrealized_pnl_pct": (total_market_value / total_cost_basis - 1.0) if total_cost_basis > 0 else 0.0,
            "available_cash": max(account_value - total_market_value, 0.0),
            "invested_pct": total_market_value / account_value if account_value > 0 else 0.0,
            "portfolio_heat": stop_risk / account_value if account_value > 0 else 0.0,
            "ticker_allocations_pct": ticker_allocations,
            "sector_allocations_pct": sector_allocations,
            "positions": rows,
            "warnings": warnings,
            # Compatibility with the previous UI.
            "used_margin_pct": total_market_value / account_value if account_value > 0 else 0.0,
            "ticker_warnings": [warning for warning in warnings if any(symbol in warning["message"] for symbol in ticker_allocations)],
            "sector_warnings": [warning for warning in warnings if any(sector in warning["message"] for sector in sector_allocations)],
            "rebalance_checklist": [warning["message"] for warning in warnings],
        }
