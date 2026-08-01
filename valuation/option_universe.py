"""Curated scheduled-option universe and published refresh status helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNIVERSE_PATH = PROJECT_ROOT / "data" / "option_universe.json"
DEFAULT_STATUS_PATH = PROJECT_ROOT / "data" / "option_snapshot_status.json"


def load_option_universe(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else DEFAULT_UNIVERSE_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    categories = payload.get("categories")
    if payload.get("schema_version") != 1 or not isinstance(categories, list) or not categories:
        raise ValueError("The option universe must contain schema version 1 and at least one category.")

    normalized_categories = []
    entries = []
    seen: set[str] = set()
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError("Every option-universe category must be an object.")
        category_name = str(category.get("name") or "").strip()
        raw_symbols = category.get("symbols")
        if not category_name or not isinstance(raw_symbols, list) or not raw_symbols:
            raise ValueError("Every option-universe category needs a name and at least one symbol.")

        category_entries = []
        for item in raw_symbols:
            if not isinstance(item, dict):
                raise ValueError(f"Every symbol in {category_name} must be an object.")
            symbol = str(item.get("symbol") or "").strip().upper()
            name = str(item.get("name") or "").strip()
            if not symbol or not name:
                raise ValueError(f"Every symbol in {category_name} needs a ticker and name.")
            if symbol in seen:
                raise ValueError(f"Duplicate scheduled option symbol: {symbol}")
            seen.add(symbol)
            entry = {"symbol": symbol, "name": name, "category": category_name}
            entries.append(entry)
            category_entries.append(entry)
        normalized_categories.append({"name": category_name, "symbols": category_entries})

    return {
        "schema_version": 1,
        "description": str(payload.get("description") or ""),
        "categories": normalized_categories,
        "entries": entries,
    }


def option_universe_symbols(path: str | Path | None = None) -> list[str]:
    return [entry["symbol"] for entry in load_option_universe(path)["entries"]]


def option_universe_labels(path: str | Path | None = None) -> dict[str, str]:
    return {
        entry["symbol"]: f'{entry["symbol"]} - {entry["name"]} | {entry["category"]}'
        for entry in load_option_universe(path)["entries"]
    }


def load_option_snapshot_status(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else DEFAULT_STATUS_PATH
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {
            "schema_version": 1,
            "generated_at": None,
            "state": "status_unavailable",
            "summary": {},
            "symbols": {},
        }
    return payload if isinstance(payload, dict) else {}
