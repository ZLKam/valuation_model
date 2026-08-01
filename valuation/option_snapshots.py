"""Persistent, point-in-time option-chain snapshots for planning scans."""

from __future__ import annotations

import datetime
import json
import logging
import math
import uuid
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

SNAPSHOT_SCHEMA_VERSION = 1
DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / ".cache" / "option_snapshots"


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy/datetime values into strict JSON-compatible values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except (TypeError, ValueError):
            pass

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except (TypeError, ValueError):
            pass
    return str(value)


def _safe_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    safe = "".join(character if character.isalnum() or character in {".", "-", "_"} else "_" for character in normalized)
    return safe or "UNKNOWN"


def _timestamp(value: Any) -> datetime.datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)


class OptionSnapshotStore:
    """Save and retrieve the latest aligned regular-session chain per symbol."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else DEFAULT_SNAPSHOT_DIR

    def save(self, snapshot: dict[str, Any]) -> Path:
        payload = _json_safe(snapshot)
        symbol = str(payload.get("symbol") or "").strip().upper()
        market_date = str(payload.get("market_date") or "")
        captured_at = _timestamp(payload.get("captured_at"))
        chains = payload.get("chains")
        spot = payload.get("spot")
        if (
            not symbol
            or not market_date
            or captured_at is None
            or not isinstance(chains, dict)
            or not chains
            or not isinstance(spot, (int, float))
            or spot <= 0
        ):
            raise ValueError("Option snapshot is missing symbol, date, spot, capture time, or chains")

        payload["schema_version"] = SNAPSHOT_SCHEMA_VERSION
        symbol_dir = self.root / _safe_symbol(symbol)
        symbol_dir.mkdir(parents=True, exist_ok=True)
        target = symbol_dir / f"{market_date}.json"
        temporary = symbol_dir / f".{market_date}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            temporary.replace(target)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return target

    def load_latest(self, symbol: str) -> dict[str, Any] | None:
        symbol_dir = self.root / _safe_symbol(symbol)
        if not symbol_dir.exists():
            return None

        candidates: list[tuple[datetime.datetime, dict[str, Any]]] = []
        for path in symbol_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("Could not read option snapshot %s: %s", path, exc)
                continue
            captured_at = _timestamp(payload.get("captured_at"))
            if (
                payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
                or str(payload.get("symbol") or "").strip().upper() != symbol.strip().upper()
                or captured_at is None
                or not isinstance(payload.get("chains"), dict)
                or not payload.get("chains")
            ):
                continue
            candidates.append((captured_at, payload))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
