"""Validate batch artifacts and publish one rolling snapshot per configured symbol."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from valuation.option_snapshots import SNAPSHOT_SCHEMA_VERSION
from valuation.option_universe import (
    DEFAULT_STATUS_PATH,
    DEFAULT_UNIVERSE_PATH,
    load_option_snapshot_status,
    load_option_universe,
)


EASTERN = ZoneInfo("America/New_York")


def _timestamp(value: Any) -> datetime.datetime | None:
    try:
        parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)


def _number(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def validate_snapshot(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "snapshot is not a JSON object"
    symbol = str(payload.get("symbol") or "").strip().upper()
    market_date_text = str(payload.get("market_date") or "")
    try:
        market_date = datetime.date.fromisoformat(market_date_text)
    except ValueError:
        return False, "invalid market date"
    if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        return False, "unsupported schema version"
    captured_at = _timestamp(payload.get("captured_at"))
    if not symbol or captured_at is None:
        return False, "missing symbol or capture time"
    captured_eastern = captured_at.astimezone(EASTERN)
    if captured_eastern.date() != market_date:
        return False, "capture date does not match the market date"
    if not (
        captured_eastern.weekday() < 5
        and datetime.time(9, 30) <= captured_eastern.time() < datetime.time(16, 0)
    ):
        return False, "capture time is outside the regular session"
    if str(payload.get("session") or "").lower() != "regular":
        return False, "snapshot is not from the regular session"
    if str(payload.get("market_state") or "").upper() not in {"REGULAR", "OPEN"}:
        return False, "provider did not report an open regular session"
    if _number(payload.get("spot")) <= 0:
        return False, "invalid aligned underlying price"
    if payload.get("failed_expirations"):
        return False, "one or more expirations failed"
    chains = payload.get("chains")
    if not isinstance(chains, dict) or not chains:
        return False, "option chains are missing"

    spot = _number(payload.get("spot"))
    marketable_otm_puts = 0
    for sides in chains.values():
        if not isinstance(sides, dict):
            continue
        for contract in sides.get("puts") or []:
            if not isinstance(contract, dict):
                continue
            strike = _number(contract.get("strike"))
            bid = _number(contract.get("bid"))
            ask = _number(contract.get("ask"))
            if 0 < strike < spot and bid >= 0.05 and ask >= bid:
                marketable_otm_puts += 1
    if marketable_otm_puts <= 0:
        return False, "no marketable OTM puts in the persisted chain"
    return True, "ready"


def _atomic_json_write(path: Path, payload: dict[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        if compact:
            rendered = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
        else:
            rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_batch_statuses(input_root: Path) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in sorted(input_root.rglob("batch-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        for symbol, item in (payload.get("symbols") or {}).items():
            if isinstance(item, dict):
                merged[str(symbol).upper()] = item
    return merged


def _candidate_snapshots(input_root: Path) -> dict[str, tuple[datetime.datetime, dict[str, Any]]]:
    candidates: dict[str, tuple[datetime.datetime, dict[str, Any]]] = {}
    for path in input_root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        valid, _ = validate_snapshot(payload)
        if not valid:
            continue
        symbol = str(payload["symbol"]).strip().upper()
        captured_at = _timestamp(payload["captured_at"])
        if captured_at is None:
            continue
        if symbol not in candidates or captured_at > candidates[symbol][0]:
            candidates[symbol] = (captured_at, payload)
    return candidates


def publish_snapshots(
    input_root: str | Path,
    destination: str | Path,
    universe_path: str | Path,
    status_output: str | Path,
) -> dict[str, Any]:
    input_root = Path(input_root)
    destination = Path(destination)
    status_output = Path(status_output)
    universe = load_option_universe(universe_path)
    configured = {entry["symbol"]: entry for entry in universe["entries"]}
    batch_statuses = _load_batch_statuses(input_root)
    candidates = _candidate_snapshots(input_root)
    previous_status = load_option_snapshot_status(status_output)
    published: list[str] = []

    for symbol, (_, payload) in candidates.items():
        if symbol not in configured:
            continue
        target = destination / symbol / "latest.json"
        _atomic_json_write(target, payload, compact=True)
        published.append(symbol)

    destination.mkdir(parents=True, exist_ok=True)
    for symbol_dir in destination.iterdir():
        if not symbol_dir.is_dir():
            continue
        if symbol_dir.name.upper() not in configured:
            continue
        for path in symbol_dir.glob("*.json"):
            if path.name != "latest.json":
                path.unlink(missing_ok=True)

    generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    symbol_status: dict[str, dict[str, Any]] = {}
    prior_symbols = previous_status.get("symbols") if isinstance(previous_status, dict) else {}
    if not isinstance(prior_symbols, dict):
        prior_symbols = {}

    for symbol, entry in configured.items():
        attempt = batch_statuses.get(symbol)
        latest_path = destination / symbol / "latest.json"
        latest_payload: dict[str, Any] = {}
        if latest_path.exists():
            try:
                latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                latest_payload = {}
        previous = prior_symbols.get(symbol) if isinstance(prior_symbols.get(symbol), dict) else {}
        symbol_status[symbol] = {
            **entry,
            "attempt_status": (attempt or {}).get("status", previous.get("attempt_status", "not_attempted")),
            "attempt_message": (attempt or {}).get("message", previous.get("attempt_message", "")),
            "last_attempt_at": (attempt or {}).get("completed_at", previous.get("last_attempt_at")),
            "last_snapshot_market_date": latest_payload.get("market_date"),
            "last_snapshot_at": latest_payload.get("captured_at"),
            "spot": latest_payload.get("spot"),
            "marketable_otm_puts": latest_payload.get("marketable_otm_puts", 0),
            "marketable_otm_calls": latest_payload.get("marketable_otm_calls", 0),
            "available": bool(latest_payload),
        }

    attempted = len(batch_statuses)
    saved = sum(item.get("status") == "saved" for item in batch_statuses.values())
    available = sum(bool(item.get("available")) for item in symbol_status.values())
    failed = attempted - saved
    if attempted == 0:
        state = "no_batch_results"
    elif saved == attempted:
        state = "complete"
    elif saved > 0:
        state = "partial"
    else:
        state = "no_new_snapshots"

    status = {
        "schema_version": 1,
        "generated_at": generated_at,
        "state": state,
        "summary": {
            "configured": len(configured),
            "attempted": attempted,
            "saved": saved,
            "failed": failed,
            "available": available,
            "published_this_run": len(published),
        },
        "symbols": symbol_status,
    }
    _atomic_json_write(status_output, status, compact=False)
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish validated rolling option snapshots.")
    parser.add_argument("--input", default="collected")
    parser.add_argument("--destination", default="data/option_snapshots")
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE_PATH))
    parser.add_argument("--status-output", default=str(DEFAULT_STATUS_PATH))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        status = publish_snapshots(args.input, args.destination, args.universe, args.status_output)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Snapshot publication failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"state": status["state"], "summary": status["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
