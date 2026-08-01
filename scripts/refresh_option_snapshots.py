"""Fetch validated regular-session option snapshots for a configured batch."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from valuation.option_scoring import ShortPutPreferences
from valuation.option_snapshots import OptionSnapshotStore
from valuation.option_universe import DEFAULT_UNIVERSE_PATH, load_option_universe
from valuation.options import OptionsAnalyzer, QUOTE_BASIS_LIVE


logger = logging.getLogger("option_snapshot_refresh")
EASTERN = ZoneInfo("America/New_York")
TERMINAL_SAVE_STATUSES = {
    "outside_regular_session",
    "no_marketable_otm_puts",
}


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def within_regular_session_window(now: datetime.datetime | None = None) -> bool:
    current = (now or utc_now()).astimezone(EASTERN)
    return current.weekday() < 5 and datetime.time(9, 30) <= current.time() < datetime.time(16, 0)


def partition_entries(entries: list[dict[str, str]], batch_index: int, batch_count: int) -> list[dict[str, str]]:
    if batch_count < 1:
        raise ValueError("Batch count must be at least one.")
    if not 0 <= batch_index < batch_count:
        raise ValueError("Batch index must be between zero and batch count minus one.")
    return [entry for offset, entry in enumerate(entries) if offset % batch_count == batch_index]


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _entry_map(entries: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {entry["symbol"]: entry for entry in entries}


def select_entries(entries: list[dict[str, str]], symbols: str | None) -> list[dict[str, str]]:
    if not symbols:
        return entries
    configured = _entry_map(entries)
    selected = []
    seen: set[str] = set()
    for raw_symbol in symbols.split(","):
        symbol = raw_symbol.strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        selected.append(configured.get(symbol, {"symbol": symbol, "name": symbol, "category": "Manual"}))
    return selected


def refresh_symbol(
    entry: dict[str, str],
    store: OptionSnapshotStore,
    *,
    attempts: int,
    retry_delay_seconds: float,
    expiration_limit: int,
) -> dict[str, Any]:
    symbol = entry["symbol"]
    preferences = ShortPutPreferences.for_profile("balanced").with_dte_window(14, 75)
    last_result: dict[str, Any] = {}
    started_at = utc_now().isoformat()

    for attempt in range(1, attempts + 1):
        analyzer = OptionsAnalyzer(snapshot_store=store)
        try:
            result = analyzer.recommend_short_puts(
                symbol,
                preferences,
                quote_basis=QUOTE_BASIS_LIVE,
                expiration_limit=expiration_limit,
            )
        except Exception as exc:  # defensive isolation for one failed symbol
            result = {"error": str(exc), "snapshot_save_status": "unhandled_error"}
        last_result = result
        save_status = str(result.get("snapshot_save_status") or "not_saved")
        if result.get("snapshot_saved"):
            return {
                **entry,
                "status": "saved",
                "attempts": attempt,
                "started_at": started_at,
                "completed_at": utc_now().isoformat(),
                "market_date": result.get("market_date"),
                "captured_at": result.get("snapshot_captured_at"),
                "spot": result.get("spot"),
                "marketable_otm_puts": result.get("snapshot_marketable_otm_puts", 0),
                "marketable_otm_calls": result.get("snapshot_marketable_otm_calls", 0),
                "message": "Validated regular-session snapshot saved.",
            }
        if save_status in TERMINAL_SAVE_STATUSES:
            break
        if attempt < attempts and retry_delay_seconds > 0:
            time.sleep(retry_delay_seconds)

    message = (
        last_result.get("snapshot_save_error")
        or last_result.get("error")
        or f"Snapshot was not saved ({last_result.get('snapshot_save_status', 'unknown status')})."
    )
    return {
        **entry,
        "status": str(last_result.get("snapshot_save_status") or "failed"),
        "attempts": attempt,
        "started_at": started_at,
        "completed_at": utc_now().isoformat(),
        "market_date": last_result.get("market_date"),
        "captured_at": last_result.get("snapshot_captured_at"),
        "spot": last_result.get("spot"),
        "marketable_otm_puts": last_result.get("snapshot_marketable_otm_puts", 0),
        "marketable_otm_calls": last_result.get("snapshot_marketable_otm_calls", 0),
        "message": str(message),
    }


def run_refresh(args: argparse.Namespace) -> dict[str, Any]:
    universe = load_option_universe(args.universe)
    selected = select_entries(universe["entries"], args.symbols)
    selected = partition_entries(selected, args.batch_index, args.batch_count)
    generated_at = utc_now().isoformat()
    status_path = Path(args.status_output) if args.status_output else (
        Path(args.output).parent / "status" / f"batch-{args.batch_index}.json"
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": generated_at,
        "batch_index": args.batch_index,
        "batch_count": args.batch_count,
        "configured": len(universe["entries"]),
        "selected": len(selected),
        "symbols": {},
    }

    if args.dry_run:
        payload["state"] = "dry_run"
        payload["selected_symbols"] = [entry["symbol"] for entry in selected]
        _atomic_json_write(status_path, payload)
        return payload

    if not args.allow_outside_session and not within_regular_session_window():
        payload["state"] = "outside_regular_session_window"
        payload["symbols"] = {
            entry["symbol"]: {
                **entry,
                "status": "outside_regular_session_window",
                "attempts": 0,
                "started_at": generated_at,
                "completed_at": generated_at,
                "message": "Refresh skipped outside 09:30-16:00 America/New_York on a weekday.",
            }
            for entry in selected
        }
        _atomic_json_write(status_path, payload)
        return payload

    store = OptionSnapshotStore(Path(args.output), fallback_roots=())
    for offset, entry in enumerate(selected):
        logger.info("Refreshing %s (%s)", entry["symbol"], entry["category"])
        payload["symbols"][entry["symbol"]] = refresh_symbol(
            entry,
            store,
            attempts=max(1, args.attempts),
            retry_delay_seconds=max(0.0, args.retry_delay_seconds),
            expiration_limit=max(1, args.expiration_limit),
        )
        if offset < len(selected) - 1 and args.symbol_delay_seconds > 0:
            time.sleep(args.symbol_delay_seconds)

    saved = sum(item.get("status") == "saved" for item in payload["symbols"].values())
    payload["state"] = "complete" if saved == len(selected) else "partial"
    payload["summary"] = {
        "attempted": len(selected),
        "saved": saved,
        "failed": len(selected) - saved,
    }
    _atomic_json_write(status_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh regular-session option snapshots.")
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE_PATH))
    parser.add_argument("--output", default="generated/option_snapshots")
    parser.add_argument("--status-output")
    parser.add_argument("--symbols", help="Optional comma-separated symbols, including custom tickers.")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-count", type=int, default=1)
    parser.add_argument("--expiration-limit", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=8.0)
    parser.add_argument("--symbol-delay-seconds", type=float, default=1.0)
    parser.add_argument("--allow-outside-session", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    try:
        payload = run_refresh(args)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.error("Refresh configuration failed: %s", exc)
        return 2
    print(json.dumps({
        "state": payload.get("state"),
        "batch": payload.get("batch_index"),
        "selected": payload.get("selected"),
        "summary": payload.get("summary", {}),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
