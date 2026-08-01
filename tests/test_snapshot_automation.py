import datetime
import json
import tempfile
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.publish_option_snapshots import publish_snapshots, validate_snapshot
from scripts.refresh_option_snapshots import (
    build_parser,
    partition_entries,
    run_refresh,
    within_regular_session_window,
)
from valuation.option_snapshots import OptionSnapshotStore
from valuation.option_universe import DEFAULT_UNIVERSE_PATH, load_option_universe
from valuation.options import compact_snapshot_chains


def valid_snapshot(symbol: str = "QQQ") -> dict:
    expiration = "2026-08-28"
    return {
        "schema_version": 1,
        "symbol": symbol,
        "market_date": "2026-07-30",
        "captured_at": "2026-07-30T18:00:00+00:00",
        "session": "regular",
        "market_state": "REGULAR",
        "spot": 100.0,
        "underlying_quote": {
            "price": 100.0,
            "timestamp": "2026-07-30T18:00:00+00:00",
        },
        "historical_volatility": 0.22,
        "dividend_yield": 0.01,
        "risk_free_rate": 0.04,
        "provider": "test fixture",
        "failed_expirations": [],
        "listed_expirations": [expiration],
        "marketable_puts": 1,
        "marketable_calls": 1,
        "marketable_otm_puts": 1,
        "marketable_otm_calls": 1,
        "chains": {
            expiration: {
                "puts": [
                    {
                        "contractSymbol": f"{symbol}P90",
                        "expiration": expiration,
                        "dte": 29,
                        "strike": 90.0,
                        "bid": 2.0,
                        "ask": 2.1,
                        "volume": 100,
                        "openInterest": 1_000,
                        "impliedVolatility": 0.25,
                        "ivUsed": 0.25,
                    }
                ],
                "calls": [
                    {
                        "contractSymbol": f"{symbol}C105",
                        "expiration": expiration,
                        "dte": 29,
                        "strike": 105.0,
                        "bid": 1.7,
                        "ask": 1.8,
                        "volume": 100,
                        "openInterest": 1_000,
                        "impliedVolatility": 0.24,
                        "ivUsed": 0.24,
                    }
                ],
            }
        },
    }


class OptionUniverseTests(unittest.TestCase):
    def test_curated_universe_is_unique_and_covers_requested_groups(self):
        universe = load_option_universe()
        symbols = [entry["symbol"] for entry in universe["entries"]]

        self.assertEqual(len(symbols), 52)
        self.assertEqual(len(symbols), len(set(symbols)))
        self.assertTrue({"SPY", "QQQ", "IWM", "DIA", "SMH", "SOXX", "TLT", "GLD"}.issubset(symbols))
        self.assertTrue({"NVDA", "AMD", "AVGO", "TSM", "ASML", "AMAT", "LRCX", "KLAC"}.issubset(symbols))
        self.assertTrue({"MU", "SNDK", "WDC", "STX"}.issubset(symbols))

    def test_six_batches_are_disjoint_and_complete(self):
        entries = load_option_universe()["entries"]
        batches = [partition_entries(entries, index, 6) for index in range(6)]
        flattened = [entry["symbol"] for batch in batches for entry in batch]

        self.assertEqual(len(flattened), len(entries))
        self.assertEqual(set(flattened), {entry["symbol"] for entry in entries})
        self.assertLessEqual(max(map(len, batches)) - min(map(len, batches)), 1)


class SnapshotRefreshTests(unittest.TestCase):
    def test_regular_session_window_uses_new_york_time(self):
        eastern = ZoneInfo("America/New_York")
        self.assertTrue(within_regular_session_window(datetime.datetime(2026, 7, 30, 13, 17, tzinfo=eastern)))
        self.assertFalse(within_regular_session_window(datetime.datetime(2026, 7, 30, 9, 29, tzinfo=eastern)))
        self.assertFalse(within_regular_session_window(datetime.datetime(2026, 8, 1, 13, 17, tzinfo=eastern)))

    def test_dry_run_partitions_without_contacting_a_provider(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            status_path = root / "status" / "batch-2.json"
            args = build_parser().parse_args(
                [
                    "--dry-run",
                    "--batch-index",
                    "2",
                    "--batch-count",
                    "6",
                    "--output",
                    str(root / "snapshots"),
                    "--status-output",
                    str(status_path),
                ]
            )

            result = run_refresh(args)

            self.assertEqual(result["state"], "dry_run")
            self.assertEqual(result["selected"], len(result["selected_symbols"]))
            self.assertTrue(status_path.exists())
            self.assertFalse((root / "snapshots").exists())


class SnapshotStorageTests(unittest.TestCase):
    def test_snapshot_compaction_keeps_replay_fields_only(self):
        chains = valid_snapshot()["chains"]
        put = chains["2026-08-28"]["puts"][0]
        put.update({"gamma": 0.01, "theta": -0.03, "lastTradeDate": "2026-07-30T17:55:00Z"})
        chains["2026-08-28"]["puts"].extend(
            [
                {**put, "contractSymbol": "QQQP110", "strike": 110.0},
                {**put, "contractSymbol": "QQQP80", "strike": 80.0, "bid": 0.0, "ask": 0.0},
            ]
        )

        compacted = compact_snapshot_chains(chains, 100.0)
        compacted_put = compacted["2026-08-28"]["puts"][0]

        self.assertEqual(len(compacted["2026-08-28"]["puts"]), 1)
        self.assertEqual(compacted_put["contractSymbol"], "QQQP90")
        self.assertEqual(compacted_put["openInterest"], 1_000)
        self.assertNotIn("gamma", compacted_put)
        self.assertNotIn("theta", compacted_put)
        self.assertNotIn("lastTradeDate", compacted_put)

    def test_runtime_store_reads_a_bundled_fallback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fallback = root / "bundled"
            OptionSnapshotStore(fallback, fallback_roots=()).save(valid_snapshot())
            store = OptionSnapshotStore(root / "runtime", fallback_roots=(fallback,))

            loaded = store.load_latest("qqq")

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["market_date"], "2026-07-30")


class SnapshotPublishingTests(unittest.TestCase):
    def test_validation_rejects_off_session_or_unmarketable_data(self):
        snapshot = valid_snapshot()
        snapshot["session"] = "post"
        self.assertEqual(validate_snapshot(snapshot), (False, "snapshot is not from the regular session"))

        snapshot = valid_snapshot()
        snapshot["chains"]["2026-08-28"]["puts"][0]["bid"] = 0.0
        snapshot["marketable_otm_puts"] = 0
        self.assertEqual(
            validate_snapshot(snapshot),
            (False, "no marketable OTM puts in the persisted chain"),
        )

        snapshot = valid_snapshot()
        snapshot["captured_at"] = "2026-07-31T00:01:00+00:00"
        self.assertEqual(
            validate_snapshot(snapshot),
            (False, "capture time is outside the regular session"),
        )

    def test_publisher_keeps_last_good_snapshot_after_failed_refresh(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_root = root / "collected-first"
            snapshot_path = input_root / "option_snapshots" / "QQQ" / "2026-07-30.json"
            snapshot_path.parent.mkdir(parents=True)
            snapshot_path.write_text(json.dumps(valid_snapshot()), encoding="utf-8")
            status_path = input_root / "status" / "batch-0.json"
            status_path.parent.mkdir(parents=True)
            status_path.write_text(
                json.dumps(
                    {
                        "symbols": {
                            "QQQ": {
                                "status": "saved",
                                "message": "Validated regular-session snapshot saved.",
                                "completed_at": "2026-07-30T18:01:00+00:00",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            destination = root / "published"
            published_status = root / "option_snapshot_status.json"

            first = publish_snapshots(input_root, destination, DEFAULT_UNIVERSE_PATH, published_status)
            latest_path = destination / "QQQ" / "latest.json"

            self.assertEqual(first["state"], "complete")
            self.assertEqual(first["summary"]["published_this_run"], 1)
            self.assertTrue(latest_path.exists())
            first_contents = latest_path.read_text(encoding="utf-8")

            failed_input = root / "collected-failed"
            failed_status_path = failed_input / "status" / "batch-0.json"
            failed_status_path.parent.mkdir(parents=True)
            failed_status_path.write_text(
                json.dumps(
                    {
                        "symbols": {
                            "QQQ": {
                                "status": "outside_regular_session",
                                "message": "Provider reported a closed market.",
                                "completed_at": "2026-07-31T18:01:00+00:00",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            second = publish_snapshots(failed_input, destination, DEFAULT_UNIVERSE_PATH, published_status)

            self.assertEqual(second["state"], "no_new_snapshots")
            self.assertEqual(second["summary"]["available"], 1)
            self.assertEqual(second["summary"]["published_this_run"], 0)
            self.assertEqual(latest_path.read_text(encoding="utf-8"), first_contents)
            self.assertEqual(second["symbols"]["QQQ"]["attempt_status"], "outside_regular_session")


if __name__ == "__main__":
    unittest.main()
