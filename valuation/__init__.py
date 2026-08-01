"""Valuation package configuration."""

from __future__ import annotations

import os

import yfinance as yf


# yfinance persists timezone/cookie data in SQLite. Keeping that cache beside
# the application makes local and container deployments deterministic and
# avoids failures when a service account has no writable home directory.
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "yfinance")
os.makedirs(_CACHE_DIR, exist_ok=True)
yf.set_tz_cache_location(_CACHE_DIR)
