import math
import json
import numpy as np
import scipy.stats as stats
import yfinance as yf
import datetime
import logging
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo

from valuation.option_scoring import (
    BullishComboPreferences,
    ShortPutPreferences,
    black_scholes_metrics,
    calculate_iv_percentile,
    calculate_iv_rank,
    score_premium_funded_bullish_pairs,
    score_short_put_chain,
)
from valuation.option_snapshots import OptionSnapshotStore, SNAPSHOT_SCHEMA_VERSION

logger = logging.getLogger(__name__)

_IV_HISTORY_FILE = Path(__file__).resolve().parent.parent / ".cache" / "option_iv_history.json"
QUOTE_BASIS_AUTO = "auto"
QUOTE_BASIS_LIVE = "live"
QUOTE_BASIS_PREVIOUS_SESSION = "previous_session"
QUOTE_BASES = {QUOTE_BASIS_AUTO, QUOTE_BASIS_LIVE, QUOTE_BASIS_PREVIOUS_SESSION}
_VOLATILITY_PROXIES = {
    "QQQ": ("^VXN", "Nasdaq-100"),
    "SPY": ("^VIX", "S&P 500"),
    "IWM": ("^RVX", "Russell 2000"),
    "DIA": ("^VXD", "Dow Jones Industrial Average"),
}
_SNAPSHOT_CONTRACT_FIELDS = (
    "contractSymbol",
    "expiration",
    "dte",
    "strike",
    "bid",
    "ask",
    "volume",
    "openInterest",
    "impliedVolatility",
    "ivUsed",
)


def _finite_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _replayable_snapshot_contract(contract: dict, aligned_spot: float, option_type: str) -> bool:
    if aligned_spot <= 0:
        return True
    strike = _finite_float(contract.get("strike"), -1.0)
    bid = _finite_float(contract.get("bid"), -1.0)
    ask = _finite_float(contract.get("ask"), -1.0)
    is_otm = strike < aligned_spot if option_type == "puts" else strike > aligned_spot
    return is_otm and bid >= 0.05 and ask >= bid


def compact_snapshot_chains(chains: dict, spot: float | None = None) -> dict:
    """Keep replay-required fields and, when possible, only usable OTM quotes."""
    aligned_spot = _finite_float(spot, -1.0)
    compacted: dict[str, dict[str, list[dict]]] = {}
    for expiration, sides in (chains or {}).items():
        if not isinstance(sides, dict):
            continue
        compacted_sides: dict[str, list[dict]] = {}
        for option_type in ("puts", "calls"):
            contracts = sides.get(option_type) or []
            compacted_sides[option_type] = [
                {field: contract.get(field) for field in _SNAPSHOT_CONTRACT_FIELDS if field in contract}
                for contract in contracts
                if isinstance(contract, dict)
                and _replayable_snapshot_contract(contract, aligned_spot, option_type)
            ]
        compacted[str(expiration)] = compacted_sides
    return compacted


def _market_date() -> datetime.date:
    """Use the US options market date rather than the viewer's local timezone."""
    try:
        return datetime.datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return datetime.date.today()


def _quote_timestamp(value) -> datetime.datetime | None:
    """Normalize provider timestamps to timezone-aware UTC datetimes."""
    if value is None:
        return None
    try:
        if isinstance(value, datetime.datetime):
            parsed = value
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            parsed = datetime.datetime.fromtimestamp(float(value), tz=datetime.timezone.utc)
        else:
            parsed = pd.Timestamp(value).to_pydatetime()
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        return parsed.astimezone(datetime.timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _normalize_quote_basis(value: str | None) -> str:
    normalized = str(value or QUOTE_BASIS_LIVE).strip().lower().replace("-", "_")
    aliases = {
        "current": QUOTE_BASIS_LIVE,
        "current_session": QUOTE_BASIS_LIVE,
        "latest_snapshot": QUOTE_BASIS_PREVIOUS_SESSION,
        "previous": QUOTE_BASIS_PREVIOUS_SESSION,
        "snapshot": QUOTE_BASIS_PREVIOUS_SESSION,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in QUOTE_BASES:
        raise ValueError(f"Unsupported option quote basis: {value}")
    return normalized


def _regular_session_state(market_state: str | None) -> bool:
    return str(market_state or "").upper() in {"REGULAR", "OPEN"}


def _marketable_quote_count(contracts: list[dict], min_bid: float = 0.05) -> int:
    count = 0
    for contract in contracts:
        bid = _finite_float(contract.get("bid"), -1.0)
        ask = _finite_float(contract.get("ask"), -1.0)
        if bid >= min_bid and ask > 0 and ask >= bid:
            count += 1
    return count


def _marketable_otm_quote_count(
    contracts: list[dict],
    spot: float,
    option_type: str,
    min_bid: float = 0.05,
) -> int:
    """Count two-sided executable quotes that are OTM for the requested leg."""

    if spot <= 0 or option_type not in {"put", "call"}:
        return 0
    count = 0
    for contract in contracts:
        strike = _finite_float(contract.get("strike"), -1.0)
        bid = _finite_float(contract.get("bid"), -1.0)
        ask = _finite_float(contract.get("ask"), -1.0)
        is_otm = strike < spot if option_type == "put" else strike > spot
        if is_otm and bid >= min_bid and ask > 0 and ask >= bid:
            count += 1
    return count


def _select_underlying_quote(ticker, info: dict | None = None) -> dict:
    """Select the freshest session-appropriate underlying quote with provenance.

    Explicit session quotes with provider timestamps take precedence. A recent
    intraday bar and ``fast_info.last_price`` are fallbacks; ``previousClose`` is
    used only when no fresher price can be obtained.
    """
    info = info or {}
    market_state = str(info.get("marketState") or "UNKNOWN").upper()
    retrieved_at = datetime.datetime.now(datetime.timezone.utc)

    field_specs = {
        "preMarketPrice": ("preMarketTime", "pre-market"),
        "regularMarketPrice": ("regularMarketTime", "regular"),
        "postMarketPrice": ("postMarketTime", "post-market"),
        "currentPrice": ("regularMarketTime", "regular"),
    }
    explicit_quotes = []
    for source, (time_key, session) in field_specs.items():
        price = _finite_float(info.get(source))
        if price <= 0:
            continue
        timestamp = _quote_timestamp(info.get(time_key))
        explicit_quotes.append(
            {
                "price": price,
                "source": source,
                "session": session,
                "timestamp": timestamp,
            }
        )

    if market_state in {"REGULAR", "OPEN"}:
        preferred_sessions = {"regular"}
        fallback_order = ["regularMarketPrice", "currentPrice"]
    elif market_state in {"PRE", "PREPRE"}:
        preferred_sessions = {"pre-market"}
        fallback_order = ["preMarketPrice", "postMarketPrice", "regularMarketPrice", "currentPrice"]
    elif market_state in {"POST", "POSTPOST"}:
        preferred_sessions = {"post-market"}
        fallback_order = ["postMarketPrice", "regularMarketPrice", "currentPrice", "preMarketPrice"]
    else:
        preferred_sessions = {"pre-market", "regular", "post-market"}
        fallback_order = ["postMarketPrice", "regularMarketPrice", "preMarketPrice", "currentPrice"]

    preferred_quotes = [item for item in explicit_quotes if item["session"] in preferred_sessions]
    timestamped_quotes = [item for item in preferred_quotes if item["timestamp"] is not None]
    if not timestamped_quotes and market_state not in {"REGULAR", "OPEN", "PRE", "PREPRE", "POST", "POSTPOST"}:
        timestamped_quotes = [item for item in explicit_quotes if item["timestamp"] is not None]

    selected = max(timestamped_quotes, key=lambda item: item["timestamp"]) if timestamped_quotes else None
    by_source = {item["source"]: item for item in explicit_quotes}
    untimestamped_explicit = next(
        (by_source[source] for source in fallback_order if source in by_source),
        None,
    )

    if selected is None:
        try:
            intraday = ticker.history(period="1d", interval="1m", prepost=True, auto_adjust=False)
            if intraday is not None and not intraday.empty:
                close = intraday["Close"].dropna()
                if not close.empty:
                    selected = {
                        "price": _finite_float(close.iloc[-1]),
                        "source": "latest_1m_close",
                        "session": "latest trade",
                        "timestamp": _quote_timestamp(close.index[-1]),
                    }
        except Exception as exc:
            logger.debug("Could not fetch an intraday underlying quote: %s", exc)

    if selected is None:
        selected = untimestamped_explicit

    if selected is None or selected["price"] <= 0:
        try:
            fast_price = _finite_float(ticker.fast_info.get("last_price"))
        except Exception:
            fast_price = 0.0
        if fast_price > 0:
            selected = {
                "price": fast_price,
                "source": "fast_info.last_price",
                "session": "provider last",
                "timestamp": None,
            }

    used_previous_close = False
    if selected is None or selected["price"] <= 0:
        previous_close = _finite_float(info.get("previousClose"))
        if previous_close > 0:
            selected = {
                "price": previous_close,
                "source": "previousClose",
                "session": "previous close",
                "timestamp": None,
            }
            used_previous_close = True

    if selected is None:
        selected = {"price": 0.0, "source": "unavailable", "session": "unknown", "timestamp": None}

    timestamp = selected.get("timestamp")
    age_seconds = None
    if timestamp is not None:
        age_seconds = max((retrieved_at - timestamp).total_seconds(), 0.0)

    return {
        "price": selected["price"],
        "source": selected["source"],
        "session": selected["session"],
        "timestamp": timestamp.isoformat() if timestamp is not None else None,
        "retrieved_at": retrieved_at.isoformat(),
        "age_seconds": age_seconds,
        "market_state": market_state,
        "used_previous_close": used_previous_close,
        "regular_market_price": _finite_float(info.get("regularMarketPrice"), None),
        "pre_market_price": _finite_float(info.get("preMarketPrice"), None),
        "post_market_price": _finite_float(info.get("postMarketPrice"), None),
    }


def _quote_result_fields(quote: dict) -> dict:
    warning = None
    if quote.get("used_previous_close"):
        warning = "Only the previous close was available; recommendations may be stale."
    elif quote.get("session") in {"pre-market", "post-market"}:
        warning = (
            "The underlying uses an extended-hours quote while listed option bid/ask quotes may reflect "
            "the most recent option-market session."
        )
    return {
        "spot_source": quote.get("source"),
        "spot_session": quote.get("session"),
        "spot_as_of": quote.get("timestamp"),
        "spot_retrieved_at": quote.get("retrieved_at"),
        "spot_age_seconds": quote.get("age_seconds"),
        "market_state": quote.get("market_state"),
        "spot_used_previous_close": bool(quote.get("used_previous_close")),
        "regular_market_price": quote.get("regular_market_price"),
        "pre_market_price": quote.get("pre_market_price"),
        "post_market_price": quote.get("post_market_price"),
        "spot_warning": warning,
    }


def _snapshot_result_fields(snapshot: dict, requested_basis: str) -> dict:
    captured_at = _quote_timestamp(snapshot.get("captured_at"))
    retrieved_at = datetime.datetime.now(datetime.timezone.utc)
    age_seconds = (
        max((retrieved_at - captured_at).total_seconds(), 0.0)
        if captured_at is not None
        else None
    )
    underlying_quote = snapshot.get("underlying_quote") or {}
    market_date = str(snapshot.get("market_date") or "unknown date")
    captured_text = captured_at.isoformat() if captured_at is not None else snapshot.get("captured_at")
    return {
        "quote_basis_requested": requested_basis,
        "quote_basis_used": QUOTE_BASIS_PREVIOUS_SESSION,
        "is_snapshot": True,
        "pricing_status": "planning_snapshot",
        "spot_source": "saved_regular_session_snapshot",
        "spot_session": "regular-session snapshot",
        "spot_as_of": underlying_quote.get("timestamp") or captured_text,
        "spot_retrieved_at": retrieved_at.isoformat(),
        "spot_age_seconds": age_seconds,
        "market_state": snapshot.get("market_state") or "REGULAR",
        "spot_used_previous_close": False,
        "regular_market_price": _finite_float(snapshot.get("spot"), None),
        "pre_market_price": None,
        "post_market_price": None,
        "snapshot_captured_at": captured_text,
        "snapshot_market_date": market_date,
        "snapshot_age_seconds": age_seconds,
        "snapshot_marketable_puts": int(_finite_float(snapshot.get("marketable_puts"))),
        "snapshot_marketable_calls": int(_finite_float(snapshot.get("marketable_calls"))),
        "snapshot_marketable_otm_puts": int(_finite_float(snapshot.get("marketable_otm_puts"))),
        "snapshot_marketable_otm_calls": int(_finite_float(snapshot.get("marketable_otm_calls"))),
        "snapshot_provider": snapshot.get("provider") or "Yahoo Finance via yfinance",
        "spot_warning": (
            f"Planning snapshot from the {market_date} regular session. Bid/ask prices are historical and "
            "not currently executable; the ranking shows what the model would have suggested from the saved data."
        ),
    }


def calculate_bs_delta(S, K, T, r, sigma, option_type="call"):
    """
    Calculates the option Delta using the Black-Scholes-Merton formula.
    """
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0:
        # Expiration day behavior
        if option_type == "call":
            return 1.0 if S >= K else 0.0
        else:
            return -1.0 if S < K else 0.0
            
    if sigma <= 0:
        sigma = 0.0001
        
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        if option_type == "call":
            return float(stats.norm.cdf(d1))
        else:
            return float(stats.norm.cdf(d1) - 1.0)
    except Exception as e:
        logger.error(f"Error calculating BS delta: {e}")
        return 0.0

def calculate_bs_price(S, K, T, r, sigma, option_type="call"):
    """
    Calculates the theoretical Black-Scholes option price.
    """
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
            
    if sigma <= 0:
        sigma = 0.0001
        
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == "call":
            return float(S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2))
        else:
            return float(K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1))
    except Exception as e:
        logger.error(f"Error calculating BS price: {e}")
        return 0.0

class OptionsAnalyzer:
    def __init__(self, data_provider=None, *, snapshot_store=None, snapshot_dir=None):
        self.data_provider = data_provider
        self.snapshot_store = snapshot_store or OptionSnapshotStore(snapshot_dir)

    def get_historical_volatility(self, ticker_symbol: str, days: int = 30, ticker=None) -> float:
        """
        Calculates annualized historical volatility over the past N days.
        """
        ticker_symbol = ticker_symbol.strip().upper()
        try:
            ticker = ticker or yf.Ticker(ticker_symbol)
            hist = ticker.history(period="3mo")
            if not hist.empty and len(hist) >= days:
                close = hist["Close"].tail(days)
                returns = close.pct_change().dropna()
                daily_vol = returns.std()
                annualized_vol = daily_vol * math.sqrt(252)
                return max(float(annualized_vol), 0.05) # floor at 5%
        except Exception as e:
            logger.warning(f"Error calculating historical volatility for {ticker_symbol}: {e}")
        return 0.20 # default 20% fallback

    def get_options_data(
        self,
        ticker_symbol: str,
        expiration_date: str,
        r_rate: float = 0.0425,
        *,
        ticker=None,
        spot: float | None = None,
        historical_volatility: float | None = None,
        dividend_yield: float | None = None,
        as_of_date: datetime.date | None = None,
    ) -> dict:
        """
        Fetches option chain for a given expiration date, calculates Delta and filters data.
        """
        ticker_symbol = ticker_symbol.strip().upper()
        ticker = ticker or yf.Ticker(ticker_symbol)
        info = {}
        if spot is None or dividend_yield is None:
            try:
                info = ticker.info or {}
            except Exception as exc:
                logger.warning("Could not fetch option underlying metadata for %s: %s", ticker_symbol, exc)
        quote = None
        S = _finite_float(spot)
        if S <= 0:
            quote = _select_underlying_quote(ticker, info)
            S = _finite_float(quote.get("price"))
        if S <= 0:
            return {
                "calls": [],
                "puts": [],
                "S": 0.0,
                "dte": 0,
                "error": f"No usable underlying price was available for {ticker_symbol}.",
            }

        q_rate = _finite_float(dividend_yield, -1.0)
        if q_rate < 0:
            q_rate = _finite_float(info.get("dividendYield"), 0.0)
        if not 0.0 <= q_rate <= 0.25:
            q_rate = 0.0
        
        # Calculate DTE and T (time to maturity in years)
        try:
            exp_dt = datetime.datetime.strptime(expiration_date, "%Y-%m-%d").date()
            today = as_of_date or _market_date()
            dte = max((exp_dt - today).days, 0)
        except Exception:
            dte = 30
            
        T = dte / 365.25
        
        # Fetch historical volatility as a fallback for implied volatility
        hist_vol = _finite_float(historical_volatility, -1.0)
        if hist_vol <= 0:
            hist_vol = self.get_historical_volatility(ticker_symbol, ticker=ticker)
        
        # Fetch option chain
        try:
            opt_chain = ticker.option_chain(expiration_date)
            calls = opt_chain.calls.copy()
            puts = opt_chain.puts.copy()
        except Exception as e:
            logger.error(f"Failed to fetch option chain for {ticker_symbol} on {expiration_date}: {e}")
            return {"calls": [], "puts": [], "S": S, "dte": dte, "error": str(e)}

        def process_df(df, option_type):
            processed = []
            for _, row in df.iterrows():
                K = _finite_float(row.get("strike"))
                last_price = _finite_float(row.get("lastPrice"))
                bid = _finite_float(row.get("bid"))
                ask = _finite_float(row.get("ask"))
                iv = _finite_float(row.get("impliedVolatility"), 0.0)
                vol = _finite_float(row.get("volume"))
                oi = _finite_float(row.get("openInterest"))
                
                # Determine premium midpoint
                premium = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else last_price
                if premium <= 0:
                    premium = last_price if last_price > 0 else 0.01
                    
                # Fix distorted IV (e.g. market closed or no bid/ask)
                iv_used = iv
                if iv < 0.02 or (bid == 0.0 and ask == 0.0):
                    iv_used = hist_vol
                    
                try:
                    greeks = black_scholes_metrics(
                        S, K, T, r_rate, iv_used, q_rate, option_type
                    )
                except ValueError:
                    greeks = {
                        "delta": calculate_bs_delta(S, K, T, r_rate, iv_used, option_type),
                        "gamma": 0.0,
                        "theta_per_day": 0.0,
                        "vega_per_vol_point": 0.0,
                        "probability_itm": 0.0,
                    }
                
                processed.append({
                    "contractSymbol": row["contractSymbol"],
                    "expiration": expiration_date,
                    "dte": dte,
                    "strike": K,
                    "lastPrice": last_price,
                    "bid": bid,
                    "ask": ask,
                    "volume": int(vol),
                    "openInterest": int(oi),
                    "impliedVolatility": iv,
                    "ivUsed": iv_used,
                    "delta": greeks["delta"],
                    "gamma": greeks["gamma"],
                    "theta": greeks["theta_per_day"],
                    "vega": greeks["vega_per_vol_point"],
                    "probabilityITM": greeks["probability_itm"],
                    "premium": premium,
                    "inTheMoney": bool(row.get("inTheMoney", False)),
                    "lastTradeDate": row.get("lastTradeDate"),
                })
            # Sort by strike price
            processed = sorted(processed, key=lambda x: x["strike"])
            return processed

        processed_calls = process_df(calls, "call")
        processed_puts = process_df(puts, "put")
        
        return {
            "calls": processed_calls,
            "puts": processed_puts,
            "S": S,
            "dte": dte,
            "T": T,
            "hist_vol": hist_vol,
            "dividend_yield": q_rate,
            **({"underlying_quote": quote} if quote is not None else {}),
        }

    def get_expiration_dates(self, ticker_symbol: str) -> list[str]:
        """Return available future expirations in chronological order."""
        ticker_symbol = ticker_symbol.strip().upper()
        try:
            expirations = list(yf.Ticker(ticker_symbol).options or [])
        except Exception as exc:
            logger.warning("Could not fetch expirations for %s: %s", ticker_symbol, exc)
            return []
        today = _market_date()
        valid = []
        for expiration in expirations:
            try:
                expiration_date = datetime.datetime.strptime(expiration, "%Y-%m-%d").date()
            except (TypeError, ValueError):
                continue
            if expiration_date >= today:
                valid.append(expiration)
        return sorted(set(valid))

    def _local_iv_context(self, ticker_symbol: str, current_iv: float) -> dict:
        """Persist one daily ATM-IV snapshot when no public volatility proxy exists."""
        today = _market_date()
        cutoff = today - datetime.timedelta(days=370)
        payload = {}
        try:
            if _IV_HISTORY_FILE.exists():
                payload = json.loads(_IV_HISTORY_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Could not read local IV history: %s", exc)
            payload = {}

        symbol_history = payload.get(ticker_symbol, {})
        if not isinstance(symbol_history, dict):
            symbol_history = {}
        cleaned = {}
        for date_text, value in symbol_history.items():
            try:
                observation_date = datetime.date.fromisoformat(date_text)
            except (TypeError, ValueError):
                continue
            numeric = _finite_float(value, -1.0)
            if observation_date >= cutoff and numeric > 0:
                cleaned[date_text] = numeric
        if current_iv > 0:
            cleaned[today.isoformat()] = current_iv
        payload[ticker_symbol] = cleaned
        try:
            _IV_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            _IV_HISTORY_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not update local IV history: %s", exc)

        values = list(cleaned.values())
        enough_history = len(values) >= 20
        return {
            "current_iv": current_iv,
            "iv_rank": calculate_iv_rank(current_iv, values) if enough_history else None,
            "iv_percentile": calculate_iv_percentile(current_iv, values) if enough_history else None,
            "history_low": min(values) if values else None,
            "history_high": max(values) if values else None,
            "sample_count": len(values),
            "source": f"Local ATM IV history ({len(values)}/20 minimum daily snapshots)",
            "is_proxy": False,
        }

    def get_iv_context(self, ticker_symbol: str, current_atm_iv: float) -> dict:
        """Return 52-week IV-rank context, using a named index proxy when available."""
        ticker_symbol = ticker_symbol.strip().upper()
        proxy = _VOLATILITY_PROXIES.get(ticker_symbol)
        if proxy:
            proxy_symbol, proxy_name = proxy
            try:
                history = yf.Ticker(proxy_symbol).history(period="1y", auto_adjust=False)
                values = [
                    _finite_float(value) / 100.0
                    for value in history.get("Close", pd.Series(dtype=float)).dropna().tolist()
                    if _finite_float(value) > 0
                ]
                if len(values) >= 20:
                    current = values[-1]
                    return {
                        "current_iv": current,
                        "iv_rank": calculate_iv_rank(current, values),
                        "iv_percentile": calculate_iv_percentile(current, values),
                        "history_low": min(values),
                        "history_high": max(values),
                        "sample_count": len(values),
                        "source": f"{proxy_symbol} 52-week {proxy_name} volatility proxy",
                        "is_proxy": True,
                    }
            except Exception as exc:
                logger.warning("Volatility proxy %s failed: %s", proxy_symbol, exc)
        return self._local_iv_context(ticker_symbol, current_atm_iv)

    @staticmethod
    def _short_put_empty_result(symbol: str, prefs: ShortPutPreferences, requested_basis: str) -> dict:
        return {
            "strategy": "cash_secured_short_put",
            "symbol": symbol,
            "recommendations": [],
            "input_count": 0,
            "eligible_count": 0,
            "eligible_before_limit": 0,
            "rejected_counts": {},
            "quote_basis_requested": requested_basis,
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

    def _recommend_short_puts_from_snapshot(
        self,
        symbol: str,
        snapshot: dict,
        prefs: ShortPutPreferences,
        fallback_rate: float,
        requested_basis: str,
        include_calls: bool,
    ) -> dict:
        empty_result = self._short_put_empty_result(symbol, prefs, requested_basis)
        spot = _finite_float(snapshot.get("spot"))
        historical_volatility = _finite_float(snapshot.get("historical_volatility"), 0.20)
        dividend_yield = _finite_float(snapshot.get("dividend_yield"), 0.0)
        snapshot_rate = _finite_float(snapshot.get("risk_free_rate"), fallback_rate)
        try:
            snapshot_date = datetime.date.fromisoformat(str(snapshot.get("market_date")))
        except (TypeError, ValueError):
            return {**empty_result, "error": "The saved option snapshot has an invalid market date."}
        if spot <= 0 or historical_volatility <= 0:
            return {**empty_result, "error": "The saved option snapshot has invalid market inputs."}

        selected_expirations: list[tuple[str, int]] = []
        chains = snapshot.get("chains") or {}
        # A provider may list more expirations than were deliberately captured.
        # Replay only the aligned chains that actually exist in this snapshot.
        expiration_universe = snapshot.get("captured_expirations") or list(chains)
        for expiration in expiration_universe:
            try:
                expiry = datetime.date.fromisoformat(str(expiration))
            except (TypeError, ValueError):
                continue
            dte = (expiry - snapshot_date).days
            if prefs.min_dte <= dte <= prefs.max_dte:
                selected_expirations.append((str(expiration), dte))

        selected_expirations.sort(key=lambda item: (abs(item[1] - prefs.target_dte), item[1]))
        selected_expirations = selected_expirations[:12]
        selected_expirations.sort(key=lambda item: item[1])
        snapshot_fields = _snapshot_result_fields(snapshot, requested_basis)
        if not selected_expirations:
            return {
                **empty_result,
                "spot": spot,
                **snapshot_fields,
                "historical_volatility": historical_volatility,
                "error": (
                    f"The latest saved regular-session snapshot does not contain expirations inside "
                    f"{prefs.min_dte}-{prefs.max_dte} DTE. Run a live scan with this window during the regular session."
                ),
            }
        missing_expirations = [expiration for expiration, _ in selected_expirations if expiration not in chains]
        if missing_expirations:
            return {
                **empty_result,
                "spot": spot,
                **snapshot_fields,
                "historical_volatility": historical_volatility,
                "error": (
                    "The latest snapshot does not contain the complete chain coverage required by this DTE window "
                    f"and mode ({', '.join(missing_expirations)} missing). Run the same scan once during the regular session."
                ),
            }

        contracts: list[dict] = []
        call_contracts: list[dict] = []
        for expiration, dte in selected_expirations:
            chain = chains.get(expiration) or {}
            contracts.extend({**item, "expiration": expiration, "dte": dte} for item in chain.get("puts", []))
            call_contracts.extend(
                {**item, "expiration": expiration, "dte": dte} for item in chain.get("calls", [])
            )
        if not contracts:
            return {
                **empty_result,
                "spot": spot,
                **snapshot_fields,
                "historical_volatility": historical_volatility,
                "error": "The saved snapshot contains no put contracts for the selected DTE window.",
            }

        marketable_puts = _marketable_quote_count(contracts, prefs.min_bid)
        marketable_otm_puts = _marketable_otm_quote_count(contracts, spot, "put", prefs.min_bid)
        marketable_calls = _marketable_quote_count(call_contracts)
        marketable_otm_calls = _marketable_otm_quote_count(call_contracts, spot, "call")
        data_readiness = "ready" if marketable_otm_puts > 0 else "no_marketable_otm_puts"

        atm_contract = min(
            contracts,
            key=lambda item: (
                abs(_finite_float(item.get("dte")) - 30.0),
                abs(_finite_float(item.get("strike")) - spot),
            ),
        )
        current_atm_iv = _finite_float(
            atm_contract.get("impliedVolatility") or atm_contract.get("ivUsed"),
            historical_volatility,
        )
        iv_context = snapshot.get("iv_context") or {
            "current_iv": current_atm_iv,
            "iv_rank": None,
            "iv_percentile": None,
            "sample_count": 0,
            "source": "IV context was not preserved in this snapshot",
            "is_proxy": False,
        }
        scored = score_short_put_chain(
            contracts,
            spot,
            historical_volatility,
            prefs,
            rate=snapshot_rate,
            dividend_yield=dividend_yield,
            iv_context=iv_context,
        )
        return {
            **scored,
            "strategy": "cash_secured_short_put",
            "symbol": symbol,
            "spot": spot,
            **snapshot_fields,
            "historical_volatility": historical_volatility,
            "dividend_yield": dividend_yield,
            "risk_free_rate": snapshot_rate,
            "iv_context": iv_context,
            "scanned_expirations": [item[0] for item in selected_expirations],
            "failed_expirations": snapshot.get("failed_expirations") or [],
            "as_of": snapshot.get("captured_at"),
            "market_date": snapshot_date.isoformat(),
            "marketable_put_count": marketable_puts,
            "marketable_otm_put_count": marketable_otm_puts,
            "marketable_call_count": marketable_calls,
            "marketable_otm_call_count": marketable_otm_calls,
            "data_readiness": data_readiness,
            "assignment_note": (
                "The percentage is a Black-Scholes risk-neutral estimate of finishing in the money, "
                "not a guaranteed assignment probability. American-style puts can be assigned early."
            ),
            **({"_call_contracts": call_contracts} if include_calls else {}),
        }

    def recommend_short_puts(
        self,
        ticker_symbol: str,
        preferences: ShortPutPreferences | None = None,
        r_rate: float = 0.0425,
        *,
        include_calls: bool = False,
        quote_basis: str = QUOTE_BASIS_LIVE,
        expiration_limit: int = 12,
    ) -> dict:
        """Scan and rank puts from either a live chain or an aligned saved snapshot."""
        symbol = ticker_symbol.strip().upper()
        prefs = preferences or ShortPutPreferences.for_profile("lowest_risk")
        try:
            requested_basis = _normalize_quote_basis(quote_basis)
        except ValueError as exc:
            requested_basis = str(quote_basis)
            return {
                **self._short_put_empty_result(symbol, prefs, requested_basis),
                "error": str(exc),
            }
        empty_result = self._short_put_empty_result(symbol, prefs, requested_basis)

        if requested_basis == QUOTE_BASIS_PREVIOUS_SESSION:
            snapshot = self.snapshot_store.load_latest(symbol)
            if snapshot is None:
                return {
                    **empty_result,
                    "quote_basis_used": QUOTE_BASIS_PREVIOUS_SESSION,
                    "is_snapshot": True,
                    "error": (
                        f"No saved regular-session option snapshot is available for {symbol}. "
                        "Run one live scan during the regular session to create it automatically."
                    ),
                }
            return self._recommend_short_puts_from_snapshot(
                symbol, snapshot, prefs, r_rate, requested_basis, include_calls
            )

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
        except Exception as exc:
            return {**empty_result, "error": f"Could not load options for {symbol}: {exc}"}

        market_state = str(info.get("marketState") or "UNKNOWN").upper()
        off_session_states = {"PRE", "PREPRE", "POST", "POSTPOST", "CLOSED", "CLOSE"}
        if requested_basis == QUOTE_BASIS_AUTO and market_state in off_session_states:
            snapshot = self.snapshot_store.load_latest(symbol)
            if snapshot is None:
                return {
                    **empty_result,
                    "quote_basis_used": QUOTE_BASIS_PREVIOUS_SESSION,
                    "is_snapshot": True,
                    "market_state": market_state,
                    "error": (
                        f"{symbol} is outside the regular session and no saved option snapshot exists yet. "
                        "Run one live scan during the regular session; later off-hours scans will replay it automatically."
                    ),
                }
            return self._recommend_short_puts_from_snapshot(
                symbol, snapshot, prefs, r_rate, requested_basis, include_calls
            )
        if requested_basis == QUOTE_BASIS_LIVE and market_state in off_session_states:
            return {
                **empty_result,
                "quote_basis_used": QUOTE_BASIS_LIVE,
                "is_snapshot": False,
                "market_state": market_state,
                "error": (
                    f"Current-session option quotes are unavailable while the provider market state is {market_state}. "
                    "Choose Auto or Latest saved regular session instead."
                ),
            }

        try:
            expirations = list(ticker.options or [])
        except Exception as exc:
            return {**empty_result, "error": f"Could not load option expirations for {symbol}: {exc}"}

        spot_quote = _select_underlying_quote(ticker, info)
        spot = _finite_float(spot_quote.get("price"))
        if spot <= 0:
            return {**empty_result, "error": f"No usable underlying price was available for {symbol}."}

        historical_volatility = self.get_historical_volatility(symbol, ticker=ticker)
        dividend_yield = _finite_float(info.get("dividendYield"), 0.0)
        if not 0.0 <= dividend_yield <= 0.25:
            dividend_yield = 0.0

        today = _market_date()
        selected_expirations: list[tuple[str, int]] = []
        for expiration in expirations:
            try:
                expiry = datetime.datetime.strptime(expiration, "%Y-%m-%d").date()
            except (TypeError, ValueError):
                continue
            dte = (expiry - today).days
            if prefs.min_dte <= dte <= prefs.max_dte:
                selected_expirations.append((expiration, dte))

        live_fields = {
            **_quote_result_fields(spot_quote),
            "quote_basis_requested": requested_basis,
            "quote_basis_used": QUOTE_BASIS_LIVE,
            "is_snapshot": False,
            "pricing_status": "live_provider",
        }
        if not selected_expirations:
            return {
                **empty_result,
                "spot": spot,
                **live_fields,
                "historical_volatility": historical_volatility,
                "error": f"No listed expirations fall inside {prefs.min_dte}-{prefs.max_dte} DTE.",
            }

        try:
            bounded_expiration_limit = max(1, min(int(expiration_limit), 24))
        except (TypeError, ValueError):
            bounded_expiration_limit = 12
        selected_expirations.sort(key=lambda item: (abs(item[1] - prefs.target_dte), item[1]))
        selected_expirations = selected_expirations[:bounded_expiration_limit]
        selected_expirations.sort(key=lambda item: item[1])
        contracts: list[dict] = []
        all_call_contracts: list[dict] = []
        chains: dict[str, dict[str, list[dict]]] = {}
        failed_expirations: list[str] = []
        capture_started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for expiration, _ in selected_expirations:
            data = self.get_options_data(
                symbol,
                expiration,
                r_rate,
                ticker=ticker,
                spot=spot,
                historical_volatility=historical_volatility,
                dividend_yield=dividend_yield,
                as_of_date=today,
            )
            if data.get("error"):
                failed_expirations.append(expiration)
                continue
            puts = data.get("puts", [])
            calls = data.get("calls", [])
            chains[expiration] = {"puts": puts, "calls": calls}
            contracts.extend(puts)
            all_call_contracts.extend(calls)

        if not contracts:
            return {
                **empty_result,
                "spot": spot,
                **live_fields,
                "historical_volatility": historical_volatility,
                "error": "No usable put quotes were returned for the selected DTE window.",
                "failed_expirations": failed_expirations,
            }

        atm_contract = min(
            contracts,
            key=lambda item: (
                abs(_finite_float(item.get("dte")) - 30.0),
                abs(_finite_float(item.get("strike")) - spot),
            ),
        )
        current_atm_iv = _finite_float(
            atm_contract.get("impliedVolatility") or atm_contract.get("ivUsed"),
            historical_volatility,
        )
        iv_context = self.get_iv_context(symbol, current_atm_iv)
        captured_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        marketable_puts = _marketable_quote_count(contracts, prefs.min_bid)
        marketable_calls = _marketable_quote_count(all_call_contracts)
        marketable_otm_puts = _marketable_otm_quote_count(contracts, spot, "put", prefs.min_bid)
        marketable_otm_calls = _marketable_otm_quote_count(all_call_contracts, spot, "call")
        data_readiness = "ready" if marketable_otm_puts > 0 else "no_marketable_otm_puts"
        snapshot_saved = False
        snapshot_save_error = None
        if not _regular_session_state(market_state) or spot_quote.get("session") != "regular":
            snapshot_save_status = "outside_regular_session"
        elif failed_expirations:
            snapshot_save_status = "incomplete_expiration_fetch"
        elif marketable_otm_puts <= 0 or not chains:
            snapshot_save_status = "no_marketable_otm_puts"
        else:
            snapshot_save_status = "ready"
            snapshot = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "symbol": symbol,
                "market_date": today.isoformat(),
                "capture_started_at": capture_started_at,
                "captured_at": captured_at,
                "session": "regular",
                "market_state": market_state,
                "provider": "Yahoo Finance via yfinance",
                "spot": spot,
                "underlying_quote": spot_quote,
                "historical_volatility": historical_volatility,
                "dividend_yield": dividend_yield,
                "risk_free_rate": r_rate,
                "iv_context": iv_context,
                "marketable_puts": marketable_puts,
                "marketable_calls": marketable_calls,
                "marketable_otm_puts": marketable_otm_puts,
                "marketable_otm_calls": marketable_otm_calls,
                "failed_expirations": failed_expirations,
                "listed_expirations": expirations,
                "captured_expirations": list(chains),
                "chains": compact_snapshot_chains(chains, spot),
            }
            try:
                self.snapshot_store.save(snapshot)
                snapshot_saved = True
                snapshot_save_status = "saved"
            except (OSError, ValueError, TypeError) as exc:
                snapshot_save_status = "save_failed"
                snapshot_save_error = str(exc)
                logger.warning("Could not save option snapshot for %s: %s", symbol, exc)

        scored = score_short_put_chain(
            contracts,
            spot,
            historical_volatility,
            prefs,
            rate=r_rate,
            dividend_yield=dividend_yield,
            iv_context=iv_context,
        )
        return {
            **scored,
            "strategy": "cash_secured_short_put",
            "symbol": symbol,
            "spot": spot,
            **live_fields,
            "historical_volatility": historical_volatility,
            "dividend_yield": dividend_yield,
            "risk_free_rate": r_rate,
            "iv_context": iv_context,
            "scanned_expirations": [item[0] for item in selected_expirations],
            "failed_expirations": failed_expirations,
            "as_of": spot_quote.get("timestamp") or spot_quote.get("retrieved_at"),
            "market_date": today.isoformat(),
            "snapshot_saved": snapshot_saved,
            "snapshot_captured_at": captured_at if snapshot_saved else None,
            "snapshot_marketable_puts": marketable_puts,
            "snapshot_marketable_calls": marketable_calls,
            "snapshot_marketable_otm_puts": marketable_otm_puts,
            "snapshot_marketable_otm_calls": marketable_otm_calls,
            "snapshot_save_error": snapshot_save_error,
            "snapshot_save_status": snapshot_save_status,
            "marketable_put_count": marketable_puts,
            "marketable_otm_put_count": marketable_otm_puts,
            "marketable_call_count": marketable_calls,
            "marketable_otm_call_count": marketable_otm_calls,
            "data_readiness": data_readiness,
            "assignment_note": (
                "The percentage is a Black-Scholes risk-neutral estimate of finishing in the money, "
                "not a guaranteed assignment probability. American-style puts can be assigned early."
            ),
            **({"_call_contracts": all_call_contracts} if include_calls else {}),
        }

    def recommend_premium_funded_bullish_combo(
        self,
        ticker_symbol: str,
        put_preferences: ShortPutPreferences | None = None,
        combo_preferences: BullishComboPreferences | None = None,
        r_rate: float = 0.0425,
        *,
        quote_basis: str = QUOTE_BASIS_LIVE,
    ) -> dict:
        """Jointly rank OTM short puts and premium-funded OTM long calls."""
        symbol = ticker_symbol.strip().upper()
        put_prefs = put_preferences or ShortPutPreferences.for_profile(
            "balanced", max_assignment_probability=0.25
        )
        combo_prefs = combo_preferences or BullishComboPreferences.for_profile("balanced")
        expanded_put_preferences = put_prefs.with_overrides(limit=5_000)
        short_put_result = self.recommend_short_puts(
            symbol,
            expanded_put_preferences,
            r_rate,
            include_calls=True,
            quote_basis=quote_basis,
        )
        if short_put_result.get("error"):
            return {
                **short_put_result,
                "strategy": "premium_funded_bullish_risk_reversal",
                "recommendations": [],
            }

        calls = short_put_result.pop("_call_contracts", [])
        put_candidates = short_put_result.get("recommendations", [])
        if not calls:
            return {
                **short_put_result,
                "strategy": "premium_funded_bullish_risk_reversal",
                "recommendations": [],
                "error": "No usable call quotes were returned for the selected DTE window.",
            }

        effective_rate = _finite_float(short_put_result.get("risk_free_rate"), r_rate)
        if int(_finite_float(short_put_result.get("marketable_otm_put_count"))) <= 0:
            combo_data_readiness = "no_marketable_otm_puts"
        elif int(_finite_float(short_put_result.get("marketable_otm_call_count"))) <= 0:
            combo_data_readiness = "no_marketable_otm_calls"
        else:
            combo_data_readiness = "ready"
        paired = score_premium_funded_bullish_pairs(
            put_candidates,
            calls,
            short_put_result["spot"],
            short_put_result["historical_volatility"],
            combo_prefs,
            rate=effective_rate,
            dividend_yield=short_put_result.get("dividend_yield", 0.0),
        )
        return {
            **paired,
            "strategy": "premium_funded_bullish_risk_reversal",
            "symbol": symbol,
            "spot": short_put_result["spot"],
            **{
                key: short_put_result.get(key)
                for key in (
                    "spot_source",
                    "spot_session",
                    "spot_as_of",
                    "spot_retrieved_at",
                    "spot_age_seconds",
                    "market_state",
                    "spot_used_previous_close",
                    "regular_market_price",
                    "pre_market_price",
                    "post_market_price",
                    "spot_warning",
                    "market_date",
                    "quote_basis_requested",
                    "quote_basis_used",
                    "is_snapshot",
                    "pricing_status",
                    "snapshot_saved",
                    "snapshot_captured_at",
                    "snapshot_market_date",
                    "snapshot_age_seconds",
                    "snapshot_marketable_puts",
                    "snapshot_marketable_calls",
                    "snapshot_marketable_otm_puts",
                    "snapshot_marketable_otm_calls",
                    "snapshot_provider",
                    "snapshot_save_error",
                    "snapshot_save_status",
                    "marketable_put_count",
                    "marketable_otm_put_count",
                    "marketable_call_count",
                    "marketable_otm_call_count",
                )
            },
            "historical_volatility": short_put_result["historical_volatility"],
            "dividend_yield": short_put_result.get("dividend_yield", 0.0),
            "risk_free_rate": effective_rate,
            "data_readiness": combo_data_readiness,
            "iv_context": short_put_result.get("iv_context", {}),
            "scanned_expirations": short_put_result.get("scanned_expirations", []),
            "failed_expirations": short_put_result.get("failed_expirations", []),
            "as_of": short_put_result.get("as_of"),
            "short_put_input_count": short_put_result.get("input_count", 0),
            "eligible_short_put_count": short_put_result.get("eligible_before_limit", 0),
            "short_put_rejected_counts": short_put_result.get("rejected_counts", {}),
            "short_put_constraints": short_put_result.get("constraints", {}),
            "combo_constraints": paired.get("constraints", {}),
            "assignment_note": short_put_result.get("assignment_note", ""),
            "strategy_note": (
                "This is a bullish risk reversal: an OTM long call and an OTM short put with "
                "the same expiration. Upside is unlimited, but the short put retains substantial "
                "downside and early-assignment risk."
            ),
        }

    def recommend_strategies(self, ticker_symbol: str, expiration_date: str, strategy: str, r_rate: float = 0.0425, risk_profile: str = "moderate") -> list:
        """
        Main interface to fetch options data and return strategy recommendations.
        """
        data = self.get_options_data(ticker_symbol, expiration_date, r_rate)
        if "error" in data:
            return []
            
        S = data["S"]
        T = data["T"]
        dte = data["dte"]
        calls = data["calls"]
        puts = data["puts"]
        
        if not calls and not puts:
            return []
            
        if strategy == "short_put":
            return self._recommend_short_put(puts, S, T, dte, risk_profile)
        elif strategy == "covered_call":
            return self._recommend_covered_call(calls, S, T, dte, risk_profile)
        elif strategy == "long_call":
            return self._recommend_long_call(calls, S, T, dte, risk_profile)
        elif strategy == "long_put":
            return self._recommend_long_put(puts, S, T, dte, risk_profile)
        elif strategy.startswith("spread_"):
            return self._recommend_spreads(calls, puts, S, T, dte, strategy, risk_profile)
        elif strategy == "straddle":
            return self._recommend_straddle(calls, puts, S, T, dte, risk_profile)
            
        return []

    def _recommend_short_put(self, puts, S, T, dte, risk_profile):
        """
        Recommends Short Put contracts.
        """
        if risk_profile == "safest":
            min_delta, max_delta = -0.16, -0.08
        elif risk_profile == "aggressive":
            min_delta, max_delta = -0.50, -0.36
        else: # moderate
            min_delta, max_delta = -0.35, -0.20
            
        matches = [p for p in puts if min_delta <= p["delta"] <= max_delta]
        if not matches:
            target = -0.12 if risk_profile == "safest" else -0.45 if risk_profile == "aggressive" else -0.28
            sorted_puts = sorted(puts, key=lambda x: abs(x["delta"] - target))
            matches = sorted_puts[:3]
            
        recs = []
        for p in matches:
            prem = p["premium"]
            strike = p["strike"]
            prob_worthless = 1.0 - abs(p["delta"])
            
            # Annualized Yield = (Premium / Strike) / (Days/365)
            ann_yield = (prem / strike) / T if T > 0 else 0.0
            
            recs.append({
                "strike": strike,
                "premium": prem,
                "bid": p["bid"],
                "ask": p["ask"],
                "delta": p["delta"],
                "volume": p["volume"],
                "openInterest": p["openInterest"],
                "prob_worthless": prob_worthless,
                "annualized_yield": ann_yield,
                "break_even": strike - prem,
                "max_profit": prem * 100,
                "max_loss": (strike - prem) * 100,
                "description_en": f"Sell ${strike:.1f} Put to collect ${prem:.2f} premium. Break-even is ${strike - prem:.2f}.",
                "description_zh": f"卖出 ${strike:.1f} 的看跌期权 (Put) 收取 ${prem:.2f} 权利金。保本价为 ${strike - prem:.2f}。"
            })
        return sorted(recs, key=lambda x: abs(x["delta"]))

    def _recommend_covered_call(self, calls, S, T, dte, risk_profile):
        """
        Recommends Short Call for Covered Call write.
        """
        if risk_profile == "safest":
            min_delta, max_delta = 0.08, 0.16
        elif risk_profile == "aggressive":
            min_delta, max_delta = 0.36, 0.50
        else: # moderate
            min_delta, max_delta = 0.20, 0.35
            
        matches = [c for c in calls if min_delta <= c["delta"] <= max_delta]
        if not matches:
            target = 0.12 if risk_profile == "safest" else 0.45 if risk_profile == "aggressive" else 0.28
            sorted_calls = sorted(calls, key=lambda x: abs(x["delta"] - target))
            matches = sorted_calls[:3]
            
        recs = []
        for c in matches:
            prem = c["premium"]
            strike = c["strike"]
            prob_worthless = 1.0 - c["delta"]
            
            # Annualized Yield = (Premium / Current Price) / (Days/365)
            ann_yield = (prem / S) / T if T > 0 else 0.0
            
            recs.append({
                "strike": strike,
                "premium": prem,
                "bid": c["bid"],
                "ask": c["ask"],
                "delta": c["delta"],
                "volume": c["volume"],
                "openInterest": c["openInterest"],
                "prob_worthless": prob_worthless,
                "annualized_yield": ann_yield,
                "break_even": strike + prem,
                "max_profit": (strike - S + prem) * 100,
                "max_loss": (S - prem) * 100,
                "description_en": f"Sell ${strike:.1f} Call to generate ${prem:.2f} income. If exercised, shares called away at ${strike:.1f}.",
                "description_zh": f"卖出 ${strike:.1f} 的看涨期权 (Call) 产生 ${prem:.2f} 收益。若被行权，股票将在 ${strike:.1f} 被呼出。"
            })
        return sorted(recs, key=lambda x: abs(x["delta"]))

    def _recommend_long_call(self, calls, S, T, dte, risk_profile):
        """
        Recommends Long Call for speculative buying.
        """
        if risk_profile == "safest":
            min_delta, max_delta = 0.65, 0.85
        elif risk_profile == "aggressive":
            min_delta, max_delta = 0.20, 0.35
        else: # moderate
            min_delta, max_delta = 0.40, 0.60
            
        matches = [c for c in calls if min_delta <= c["delta"] <= max_delta]
        if not matches:
            target = 0.75 if risk_profile == "safest" else 0.28 if risk_profile == "aggressive" else 0.50
            sorted_calls = sorted(calls, key=lambda x: abs(x["delta"] - target))
            matches = sorted_calls[:3]
            
        recs = []
        for c in matches:
            prem = c["premium"]
            strike = c["strike"]
            prob_profit = c["delta"]
            
            recs.append({
                "strike": strike,
                "premium": prem,
                "bid": c["bid"],
                "ask": c["ask"],
                "delta": c["delta"],
                "volume": c["volume"],
                "openInterest": c["openInterest"],
                "prob_worthless": 1.0 - prob_profit,
                "break_even": strike + prem,
                "max_profit": float('inf'),
                "max_loss": prem * 100,
                "description_en": f"Buy ${strike:.1f} Call for ${prem:.2f}. Break-even is ${strike + prem:.2f}.",
                "description_zh": f"以 ${prem:.2f} 买入 ${strike:.1f} 的看涨期权 (Call)。保本价为 ${strike + prem:.2f}。"
            })
        return sorted(recs, key=lambda x: abs(x["delta"]), reverse=(risk_profile == "safest"))

    def _recommend_long_put(self, puts, S, T, dte, risk_profile):
        """
        Recommends Long Put for hedging or speculation.
        """
        if risk_profile == "safest":
            min_delta, max_delta = -0.85, -0.65
        elif risk_profile == "aggressive":
            min_delta, max_delta = -0.35, -0.20
        else: # moderate
            min_delta, max_delta = -0.60, -0.40
            
        matches = [p for p in puts if min_delta <= p["delta"] <= max_delta]
        if not matches:
            target = -0.75 if risk_profile == "safest" else -0.28 if risk_profile == "aggressive" else -0.50
            sorted_puts = sorted(puts, key=lambda x: abs(x["delta"] - target))
            matches = sorted_puts[:3]
            
        recs = []
        for p in matches:
            prem = p["premium"]
            strike = p["strike"]
            prob_profit = abs(p["delta"])
            
            recs.append({
                "strike": strike,
                "premium": prem,
                "bid": p["bid"],
                "ask": p["ask"],
                "delta": p["delta"],
                "volume": p["volume"],
                "openInterest": p["openInterest"],
                "prob_worthless": 1.0 - prob_profit,
                "break_even": strike - prem,
                "max_profit": (strike - prem) * 100,
                "max_loss": prem * 100,
                "description_en": f"Buy ${strike:.1f} Put for ${prem:.2f}. Break-even is ${strike - prem:.2f}.",
                "description_zh": f"以 ${prem:.2f} 买入 ${strike:.1f} 的看跌期权 (Put)。保本价为 ${strike - prem:.2f}。"
            })
        return sorted(recs, key=lambda x: abs(x["delta"]), reverse=(risk_profile == "safest"))

    def _recommend_spreads(self, calls, puts, S, T, dte, strategy, risk_profile):
        """
        Recommends spreads: bull_call_spread, bear_put_spread, bull_put_spread, bear_call_spread.
        """
        recs = []
        
        # Bull Call Spread (Debit Spread)
        if strategy == "spread_bull_call":
            for c_long in calls:
                if 0.42 <= c_long["delta"] <= 0.58:
                    for c_short in calls:
                        if 0.20 <= c_short["delta"] <= 0.38 and c_short["strike"] > c_long["strike"]:
                            net_debit = c_long["premium"] - c_short["premium"]
                            if net_debit <= 0: continue
                            width = c_short["strike"] - c_long["strike"]
                            max_profit = width - net_debit
                            recs.append({
                                "buy_strike": c_long["strike"],
                                "sell_strike": c_short["strike"],
                                "net_cost": net_debit,
                                "max_profit": max_profit * 100,
                                "max_loss": net_debit * 100,
                                "break_even": c_long["strike"] + net_debit,
                                "description_en": f"Buy ${c_long['strike']:.1f} Call / Sell ${c_short['strike']:.1f} Call. Net cost: ${net_debit:.2f}. Max Profit: ${max_profit*100:.0f}.",
                                "description_zh": f"买入 ${c_long['strike']:.1f} Call / 卖出 ${c_short['strike']:.1f} Call。净成本: ${net_debit:.2f}。最大利润: ${max_profit*100:.0f}。"
                            })
                            
        # Bear Put Spread (Debit Spread)
        elif strategy == "spread_bear_put":
            for p_long in puts:
                if -0.58 <= p_long["delta"] <= -0.42:
                    for p_short in puts:
                        if -0.38 <= p_short["delta"] <= -0.20 and p_short["strike"] < p_long["strike"]:
                            net_debit = p_long["premium"] - p_short["premium"]
                            if net_debit <= 0: continue
                            width = p_long["strike"] - p_short["strike"]
                            max_profit = width - net_debit
                            recs.append({
                                "buy_strike": p_short["strike"],
                                "sell_strike": p_long["strike"],
                                "net_cost": net_debit,
                                "max_profit": max_profit * 100,
                                "max_loss": net_debit * 100,
                                "break_even": p_long["strike"] - net_debit,
                                "description_en": f"Buy ${p_long['strike']:.1f} Put / Sell ${p_short['strike']:.1f} Put. Net cost: ${net_debit:.2f}. Max Profit: ${max_profit*100:.0f}.",
                                "description_zh": f"买入 ${p_long['strike']:.1f} Put / 卖出 ${p_short['strike']:.1f} Put。净成本: ${net_debit:.2f}。最大利润: ${max_profit*100:.0f}。"
                            })
                            
        # Bull Put Credit Spread
        elif strategy == "spread_bull_put":
            if risk_profile == "safest":
                sell_target = -0.15
            elif risk_profile == "aggressive":
                sell_target = -0.42
            else:
                sell_target = -0.28
                
            for p_short in puts:
                if abs(p_short["delta"] - sell_target) <= 0.08:
                    for p_buy in puts:
                        if p_buy["strike"] < p_short["strike"] and (p_short["strike"] - p_buy["strike"]) <= 20:
                            net_credit = p_short["premium"] - p_buy["premium"]
                            if net_credit <= 0: continue
                            width = p_short["strike"] - p_buy["strike"]
                            max_loss = width - net_credit
                            recs.append({
                                "buy_strike": p_buy["strike"],
                                "sell_strike": p_short["strike"],
                                "net_cost": -net_credit,
                                "max_profit": net_credit * 100,
                                "max_loss": max_loss * 100,
                                "break_even": p_short["strike"] - net_credit,
                                "description_en": f"Sell ${p_short['strike']:.1f} Put / Buy ${p_buy['strike']:.1f} Put. Collect ${net_credit:.2f} credit. Max Risk: ${max_loss*100:.0f}.",
                                "description_zh": f"卖出 ${p_short['strike']:.1f} Put / 买入 ${p_buy['strike']:.1f} Put。收取 ${net_credit:.2f} 净权利金。最大风险: ${max_loss*100:.0f}。"
                            })
                            
        # Bear Call Credit Spread
        elif strategy == "spread_bear_call":
            if risk_profile == "safest":
                sell_target = 0.15
            elif risk_profile == "aggressive":
                sell_target = 0.42
            else:
                sell_target = 0.28
                
            for c_short in calls:
                if abs(c_short["delta"] - sell_target) <= 0.08:
                    for c_buy in calls:
                        if c_buy["strike"] > c_short["strike"] and (c_buy["strike"] - c_short["strike"]) <= 20:
                            net_credit = c_short["premium"] - c_buy["premium"]
                            if net_credit <= 0: continue
                            width = c_buy["strike"] - c_short["strike"]
                            max_loss = width - net_credit
                            recs.append({
                                "buy_strike": c_short["strike"],
                                "sell_strike": c_buy["strike"],
                                "net_cost": -net_credit,
                                "max_profit": net_credit * 100,
                                "max_loss": max_loss * 100,
                                "break_even": c_short["strike"] + net_credit,
                                "description_en": f"Sell ${c_short['strike']:.1f} Call / Buy ${c_buy['strike']:.1f} Call. Collect ${net_credit:.2f} credit. Max Risk: ${max_loss*100:.0f}.",
                                "description_zh": f"卖出 ${c_short['strike']:.1f} Call / 买入 ${c_buy['strike']:.1f} Call。收取 ${net_credit:.2f} 净权利金。最大风险: ${max_loss*100:.0f}。"
                            })
                            
        return sorted(recs, key=lambda x: x["max_profit"], reverse=True)[:5]

    def _recommend_straddle(self, calls, puts, S, T, dte, risk_profile):
        """
        Recommends Straddle: ATM calls and puts on same strike.
        """
        sorted_calls = sorted(calls, key=lambda x: abs(x["strike"] - S))
        if not sorted_calls:
            return []
            
        atm_strike = sorted_calls[0]["strike"]
        matching_puts = [p for p in puts if p["strike"] == atm_strike]
        if not matching_puts:
            return []
            
        atm_call = sorted_calls[0]
        atm_put = matching_puts[0]
        
        call_prem = atm_call["premium"]
        put_prem = atm_put["premium"]
        total_premium = call_prem + put_prem
        
        is_long = (risk_profile != "safest")
        
        rec = {
            "strike": atm_strike,
            "call_premium": call_prem,
            "put_premium": put_prem,
            "net_cost": total_premium if is_long else -total_premium,
            "max_profit": float('inf') if is_long else total_premium * 100,
            "max_loss": total_premium * 100 if is_long else float('inf'),
            "break_even_lower": atm_strike - total_premium,
            "break_even_upper": atm_strike + total_premium,
            "description_en": f"{'Buy' if is_long else 'Sell'} ATM ${atm_strike:.1f} Straddle (Call + Put) for ${total_premium:.2f}. Break-evens: ${atm_strike - total_premium:.2f} and ${atm_strike + total_premium:.2f}.",
            "description_zh": f"{'买入' if is_long else '卖出'} ATM ${atm_strike:.1f} 跨式组合 (Straddle, Call + Put)，价格为 ${total_premium:.2f}。盈亏平衡点: ${atm_strike - total_premium:.2f} 和 ${atm_strike + total_premium:.2f}。"
        }
        
        return [rec]

    def get_option_walls(self, ticker_symbol: str, current_price: float, r_rate: float = 0.0425) -> dict:
        """
        Calculates Put Wall and Call Wall based on open interest (and volume fallback)
        for the liquid near-term options expiration.
        """
        ticker_symbol = ticker_symbol.strip().upper()
        ticker = yf.Ticker(ticker_symbol)
        
        try:
            expirations = ticker.options
        except Exception:
            return {}
            
        if not expirations:
            return {}
            
        # Select expiration closest to 30 days DTE
        today = _market_date()
        selected_exp = expirations[0]
        for exp in expirations:
            try:
                exp_d = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
                if 25 <= (exp_d - today).days <= 45:
                    selected_exp = exp
                    break
            except Exception:
                pass
                
        # Fetch options data
        data = self.get_options_data(ticker_symbol, selected_exp, r_rate)
        if "error" in data or (not data["calls"] and not data["puts"]):
            # try first expiration as fallback
            if len(expirations) > 1 and selected_exp != expirations[0]:
                selected_exp = expirations[0]
                data = self.get_options_data(ticker_symbol, selected_exp, r_rate)
                if "error" in data:
                    return {}
            else:
                return {}
                
        calls = data["calls"]
        puts = data["puts"]
        
        # Helper to find wall
        def find_wall_strike(options_list):
            if not options_list:
                return None, 0.0
            best_strike = None
            max_weight = -1.0
            for opt in options_list:
                oi = opt.get("openInterest", 0.0)
                vol = opt.get("volume", 0.0)
                # Combine OI and volume (giving OI more weight)
                weight = oi * 1.0 + vol * 0.2
                if weight > max_weight:
                    max_weight = weight
                    best_strike = opt["strike"]
            return best_strike, max_weight
            
        call_wall, c_weight = find_wall_strike(calls)
        put_wall, p_weight = find_wall_strike(puts)
        
        # Midpoint of the walls acts as the options-implied price ceiling/floor consensus
        midpoint = None
        if call_wall is not None and put_wall is not None:
            midpoint = (call_wall + put_wall) / 2.0
            
        return {
            "expiration": selected_exp,
            "dte": data.get("dte", 30),
            "call_wall": call_wall,
            "call_wall_weight": c_weight,
            "put_wall": put_wall,
            "put_wall_weight": p_weight,
            "midpoint": midpoint
        }
