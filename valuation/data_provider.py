import yfinance as yf
import pandas as pd
import numpy as np
import requests
import logging
from io import StringIO
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProvider:
    def __init__(self):
        self._sp500_cache = None
        self._erp_cache = None

    def get_stock_data(self, ticker_symbol: str):
        """
        Fetches yfinance Ticker object and validates basic existence.
        """
        ticker_symbol = ticker_symbol.strip().upper()
        ticker = yf.Ticker(ticker_symbol)
        try:
            # Simple check to see if ticker exists (fetch info)
            _ = ticker.info
        except Exception as e:
            logger.warning(f"Error fetching ticker info for {ticker_symbol}: {e}")
        return ticker

    def get_financial_row(self, df: pd.DataFrame, keys: list) -> pd.Series:
        """
        Helper to robustly find a row in a financial statement DataFrame
        using a list of possible key names (case-insensitive and matching sub-strings).
        """
        if df is None or df.empty:
            return pd.Series(dtype=float)
        
        # Convert index to lowercase string list for easy matching
        idx_normalized = [str(x).strip().lower() for x in df.index]
        
        for key in keys:
            key_lower = key.strip().lower()
            # 1. Look for exact match
            if key_lower in idx_normalized:
                pos = idx_normalized.index(key_lower)
                return df.iloc[pos]
            
            # 2. Look for partial match
            for i, idx_val in enumerate(idx_normalized):
                if key_lower in idx_val or idx_val in key_lower:
                    return df.iloc[i]
                    
        return pd.Series(dtype=float)

    def fetch_raw_statements(self, ticker) -> dict:
        """
        Fetches income statement, balance sheet, and cash flow statements.
        Returns a dict of DataFrames.
        """
        try:
            income = ticker.income_stmt
        except Exception:
            income = pd.DataFrame()
            
        try:
            balance = ticker.balance_sheet
        except Exception:
            balance = pd.DataFrame()
            
        try:
            cash = ticker.cash_flow
        except Exception:
            cash = pd.DataFrame()
            
        return {
            "income": income,
            "balance": balance,
            "cash": cash
        }

    def extract_financials_history(self, ticker) -> pd.DataFrame:
        """
        Extracts key financial items over time (aligned by date) into a single DataFrame.
        """
        statements = self.fetch_raw_statements(ticker)
        inc = statements["income"]
        bal = statements["balance"]
        cf = statements["cash"]
        
        # Find all available dates
        all_dates = set()
        for df in [inc, bal, cf]:
            if df is not None and not df.empty:
                all_dates.update(df.columns)
                
        sorted_dates = sorted(list(all_dates), reverse=True) # newest first
        
        # We'll construct a combined DataFrame where columns are dates and rows are metrics
        metrics = {}
        
        # Extract metrics
        metrics["revenue"] = self.get_financial_row(inc, ["Total Revenue", "Revenue", "TotalRevenue", "Operating Revenue"])
        metrics["ebitda"] = self.get_financial_row(inc, ["EBITDA", "Normalized EBITDA", "NormalizedEBITDA"])
        metrics["ebit"] = self.get_financial_row(inc, ["EBIT", "Operating Income", "OperatingIncome", "Operating Income Value"])
        metrics["net_income"] = self.get_financial_row(inc, ["Net Income", "NetIncome", "Net Income Common Stockholders", "Net Income Loss"])
        metrics["pretax_income"] = self.get_financial_row(inc, ["Pretax Income", "PretaxIncome", "Income Before Tax", "IncomeBeforeTax"])
        metrics["tax_provision"] = self.get_financial_row(inc, ["Tax Provision", "TaxProvision", "Income Tax Expense", "IncomeTaxExpense"])
        metrics["interest_expense"] = self.get_financial_row(inc, ["Interest Expense", "InterestExpense", "Interest Expense Net OF Interest Income", "Interest Expense Net"])
        
        # Balance Sheet
        metrics["total_cash"] = self.get_financial_row(bal, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash Financial"])
        metrics["total_debt"] = self.get_financial_row(bal, ["Total Debt", "TotalDebt", "Total Debt Value"])
        
        # If total_debt is missing, calculate Long Term Debt + Short Term Debt
        if metrics["total_debt"].isna().all():
            lt_debt = self.get_financial_row(bal, ["Long Term Debt", "LongTermDebt", "Long Term Debt Value"])
            st_debt = self.get_financial_row(bal, ["Current Debt", "Short Long Term Debt", "Current Debt Value", "ShortTermDebt"])
            metrics["total_debt"] = lt_debt.fillna(0) + st_debt.fillna(0)
            
        metrics["total_assets"] = self.get_financial_row(bal, ["Total Assets", "TotalAssets", "Total Assets Value"])
        metrics["total_liabilities"] = self.get_financial_row(bal, ["Total Liabilities", "TotalLiabilities Net Minority Interest", "Total Liabilities Value"])
        metrics["equity"] = self.get_financial_row(bal, ["Stockholders Equity", "Total Stockholders Equity", "StockholdersEquity"])
        
        # Cash Flow
        metrics["operating_cash_flow"] = self.get_financial_row(cf, ["Operating Cash Flow", "Cash Flow From Operating Activities", "CashFlowFromOperatingActivities", "Net Cash Provided By Operating Activities"])
        metrics["capex"] = self.get_financial_row(cf, ["Capital Expenditure", "CapitalExpenditure", "CapEx", "Net Capital Expenditures"])
        
        # Make capex positive for our calculations (usually negative in CF stmt)
        if not metrics["capex"].isna().all():
            metrics["capex"] = metrics["capex"].abs()
            
        metrics["fcf"] = self.get_financial_row(cf, ["Free Cash Flow", "FreeCashFlow"])
        
        # If FCF is missing, calculate Operating Cash Flow - Capex
        if metrics["fcf"].isna().all():
            metrics["fcf"] = metrics["operating_cash_flow"].fillna(0) - metrics["capex"].fillna(0)
            
        # Build DataFrame
        df_combined = pd.DataFrame(index=sorted_dates)
        for name, series in metrics.items():
            if not series.empty:
                # Align series values to sorted_dates
                df_combined[name] = df_combined.index.map(series)
            else:
                df_combined[name] = np.nan
                
        # Drop rows (dates) where we have completely empty financial data
        df_combined = df_combined.dropna(how='all')
        
        # Sort index ascending (oldest to newest)
        df_combined = df_combined.sort_index(ascending=True)
        return df_combined

    def get_risk_free_rate(self) -> float:
        """
        Fetches the current 10-year US Treasury yield (^TNX).
        If fails, returns a default rate of 4.25% (0.0425).
        """
        try:
            tnx = yf.Ticker("^TNX")
            # Get latest close price of the Treasury yield
            history = tnx.history(period="1d")
            if not history.empty:
                # The yield is quoted in percentage points, e.g. 4.25 means 4.25%
                yield_val = history["Close"].iloc[-1] / 100.0
                if yield_val > 0:
                    logger.info(f"Fetched current 10-Year Treasury Yield: {yield_val*100:.3f}%")
                    return yield_val
        except Exception as e:
            logger.warning(f"Failed to fetch ^TNX: {e}. Falling back to default risk-free rate of 4.25%.")
        return 0.0425

    def get_market_risk_premium(self) -> float:
        """Fetch the latest published US implied equity risk premium.

        The prior fixed 5.5% premium materially overstated the current hurdle
        rate for large US companies.  Damodaran's regularly updated implied ERP
        is used when available, with a conservative 4.5% fallback.
        """
        if self._erp_cache is not None:
            return self._erp_cache
        fallback = 0.045
        try:
            url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm"
            response = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            plain_text = re.sub(r"<[^>]+>", "", response.text)
            match = re.search(r"Implied ERP on.{0,120}?=\s*([0-9]+(?:\.[0-9]+)?)%", plain_text, re.IGNORECASE | re.DOTALL)
            if match:
                premium = float(match.group(1)) / 100.0
                if 0.025 <= premium <= 0.08:
                    self._erp_cache = premium
                    logger.info("Fetched current US implied equity risk premium: %.2f%%", premium * 100)
                    return premium
        except Exception as exc:
            logger.warning("Failed to fetch the implied equity risk premium: %s", exc)
        self._erp_cache = fallback
        return fallback

    def get_sp500_companies(self) -> pd.DataFrame:
        """
        Scrapes list of S&P 500 companies from Wikipedia, including symbols, sector, and industry.
        Caches results in memory.
        """
        if self._sp500_cache is not None:
            return self._sp500_cache
            
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            # pandas 3 treats a raw HTML string as a possible file path.  Wrap
            # the response explicitly so peer discovery keeps working.
            tables = pd.read_html(StringIO(response.text), attrs={'id': 'constituents'})
            df = tables[0]
            # Standardize column names
            df = df.rename(columns={
                "Symbol": "symbol",
                "Security": "name",
                "GICS Sector": "sector",
                "GICS Sub-Industry": "industry"
            })
            # Clean symbols (some use '.' instead of '-' for class shares, e.g. BRK.B)
            df['symbol'] = df['symbol'].str.replace('.', '-', regex=False)
            self._sp500_cache = df[['symbol', 'name', 'sector', 'industry']]
            logger.info("Successfully fetched S&P 500 constituents from Wikipedia.")
            return self._sp500_cache
        except Exception as e:
            logger.error(f"Error fetching S&P 500 constituents: {e}")
            # Fallback to empty df or a tiny static list of market leaders
            fallback_data = [
                {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Information Technology", "industry": "Technology Hardware, Storage & Peripherals"},
                {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Information Technology", "industry": "Systems Software"},
                {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Communication Services", "industry": "Interactive Media & Services"},
                {"symbol": "AMZN", "name": "Amazon.com, Inc.", "sector": "Consumer Discretionary", "industry": "Broadline Retail"},
                {"symbol": "META", "name": "Meta Platforms, Inc.", "sector": "Communication Services", "industry": "Interactive Media & Services"},
                {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Information Technology", "industry": "Semiconductors"},
                {"symbol": "TSLA", "name": "Tesla, Inc.", "sector": "Consumer Discretionary", "industry": "Automobile Manufacturers"},
                {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financials", "industry": "Diversified Banks"},
                {"symbol": "V", "name": "Visa Inc.", "sector": "Financials", "industry": "Transaction & Payment Processing Services"},
                {"symbol": "PG", "name": "Procter & Gamble Co.", "sector": "Consumer Staples", "industry": "Household Products"},
                {"symbol": "KO", "name": "The Coca-Cola Company", "sector": "Consumer Staples", "industry": "Soft Drinks"},
                {"symbol": "XOM", "name": "Exxon Mobil Corporation", "sector": "Energy", "industry": "Integrated Oil & Gas"}
            ]
            return pd.DataFrame(fallback_data)

    def get_peer_tickers(self, ticker_symbol: str, limit: int = 5) -> list:
        """
        Finds peers from both business similarity and market-cap similarity.

        Pure industry matching compared unique market leaders such as Apple to
        much smaller commodity-hardware firms.  Large companies now receive a
        peer set that includes similarly scaled sector leaders; smaller firms
        retain a stronger same-industry emphasis.
        """
        ticker_symbol = ticker_symbol.strip().upper()
        sp500 = self.get_sp500_companies()
        try:
            info = yf.Ticker(ticker_symbol).info or {}
        except Exception:
            info = {}
        yahoo_sector = str(info.get("sector") or "")
        yahoo_industry = str(info.get("industry") or "")
        target_cap = float(info.get("marketCap") or 0.0)

        target_row = sp500[sp500["symbol"] == ticker_symbol]
        gics_industry = str(target_row.iloc[0]["industry"]) if not target_row.empty else yahoo_industry
        gics_sector = str(target_row.iloc[0]["sector"]) if not target_row.empty else yahoo_sector
        industry_peers = sp500[
            (sp500["industry"] == gics_industry) & (sp500["symbol"] != ticker_symbol)
        ]["symbol"].tolist() if gics_industry else []

        size_peers = []
        if yahoo_sector and target_cap > 0:
            try:
                query = yf.EquityQuery("and", [
                    yf.EquityQuery("eq", ["region", "us"]),
                    yf.EquityQuery("eq", ["sector", yahoo_sector]),
                ])
                screened = yf.screen(query, size=50, sortField="intradaymarketcap", sortAsc=False)
                candidates = []
                for quote in screened.get("quotes", []):
                    symbol = str(quote.get("symbol") or "").upper()
                    market_cap = float(quote.get("marketCap") or 0.0)
                    currency = str(quote.get("currency") or "USD").upper()
                    financial_currency = str(quote.get("financialCurrency") or currency).upper()
                    exchange = str(quote.get("exchange") or "").upper()
                    if (
                        symbol
                        and symbol != ticker_symbol
                        and market_cap > 0
                        and currency == "USD"
                        and financial_currency == currency
                        and exchange in {"NMS", "NYQ"}
                        and "." not in symbol
                    ):
                        size_distance = abs(np.log(market_cap / target_cap))
                        candidates.append((size_distance, symbol))
                size_peers = [symbol for _, symbol in sorted(candidates)][: max(limit, 8)]
            except Exception as exc:
                logger.warning("Market-cap peer screen failed for %s: %s", ticker_symbol, exc)

        if target_cap >= 50e9:
            ordered = size_peers + industry_peers
        else:
            ordered = industry_peers[:3] + size_peers + industry_peers[3:]

        fallback = ["MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "AVGO"]
        ordered += fallback
        unique = []
        for symbol in ordered:
            if symbol != ticker_symbol and symbol not in unique:
                unique.append(symbol)
        return unique[:limit]

    def search_tickers(self, query: str) -> list:
        """
        Uses yfinance Lookup to find matching stock and ETF symbols in real-time,
        falling back to Yahoo search API if needed, filtered strictly to US listings.
        """
        if not query or len(query.strip()) < 1:
            return []
            
        query_str = query.strip()
        us_exchanges = {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BATS", "PNK", "OBB", "BTS", "ARC", "IEX"}
        results = []
        
        # 1. Try yfinance Lookup first (highly accurate for stock prefixes and ETFs)
        try:
            lookup = yf.Lookup(query_str)
            # Stocks
            try:
                stocks_df = lookup.get_stock()
                if not stocks_df.empty:
                    for symbol, row in stocks_df.iterrows():
                        results.append({
                            "symbol": str(symbol),
                            "name": str(row.get("shortName") or symbol),
                            "type": "EQUITY",
                            "exchange": str(row.get("exchange") or "")
                        })
            except Exception as e:
                logger.warning(f"Error retrieving stocks from Lookup for query '{query_str}': {e}")
                
            # ETFs
            try:
                etfs_df = lookup.get_etf()
                if not etfs_df.empty:
                    for symbol, row in etfs_df.iterrows():
                        results.append({
                            "symbol": str(symbol),
                            "name": str(row.get("shortName") or symbol),
                            "type": "ETF",
                            "exchange": str(row.get("exchange") or "")
                        })
            except Exception as e:
                logger.warning(f"Error retrieving ETFs from Lookup for query '{query_str}': {e}")
        except Exception as e:
            logger.error(f"yfinance Lookup instantiation failed for query '{query_str}': {e}")

        # 2. Fallback to direct Yahoo search API if Lookup returned nothing
        if not results:
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query_str}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                quotes = data.get("quotes", [])
                for q in quotes:
                    symbol = q.get("symbol")
                    name = q.get("shortname") or q.get("longname") or symbol
                    q_type = str(q.get("quoteType", "")).upper()
                    exch = str(q.get("exchange", "")).upper()
                    
                    if q_type in {"EQUITY", "ETF"}:
                        results.append({
                            "symbol": symbol,
                            "name": name,
                            "type": q_type,
                            "exchange": exch
                        })
            except Exception as e:
                logger.error(f"Fallback Yahoo search API query failed for query '{query_str}': {e}")
                
        # 3. Filter results to US Equities and ETFs, and remove duplicates
        filtered_results = []
        seen_symbols = set()
        
        for r in results:
            symbol = r["symbol"]
            if symbol in seen_symbols:
                continue
                
            q_type = r["type"].upper()
            exch = r["exchange"].upper()
            
            is_valid_type = q_type in {"EQUITY", "ETF"}
            is_us_exch = exch in us_exchanges or any(ex in exch for ex in us_exchanges)
            has_no_dot = "." not in symbol
            
            if is_valid_type and (is_us_exch or has_no_dot):
                seen_symbols.add(symbol)
                filtered_results.append(r)
                
        # Sort results: exact symbol matches first, then prefix symbol matches, then others
        query_upper = query_str.upper()
        def get_match_score(item):
            symbol = str(item["symbol"]).upper()
            if symbol == query_upper:
                return 0
            elif symbol.startswith(query_upper):
                return 1
            else:
                return 2
                
        filtered_results.sort(key=get_match_score)
        return filtered_results
