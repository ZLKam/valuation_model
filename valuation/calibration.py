import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
import datetime
import logging

logger = logging.getLogger(__name__)

class CalibrationEngine:
    def __init__(self, data_provider):
        self.data_provider = data_provider

    def _get_ticker_profile(self, ticker_info: dict) -> str:
        """
        Classifies stock into one of 5 profiles:
        - FINANCIAL: banks, insurance, financial services (DCF is invalid)
        - HIGH_GROWTH_AI: tech, semiconductors, software (prioritizes multiples/forward growth)
        - CYCLICAL: energy, mining, materials, autos (volatile FCF, asset-heavy, multiples-driven)
        - CASH_COW_MATURE: high dividend yield, stable low-growth cash cows
        - STANDARD: balanced default
        """
        quote_type = str(ticker_info.get("quoteType", "")).strip().upper()
        if quote_type == "ETF":
            return "ETF"
            
        sector = ticker_info.get("sector", "")
        industry = ticker_info.get("industry", "")
        div_yield = ticker_info.get("dividendYield", 0.0) or 0.0
        
        sector_lower = str(sector).strip().lower()
        industry_lower = str(industry).strip().lower()
        
        if "financial" in sector_lower or "bank" in industry_lower or "insurance" in industry_lower:
            return "FINANCIAL"
        elif "semiconductor" in industry_lower or "software" in industry_lower or "computer" in industry_lower or "internet" in industry_lower:
            return "HIGH_GROWTH_AI"
        elif "oil" in industry_lower or "gas" in industry_lower or "mining" in industry_lower or "metals" in industry_lower or "steel" in industry_lower or "chemical" in industry_lower or "auto" in industry_lower:
            return "CYCLICAL"
        elif div_yield > 0.025:
            return "CASH_COW_MATURE"
        else:
            return "STANDARD"

    def calibrate(self, ticker_symbol: str, financials_history: pd.DataFrame, ticker_info: dict, base_growth_rate: float, base_wacc: float) -> dict:
        """
        Runs a rolling backtest of all 4 valuation models (DCF, Graham, Peer Multiples, DDM) on historical data,
        and optimizes growth/discount rate modifiers and model weights to minimize prediction error
        relative to the actual stock price 12 months later.
        """
        ticker_symbol = ticker_symbol.strip().upper()
        profile = self._get_ticker_profile(ticker_info)
        logger.info(f"Running profile-aware calibration for {ticker_symbol} (Profile: {profile})")
        
        if profile == "ETF":
            logger.info(f"Skipping calibration optimization for ETF: {ticker_symbol}")
            return self._default_calibration(base_growth_rate, base_wacc, profile)
        
        # 1. Fetch 5 years of daily price history for backtesting
        try:
            ticker = yf.Ticker(ticker_symbol)
            price_history = ticker.history(period="5y")
            if price_history.empty:
                logger.warning(f"No price history found for {ticker_symbol}. Skipping calibration.")
                return self._default_calibration(base_growth_rate, base_wacc, profile)
        except Exception as e:
            logger.error(f"Error fetching price history for calibration: {e}")
            return self._default_calibration(base_growth_rate, base_wacc, profile)

        # 2. Reconstruct historical statements and align with prices
        report_dates = financials_history.index.tolist()
        if len(report_dates) < 2:
            logger.info("Not enough historical periods (need at least 2) for calibration. Using defaults.")
            return self._default_calibration(base_growth_rate, base_wacc, profile)

        backtest_records = []
        shares = ticker_info.get("sharesOutstanding") or ticker_info.get("impliedSharesOutstanding")
        if not shares or shares <= 0:
            logger.warning("Shares outstanding missing or invalid. Skipping calibration.")
            return self._default_calibration(base_growth_rate, base_wacc, profile)

        # Pre-fetch peer multiples once for the multiples valuation model in the loss function
        try:
            from valuation.multiples import MultiplesModel
            peers = self.data_provider.get_peer_tickers(ticker_symbol)
            multiples_model = MultiplesModel()
            multiples_results = multiples_model.calculate(ticker_symbol, peers, ticker_info, financials_history)
            pe_forward_median = multiples_results["stats"].get("pe_forward", {}).get("median", np.nan)
            ev_ebitda_median = multiples_results["stats"].get("ev_ebitda", {}).get("median", np.nan)
            pe_trailing_median = multiples_results["stats"].get("pe_trailing", {}).get("median", np.nan)
            ev_revenue_median = multiples_results["stats"].get("ev_revenue", {}).get("median", np.nan)
        except Exception as e:
            logger.warning(f"Failed to fetch peer multiples for calibration: {e}")
            pe_forward_median = ev_ebitda_median = pe_trailing_median = ev_revenue_median = np.nan

        # Collect risk-free rate proxy
        risk_free_rate = ticker_info.get("riskFreeRate") or 0.0425
        bond_yield = max(risk_free_rate * 100.0, 3.0)

        # Build historical records
        for report_date in report_dates:
            eval_date = pd.to_datetime(report_date) + datetime.timedelta(days=90)
            future_date = eval_date + datetime.timedelta(days=365)
            
            # Check if we have future price data
            if future_date > pd.to_datetime(price_history.index[-1]).tz_localize(None):
                continue
                
            price_eval = self._get_closest_price(price_history, eval_date)
            price_actual_future = self._get_closest_price(price_history, future_date)
            
            if pd.isna(price_eval) or pd.isna(price_actual_future):
                continue

            hist_slice = financials_history.loc[:report_date]
            if len(hist_slice) < 1:
                continue
                
            latest_row = hist_slice.iloc[-1]
            
            # Extract historical items
            rev = latest_row.get("revenue", np.nan)
            fcf = latest_row.get("fcf", np.nan)
            cash = latest_row.get("total_cash", 0.0)
            debt = latest_row.get("total_debt", 0.0)
            ebitda = latest_row.get("ebitda", np.nan)
            eps = latest_row.get("net_income", 0.0) / shares if shares else np.nan
            bvps = latest_row.get("equity", 0.0) / shares if shares else np.nan
            
            if pd.isna(rev) or pd.isna(fcf):
                continue
                
            if pd.isna(ebitda) or ebitda <= 0:
                ebitda = latest_row.get("ebit", 0.0) or 0.0

            # Approximate historical dividend rate from cash flow statement
            div_paid = latest_row.get("cash_dividends_paid", np.nan)
            if pd.isna(div_paid) or div_paid == 0:
                div_paid = latest_row.get("common_stock_dividend_paid", 0.0)
            div_rate = abs(div_paid) / shares if (shares and not pd.isna(div_paid)) else 0.0
            if div_rate == 0:
                div_rate = ticker_info.get("dividendRate", 0.0) or 0.0

            # Calculate rolling historical growth rate
            hist_growth = base_growth_rate
            if len(hist_slice) >= 2:
                revs = hist_slice["revenue"].dropna()
                if len(revs) >= 2:
                    hist_growth = revs.pct_change().mean()
                    if pd.isna(hist_growth):
                        hist_growth = base_growth_rate

            backtest_records.append({
                "eval_date": eval_date,
                "price_eval": price_eval,
                "price_actual_future": price_actual_future,
                "rev": rev,
                "fcf": fcf,
                "cash": cash,
                "debt": debt,
                "ebitda": ebitda,
                "eps": eps,
                "bvps": bvps,
                "dividend_rate": div_rate,
                "hist_growth": hist_growth,
                "shares": shares
            })

        if len(backtest_records) < 1:
            logger.info("No valid historical evaluation data points found. Using default calibration.")
            return self._default_calibration(base_growth_rate, base_wacc, profile)

        # 3. Setup weights and constraints based on profile
        # base_weights = [dcf, graham, multiples, ddm]
        if profile == "FINANCIAL":
            base_weights = [0.0, 0.20, 0.50, 0.30]
            bounds = [(0.0, 0.0), (0.1, 0.4), (0.3, 0.7), (0.1, 0.5)]
        elif profile == "HIGH_GROWTH_AI":
            base_weights = [0.25, 0.15, 0.60, 0.00]
            bounds = [(0.1, 0.4), (0.05, 0.3), (0.4, 0.8), (0.0, 0.0)]
        elif profile == "CYCLICAL":
            base_weights = [0.10, 0.40, 0.50, 0.00]
            bounds = [(0.05, 0.3), (0.2, 0.6), (0.3, 0.7), (0.0, 0.0)]
        elif profile == "CASH_COW_MATURE":
            base_weights = [0.40, 0.15, 0.20, 0.25]
            bounds = [(0.2, 0.6), (0.05, 0.3), (0.1, 0.4), (0.1, 0.4)]
        else: # STANDARD
            base_weights = [0.35, 0.30, 0.25, 0.10]
            bounds = [(0.1, 0.6), (0.1, 0.5), (0.1, 0.5), (0.0, 0.3)]

        # Optimizer parameter layout:
        # params[0] = growth_multiplier (0.5 to 1.5)
        # params[1] = wacc_offset (-0.03 to +0.05)
        # params[2:] = raw weights for [dcf, graham, multiples, ddm]
        init_guess = [1.0, 0.0] + base_weights
        all_bounds = [(0.5, 1.5), (-0.03, 0.05)] + bounds

        def loss_function(params):
            growth_mult = params[0]
            wacc_offset = params[1]
            raw_w = params[2:]
            
            # Normalize weights internally so they sum to 1.0
            sum_rw = sum(raw_w)
            if sum_rw > 0:
                normalized_w = [v / sum_rw for v in raw_w]
            else:
                normalized_w = [0.25, 0.25, 0.25, 0.25]
                
            dcf_w, graham_w, multiples_w, ddm_w = normalized_w
            squared_errors = []
            
            for rec in backtest_records:
                # A. DCF Model
                g_param = max(min(rec["hist_growth"] * growth_mult, 0.40), -0.2)
                wacc_param = max(base_wacc + wacc_offset, 0.05)
                
                fcfs = []
                cf = rec["fcf"]
                if cf <= 0:
                    cf = rec["rev"] * 0.05
                    
                for y in range(1, 6):
                    cf *= (1.0 + g_param)
                    fcfs.append(cf)
                    
                dfs = [1.0 / ((1.0 + wacc_param) ** y) for y in range(1, 6)]
                pv_fcfs = sum(f * df for f, df in zip(fcfs, dfs))
                
                g_term = 0.02
                gap = wacc_param - g_term
                if gap <= 0.005:
                    g_term = wacc_param - 0.01
                    gap = 0.01
                tv = (fcfs[-1] * (1.0 + g_term)) / gap
                pv_tv = tv / ((1.0 + wacc_param) ** 5)
                
                ev = pv_fcfs + pv_tv
                eq_val = ev + rec["cash"] - rec["debt"]
                dcf_price = max(eq_val / rec["shares"], 0.0)
                
                # B. Graham Model
                graham_price = 0.0
                eps_val = rec["eps"]
                bvps_val = rec["bvps"]
                if eps_val and bvps_val and eps_val > 0 and bvps_val > 0:
                    # Revised Graham cap
                    g_pct = max(min(g_param * 100.0, 20.0), 0.0)
                    graham_price = (eps_val * (8.5 + 2 * g_pct) * 4.4) / bond_yield
                else:
                    graham_price = dcf_price
                    
                # C. Multiples Model
                multiples_price = dcf_price
                if not pd.isna(pe_forward_median) and eps_val and eps_val > 0:
                    multiples_price = pe_forward_median * (eps_val * (1.0 + g_param))
                elif not pd.isna(ev_ebitda_median) and rec["ebitda"] > 0:
                    implied_ev = ev_ebitda_median * rec["ebitda"]
                    implied_price = (implied_ev + rec["cash"] - rec["debt"]) / rec["shares"]
                    multiples_price = max(implied_price, 0.0)
                elif not pd.isna(pe_trailing_median) and eps_val > 0:
                    multiples_price = pe_trailing_median * eps_val
                elif not pd.isna(ev_revenue_median) and rec["rev"] > 0:
                    implied_ev = ev_revenue_median * rec["rev"]
                    implied_price = (implied_ev + rec["cash"] - rec["debt"]) / rec["shares"]
                    multiples_price = max(implied_price, 0.0)
                    
                # D. DDM Model
                ddm_price = 0.0
                div_rate = rec["dividend_rate"]
                if div_rate > 0:
                    r = wacc_param
                    g_stable = 0.02
                    g_high = g_param
                    
                    pv_stage1 = 0.0
                    current_div = div_rate
                    for t in range(1, 6):
                        current_div *= (1.0 + g_high)
                        pv_stage1 += current_div / ((1.0 + r) ** t)
                    terminal_div = current_div * (1.0 + g_stable)
                    if r > g_stable:
                        pv_term = terminal_div / (r - g_stable)
                        pv_term_disc = pv_term / ((1.0 + r) ** 5)
                        ddm_price = pv_stage1 + pv_term_disc
                    else:
                        ddm_price = pv_stage1
                    ddm_price = max(ddm_price, 0.0)
                else:
                    ddm_price = multiples_price # fallback
                    
                weighted_target = (
                    dcf_w * dcf_price +
                    graham_w * graham_price +
                    multiples_w * multiples_price +
                    ddm_w * ddm_price
                )
                
                pct_err = (weighted_target - rec["price_actual_future"]) / rec["price_actual_future"]
                squared_errors.append(pct_err ** 2)
                
            return np.mean(squared_errors)

        try:
            res = minimize(loss_function, init_guess, bounds=all_bounds, method='L-BFGS-B')
            if res.success:
                opt_growth_mult, opt_wacc_offset = res.x[0], res.x[1]
                raw_opt_w = res.x[2:]
                sum_rw = sum(raw_opt_w)
                if sum_rw > 0:
                    opt_w = [v / sum_rw for v in raw_opt_w]
                else:
                    opt_w = base_weights
                logger.info(f"Calibration successful for {ticker_symbol}! Profile: {profile}. "
                            f"Optimal Growth Mult: {opt_growth_mult:.3f}, WACC Offset: {opt_wacc_offset*100:+.2f}%. "
                            f"Learned Weights -> DCF: {opt_w[0]:.2f}, Graham: {opt_w[1]:.2f}, Multiples: {opt_w[2]:.2f}, DDM: {opt_w[3]:.2f}")
            else:
                opt_growth_mult, opt_wacc_offset = init_guess[0], init_guess[1]
                opt_w = base_weights
                logger.warning(f"Calibration optimization failed to converge for {ticker_symbol}. Using defaults.")
        except Exception as e:
            logger.error(f"Calibration optimization error for {ticker_symbol}: {e}")
            opt_growth_mult, opt_wacc_offset = init_guess[0], init_guess[1]
            opt_w = base_weights

        return {
            "calibrated": True,
            "profile": profile,
            "growth_multiplier": opt_growth_mult,
            "wacc_offset": opt_wacc_offset,
            "dcf_weight": opt_w[0],
            "graham_weight": opt_w[1],
            "multiples_weight": opt_w[2],
            "ddm_weight": opt_w[3],
            "backtest_points": len(backtest_records),
            "backtest_records": backtest_records
        }

    def _default_calibration(self, base_growth_rate: float, base_wacc: float, profile: str = "STANDARD") -> dict:
        if profile == "ETF":
            w = [0.0, 0.0, 0.0, 0.0]
        elif profile == "FINANCIAL":
            w = [0.0, 0.20, 0.50, 0.30]
        elif profile == "HIGH_GROWTH_AI":
            w = [0.25, 0.15, 0.60, 0.00]
        elif profile == "CYCLICAL":
            w = [0.10, 0.40, 0.50, 0.00]
        elif profile == "CASH_COW_MATURE":
            w = [0.40, 0.15, 0.20, 0.25]
        else:
            w = [0.35, 0.30, 0.25, 0.10]
            
        return {
            "calibrated": False,
            "profile": profile,
            "growth_multiplier": 1.0,
            "wacc_offset": 0.0,
            "dcf_weight": w[0],
            "graham_weight": w[1],
            "multiples_weight": w[2],
            "ddm_weight": w[3],
            "backtest_points": 0,
            "backtest_records": []
        }

    def _get_closest_price(self, price_history: pd.DataFrame, target_date: datetime.datetime) -> float:
        if price_history.empty:
            return np.nan
        naive_idx = price_history.index.tz_localize(None)
        diffs = np.abs(naive_idx - pd.to_datetime(target_date))
        closest_idx = diffs.argmin()
        if diffs[closest_idx] > pd.Timedelta(days=7):
            return np.nan
        return float(price_history["Close"].iloc[closest_idx])
