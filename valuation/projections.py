import pandas as pd
import numpy as np
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class ProjectionEngine:
    def calculate_trajectories(self, current_price: float, target_prices_12m: dict, ticker_symbol: str) -> dict:
        """
        Calculates target prices for 1 month, 3 months, 6 months, and 12 months
        under Bear, Base, and Bull scenarios using a momentum-decay and value-convergence model.
        
        target_prices_12m: dict with keys 'bear', 'base', 'bull' representing the 12m target prices.
        """
        # 1. Fetch recent price data to calculate short-term momentum
        momentum = 0.0
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="6mo")
            if len(hist) > 20:
                # Calculate return over the last 3 months (approx 63 trading days)
                close_prices = hist["Close"]
                price_now = close_prices.iloc[-1]
                price_3m_ago = close_prices.iloc[-60] if len(close_prices) >= 60 else close_prices.iloc[0]
                momentum = (price_now - price_3m_ago) / price_3m_ago
                # Cap momentum to prevent extreme values (e.g. +/- 30% for 3-month momentum)
                momentum = max(min(momentum, 0.30), -0.30)
                logger.info(f"Calculated 3-month momentum for {ticker_symbol}: {momentum*100:.2f}%")
        except Exception as e:
            logger.warning(f"Could not calculate momentum for {ticker_symbol}: {e}. Defaulting momentum to 0.")
            momentum = 0.0

        horizons = [1, 3, 6, 12]
        scenarios = ["bear", "base", "bull"]
        
        results = {s: {} for s in scenarios}
        
        for s in scenarios:
            t_12m = target_prices_12m[s]
            if pd.isna(t_12m) or t_12m is None:
                t_12m = current_price # fallback
                
            for m in horizons:
                if m == 12:
                    results[s][12] = t_12m
                else:
                    # Trajectory formula:
                    # 1. Short-term momentum decays exponentially: momentum_factor = momentum * exp(-m / 3)
                    # 2. Convergence factor: alpha = (m / 12) ^ 0.8 (slightly front-loaded)
                    # 3. Target = P0 * (1 + momentum_factor) + (T_12 - P0 * (1 + momentum_factor)) * alpha
                    
                    decay_factor = np.exp(-m / 3.0)
                    momentum_drift = current_price * (1.0 + momentum * decay_factor)
                    
                    alpha = (m / 12.0) ** 0.8
                    
                    target_price = momentum_drift + (t_12m - momentum_drift) * alpha
                    results[s][m] = max(float(target_price), 0.0)
                    
        return results
