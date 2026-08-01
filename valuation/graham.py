import numpy as np
import pandas as pd

class GrahamModel:
    def __init__(self, risk_free_rate: float = 0.0425):
        # We can use risk_free_rate as a proxy for the bond yield Y if AAA yield is not available.
        # Alternatively, we can use risk_free_rate + 1.0% spread to approximate AAA corporate bond yield.
        self.bond_yield = max(risk_free_rate * 100.0, 3.0) # Yield as a percentage, e.g., 4.25

    def calculate(self, ticker_info: dict, financials_history: pd.DataFrame, growth_rate: float) -> dict:
        """
        Calculates Graham Number (classic) and Revised Graham Formula.
        growth_rate is expected as decimal, e.g., 0.08 for 8% growth.
        """
        eps = ticker_info.get("trailingEps")
        if eps is None or pd.isna(eps):
            eps = ticker_info.get("forwardEps")
            
        # Get Book Value per Share
        bvps = ticker_info.get("bookValue")
        
        # Fallback for BVPS if missing: Equity / Shares Outstanding
        if (bvps is None or pd.isna(bvps)) and not financials_history.empty:
            latest = financials_history.iloc[-1]
            equity = latest.get("equity", 0.0)
            shares = ticker_info.get("sharesOutstanding")
            if equity and shares and shares > 0:
                bvps = equity / shares

        results = {
            "classic_graham_price": np.nan,
            "revised_graham_price": np.nan,
            "eps": eps,
            "bvps": bvps,
            "g": growth_rate * 100.0, # as percentage for the formula
            "bond_yield": self.bond_yield
        }

        # 1. Classic Graham Number: sqrt(22.5 * EPS * BVPS)
        if eps and bvps and eps > 0 and bvps > 0:
            results["classic_graham_price"] = np.sqrt(22.5 * eps * bvps)

        # 2. Revised Graham Formula: V = (EPS * (8.5 + 2g) * 4.4) / Y
        # We clamp growth g to be non-negative and cap it at 20% to avoid valuation explosions
        g_pct = max(min(growth_rate * 100.0, 20.0), 0.0)
        # Standard revised formula uses 4.4 as AAA yield baseline
        if eps and eps > 0:
            results["revised_graham_price"] = (eps * (8.5 + 2 * g_pct) * 4.4) / self.bond_yield

        return results
