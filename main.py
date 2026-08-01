import argparse
import sys
import os
import pandas as pd
import numpy as np
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

# Import our valuation package
from valuation.data_provider import DataProvider
from valuation.wacc import WACCCalculator
from valuation.dcf import DCFModel
from valuation.multiples import MultiplesModel
from valuation.graham import GrahamModel
from valuation.ddm import DividendDiscountModel
from valuation.calibration import CalibrationEngine
from valuation.projections import ProjectionEngine
from valuation.nlg_analyst import NLGAnalyst


# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("main")
console = Console()

def run_valuation(
    ticker_symbol: str,
    custom_growth: float = None,
    custom_terminal_growth: float = 0.02,
    custom_mrp: float = 0.055,
    custom_peers: list = None
) -> dict:
    """
    Orchestrates the entire stock valuation pipeline.
    """
    ticker_symbol = ticker_symbol.strip().upper()
    console.print(f"[bold blue]Initializing valuation pipeline for {ticker_symbol}...[/bold blue]")
    
    # 1. Fetch data
    dp = DataProvider()
    ticker = dp.get_stock_data(ticker_symbol)
    
    info = ticker.info
    if not info or "shortName" not in info:
        console.print(f"[bold red]Error: Could not retrieve ticker information for {ticker_symbol}. Please check the symbol.[/bold red]")
        sys.exit(1)
        
    current_price = info.get("currentPrice") or info.get("previousClose") or 1.0
    financials_history = dp.extract_financials_history(ticker)
    if financials_history.empty:
        console.print(f"[bold red]Error: Financial statements history is completely empty for {ticker_symbol}. Cannot proceed.[/bold red]")
        sys.exit(1)
        
    # 2. Get baseline growth rate
    base_growth = 0.05 # default fallback
    if custom_growth is not None:
        base_growth = custom_growth
        console.print(f"Using user-specified baseline growth rate: {base_growth*100:.2f}%")
    else:
        # Estimate growth from historical revenue
        rev_history = financials_history["revenue"].dropna()
        if len(rev_history) >= 2:
            # Calculate Year-over-Year growth rates and take average
            growth_rates = rev_history.pct_change().dropna()
            if not growth_rates.empty:
                base_growth = float(growth_rates.mean())
                # Clean extreme growth rates (e.g. cap at 30%, floor at -10%)
                base_growth = max(min(base_growth, 0.30), -0.10)
                console.print(f"Estimated baseline revenue growth rate from history: {base_growth*100:.2f}%")
        # Try to overlay analyst growth estimates if available
        analyst_growth = info.get("earningsGrowth") or info.get("revenueGrowth")
        if analyst_growth and not pd.isna(analyst_growth):
            # blend historical and analyst estimates (weight analyst estimates more since forward looking)
            base_growth = 0.3 * base_growth + 0.7 * float(analyst_growth)
            base_growth = max(min(base_growth, 0.30), -0.10)
            console.print(f"Incorporated analyst growth estimates. Blended baseline growth rate: {base_growth*100:.2f}%")

    # 3. Get risk-free rate and calculate WACC
    rf = dp.get_risk_free_rate()
    wacc_calc = WACCCalculator(risk_free_rate=rf, market_risk_premium=custom_mrp)
    wacc_details = wacc_calc.calculate(info, financials_history)
    base_wacc = wacc_details["wacc"]
    
    # 4. Calibrate parameters via Machine Learning / Backtesting
    cal_engine = CalibrationEngine(dp)
    calibration_results = cal_engine.calibrate(
        ticker_symbol, financials_history, info, base_growth, base_wacc
    )
    
    # 5. Apply calibrated parameters
    cal_growth = base_growth * calibration_results["growth_multiplier"]
    cal_wacc = base_wacc + calibration_results["wacc_offset"]
    
    console.print(f"Calibrated growth rate: [bold green]{cal_growth*100:.2f}%[/bold green] (vs baseline {base_growth*100:.2f}%)")
    console.print(f"Calibrated WACC: [bold green]{cal_wacc*100:.2f}%[/bold green] (vs baseline {base_wacc*100:.2f}%)")
    
    # Update WACC details with calibrated value
    wacc_details["wacc"] = cal_wacc
    
    # 6. Run DCF Model
    dcf_model = DCFModel()
    dcf_results = dcf_model.calculate(
        financials_history,
        wacc_details,
        base_growth_rate=cal_growth,
        base_terminal_growth=custom_terminal_growth,
        shares_outstanding=info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"),
        current_price=info.get("currentPrice") or info.get("previousClose")
    )
    
    if "error" in dcf_results and calibration_results.get("profile") != "ETF":
        console.print(f"[bold red]DCF Calculation Error: {dcf_results['error']}[/bold red]")
        sys.exit(1)

    # 7. Peer Multiples Valuation
    if custom_peers:
        peer_tickers = [p.strip().upper() for p in custom_peers]
    else:
        # Auto-detect peers
        peer_tickers = dp.get_peer_tickers(ticker_symbol)
        
    console.print(f"Auto-selected peer group: {', '.join(peer_tickers)}")
    multiples_model = MultiplesModel()
    multiples_results = multiples_model.calculate(ticker_symbol, peer_tickers, info, financials_history)
    
    # 8. Benjamin Graham Valuation
    graham_model = GrahamModel(risk_free_rate=rf)
    graham_results = graham_model.calculate(info, financials_history, growth_rate=cal_growth)
    
    # 9. Dividend Discount Model
    ddm_model = DividendDiscountModel()
    ddm_results = ddm_model.calculate(info, wacc_details["cost_of_equity"], growth_rate=cal_growth, terminal_growth_rate=custom_terminal_growth)
    
    # 10. Projections & Trajectories (1m, 3m, 6m, 12m)
    # Get 12-month base, bear, bull intrinsic values using calibrated weights
    dcf_wt = calibration_results.get("dcf_weight", 0.35)
    graham_wt = calibration_results.get("graham_weight", 0.30)
    multiples_wt = calibration_results.get("multiples_weight", 0.25)
    ddm_wt = calibration_results.get("ddm_weight", 0.10)
    
    dcf_base_val = dcf_results["base"]["target_price"] if ("base" in dcf_results and "target_price" in dcf_results["base"]) else np.nan
    dcf_bear_val = dcf_results["bear"]["target_price"] if ("bear" in dcf_results and "target_price" in dcf_results["bear"]) else np.nan
    dcf_bull_val = dcf_results["bull"]["target_price"] if ("bull" in dcf_results and "target_price" in dcf_results["bull"]) else np.nan

    dcf_val = dcf_base_val
    if not pd.isna(dcf_val) and dcf_val < 0.20 * current_price and multiples_wt > 0:
        shift = dcf_wt * 0.5
        dcf_wt -= shift
        multiples_wt += shift

    def get_fully_weighted_target(dcf_p, graham_p, multiples_p, ddm_p):
        valid_vals = []
        wts = []
        if not pd.isna(dcf_p) and dcf_p > 0:
            valid_vals.append(dcf_p)
            wts.append(dcf_wt)
        if not pd.isna(graham_p) and graham_p > 0:
            valid_vals.append(graham_p)
            wts.append(graham_wt)
        if not pd.isna(multiples_p) and multiples_p > 0:
            valid_vals.append(multiples_p)
            wts.append(multiples_wt)
        if not pd.isna(ddm_p) and ddm_p > 0:
            valid_vals.append(ddm_p)
            wts.append(ddm_wt)
            
        sum_w = sum(wts)
        if sum_w > 0:
            return sum(v * (w / sum_w) for v, w in zip(valid_vals, wts))
        return current_price

    graham_base = graham_results.get("revised_graham_price")
    if pd.isna(graham_base):
        graham_base = graham_results.get("classic_graham_price")
        
    multiples_base = multiples_results["implied_prices"].get("pe_forward")
    if pd.isna(multiples_base) or multiples_base <= 0:
        multiples_base = multiples_results["implied_prices"].get("ev_ebitda")
    if pd.isna(multiples_base) or multiples_base <= 0:
        multiples_base = multiples_results["implied_prices"].get("pe_trailing")
    if pd.isna(multiples_base) or multiples_base <= 0:
        multiples_base = multiples_results["implied_prices"].get("ev_revenue")
        
    ddm_base = ddm_results.get("two_stage_ddm_price")
    if pd.isna(ddm_base):
        ddm_base = ddm_results.get("gordon_growth_price")

    target_12m = {
        "base": get_fully_weighted_target(
            dcf_base_val,
            graham_base,
            multiples_base,
            ddm_base
        ),
        "bear": get_fully_weighted_target(
            dcf_bear_val,
            graham_base * 0.7 if not pd.isna(graham_base) else np.nan,
            multiples_base * 0.8 if not pd.isna(multiples_base) else np.nan,
            ddm_base * 0.8 if not pd.isna(ddm_base) else np.nan
        ),
        "bull": get_fully_weighted_target(
            dcf_bull_val,
            graham_base * 1.3 if not pd.isna(graham_base) else np.nan,
            multiples_base * 1.2 if not pd.isna(multiples_base) else np.nan,
            ddm_base * 1.2 if not pd.isna(ddm_base) else np.nan
        )
    }
    
    proj_engine = ProjectionEngine()
    trajectories = proj_engine.calculate_trajectories(current_price, target_12m, ticker_symbol)
    
    # 11. NLG Analyst report
    analyst = NLGAnalyst()
    report = analyst.generate_report(
        ticker_symbol, info, financials_history, wacc_details, dcf_results,
        multiples_results, graham_results, ddm_results, calibration_results, trajectories
    )
    
    return {
        "info": info,
        "wacc": wacc_details,
        "dcf": dcf_results,
        "multiples": multiples_results,
        "graham": graham_results,
        "ddm": ddm_results,
        "calibration": calibration_results,
        "trajectories": trajectories,
        "report": report
    }

def print_cli_report(results: dict):
    """
    Renders the valuation report nicely in the command line using rich.
    """
    info = results["info"]
    report = results["report"]
    trajectories = results["trajectories"]
    
    company_name = info.get("shortName", "N/A")
    symbol = info.get("symbol", "N/A")
    current_price = info.get("currentPrice") or info.get("previousClose") or 0.0
    
    # Header Panel
    console.print("\n")
    rec_text = f"[bold white on green] {report['recommendation']} [/bold white on green]" if "BUY" in report['recommendation'] else f"[bold white on red] {report['recommendation']} [/bold white on red]" if "SELL" in report['recommendation'] else f"[bold black on yellow] {report['recommendation']} [/bold black on yellow]"
    
    panel_title = f"{company_name} ({symbol}) - Valuation Report"
    panel_content = (
        f"[bold]Sector:[/bold] {info.get('sector', 'N/A')}  |  [bold]Industry:[/bold] {info.get('industry', 'N/A')}\n"
        f"[bold]Current Price:[/bold] ${current_price:.2f}  |  [bold]Target Intrinsic Value:[/bold] ${report['target_price']:.2f}\n"
        f"[bold]Margin of Safety:[/bold] {report['margin_of_safety']*100:+.2f}%  |  [bold]Recommendation:[/bold] {rec_text}"
    )
    console.print(Panel(panel_content, title=panel_title, border_style="blue", expand=False))
    
    # 1. Methodology Table
    table_models = Table(title="Valuation Methodologies Summary", header_style="bold blue")
    table_models.add_column("Valuation Model", style="bold")
    table_models.add_column("Target Price", justify="right")
    table_models.add_column("Discount/Premium", justify="right")
    
    def add_row_helper(name, val):
        if pd.isna(val) or val is None or val <= 0:
            table_models.add_row(name, "N/A", "N/A")
        else:
            diff = (val - current_price) / current_price
            table_models.add_row(name, f"${val:.2f}", f"{diff*100:+.1f}%")
            
    dcf_base = results["dcf"].get("base", {}).get("target_price", np.nan) if "base" in results["dcf"] else np.nan
    add_row_helper("Discounted Cash Flow (DCF) Base Case", dcf_base)
    add_row_helper("Comparable Company EV/EBITDA", results["multiples"]["implied_prices"].get("ev_ebitda"))
    add_row_helper("Comparable Company Trailing P/E", results["multiples"]["implied_prices"].get("pe_trailing"))
    add_row_helper("Revised Benjamin Graham Formula", results["graham"].get("revised_graham_price"))
    add_row_helper("Dividend Discount Model (2-Stage)", results["ddm"].get("two_stage_ddm_price"))
    add_row_helper("OPTIMIZED WEIGHTED TARGET PRICE", report["target_price"])
    
    console.print(table_models)
    
    # 2. Scenarios and Horizons Table
    table_horizons = Table(title="Price Target Projections by Scenario & Horizon", header_style="bold blue")
    table_horizons.add_column("Scenario / Horizon", style="bold")
    table_horizons.add_column("1 Month", justify="right")
    table_horizons.add_column("3 Months", justify="right")
    table_horizons.add_column("6 Months", justify="right")
    table_horizons.add_column("12 Months (Intrinsic)", justify="right")
    
    for scen in ["bear", "base", "bull"]:
        name = "Bear Case (Downside)" if scen == "bear" else "Base Case (Expected)" if scen == "base" else "Bull Case (Upside)"
        table_horizons.add_row(
            name,
            f"${trajectories[scen][1]:.2f}",
            f"${trajectories[scen][3]:.2f}",
            f"${trajectories[scen][6]:.2f}",
            f"${trajectories[scen][12]:.2f}"
        )
    console.print(table_horizons)
    
    # 3. Analyst Narrative
    console.print("\n[bold blue]FUNDAMENTAL ANALYST NARRATIVE & INSIGHTS[/bold blue]")
    console.print(f"[bold]1. Executive Summary:[/bold]\n{report['overview']}\n")
    console.print(f"[bold]2. Financial Statements Analysis:[/bold]\n{report['financials_analysis']}\n")
    console.print(f"[bold]3. Valuation Methodology & ML Calibration:[/bold]\n{report['valuation_analysis']}\n")
    console.print(f"[bold]4. Scenario Forecasts & Catalyst Analysis:[/bold]\n{report['scenarios_analysis']}\n")
    console.print(Panel(report["action_plan"], title="Actionable Advice & Action Plan", border_style="green"))

def print_cli_options(ticker_symbol: str, quote_basis: str = "auto"):
    """Display phase-one cash-secured short-put rankings on the CLI."""
    from valuation.option_scoring import ShortPutPreferences
    from valuation.options import OptionsAnalyzer

    symbol = ticker_symbol.strip().upper()
    console.print(f"\n[bold blue]Scanning cash-secured {symbol} puts...[/bold blue]")
    preferences = ShortPutPreferences.for_profile("lowest_risk", limit=5)
    provider = DataProvider()
    risk_free_rate = 0.0425 if quote_basis == "previous_session" else provider.get_risk_free_rate()
    result = OptionsAnalyzer(provider).recommend_short_puts(
        symbol,
        preferences,
        r_rate=risk_free_rate,
        quote_basis=quote_basis,
    )
    if result.get("error"):
        console.print(f"[bold red]{result['error']}[/bold red]")
        return

    iv_context = result.get("iv_context", {})
    iv_rank = iv_context.get("iv_rank")
    iv_rank_text = f"{iv_rank * 100:.0f}/100" if iv_rank is not None else "unavailable"
    console.print(
        f"Spot [bold]${result['spot']:,.2f}[/bold] | IV rank [bold]{iv_rank_text}[/bold] "
        f"({iv_context.get('source', 'no source')}) | "
        f"{result.get('eligible_before_limit', 0)} of {result.get('input_count', 0)} contracts eligible"
    )
    console.print(
        f"Underlying source: {result.get('spot_source', 'unknown')} | "
        f"session {result.get('spot_session', 'unknown')} | "
        f"as of {result.get('spot_as_of') or result.get('spot_retrieved_at', 'unknown')} | "
        f"market state {result.get('market_state', 'unknown')}"
    )
    console.print(f"Quote basis used: {result.get('quote_basis_used', 'unknown')}")
    if result.get("spot_warning"):
        console.print(f"[yellow]{result['spot_warning']}[/yellow]")
    delta_range = result.get("constraints", {}).get("short_delta", [])
    if len(delta_range) == 2:
        console.print(
            f"Short-delta profile: target {result['constraints']['target_short_delta']:.2f}, "
            f"eligible range {delta_range[0]:.2f}-{delta_range[1]:.2f}; assignment setting is a ceiling."
        )

    recommendations = result.get("recommendations", [])
    if not recommendations:
        if result.get("data_readiness") == "no_marketable_otm_puts":
            console.print(
                f"[yellow]No marketable OTM put quotes were available "
                f"({result.get('marketable_put_count', 0)} marketable puts, "
                f"{result.get('marketable_otm_put_count', 0)} OTM). "
                "Risk guardrails were not reached.[/yellow]"
            )
        else:
            console.print("[yellow]No put passed every configured risk and liquidity limit.[/yellow]")
        for reason, count in result.get("rejected_counts", {}).items():
            console.print(f"  {count:>4}  {reason}")
        return

    table = Table(title="Cash-secured short puts · lowest-risk profile", header_style="bold cyan")
    table.add_column("Rank", justify="right")
    table.add_column("Expiry")
    table.add_column("DTE", justify="right")
    table.add_column("Strike", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Credit", justify="right")
    table.add_column("Short delta", justify="right")
    table.add_column("Assign", justify="right")
    table.add_column("Ann. yield", justify="right")
    for recommendation in recommendations:
        table.add_row(
            f"#{recommendation['rank']}",
            recommendation["expiration"],
            str(recommendation["dte"]),
            f"${recommendation['strike']:,.2f}",
            f"{recommendation['score']:.1f}",
            f"${recommendation['premium_per_contract']:,.0f}",
            f"{recommendation['short_position_delta']:.3f}",
            f"{recommendation['estimated_assignment_probability'] * 100:.1f}%",
            f"{recommendation['annualized_return_on_cash'] * 100:.1f}%",
        )
    console.print(table)
    best = recommendations[0]
    console.print(
        f"Top contract details: break-even [bold]${best['break_even']:,.2f}[/bold], "
        f"cash required [bold]${best['cash_secured']:,.0f}[/bold], IV {best['implied_volatility'] * 100:.1f}%, "
        f"spread {best['bid_ask_spread_pct'] * 100:.1f}%, OI {best['open_interest']:,}."
    )
    console.print(
        "[dim]Assignment proxy is the Black-Scholes risk-neutral probability of expiring ITM, "
        "not an exact assignment probability. American-style puts can be assigned early.[/dim]"
    )


def print_cli_bullish_combo(ticker_symbol: str, quote_basis: str = "auto"):
    """Display premium-funded short-put/long-call pair rankings."""
    from valuation.option_scoring import BullishComboPreferences, ShortPutPreferences
    from valuation.options import OptionsAnalyzer

    symbol = ticker_symbol.strip().upper()
    console.print(f"\n[bold blue]Scanning premium-funded bullish {symbol} pairs...[/bold blue]")
    put_preferences = ShortPutPreferences.for_profile(
        "balanced",
        max_cash_secured=None,
        limit=5,
    )
    combo_preferences = BullishComboPreferences.for_profile(
        "balanced",
        max_extra_debit=0.0,
        limit=5,
    )
    provider = DataProvider()
    risk_free_rate = 0.0425 if quote_basis == "previous_session" else provider.get_risk_free_rate()
    result = OptionsAnalyzer(provider).recommend_premium_funded_bullish_combo(
        symbol,
        put_preferences,
        combo_preferences,
        r_rate=risk_free_rate,
        quote_basis=quote_basis,
    )
    if result.get("error"):
        console.print(f"[bold red]{result['error']}[/bold red]")
        return

    console.print(
        f"Spot [bold]${result['spot']:,.2f}[/bold] | "
        f"{result.get('eligible_before_limit', 0)} of {result.get('pair_input_count', 0)} pair combinations eligible | "
        "put bid funds call ask"
    )
    console.print(
        f"Underlying source: {result.get('spot_source', 'unknown')} | "
        f"session {result.get('spot_session', 'unknown')} | "
        f"as of {result.get('spot_as_of') or result.get('spot_retrieved_at', 'unknown')} | "
        f"market state {result.get('market_state', 'unknown')}"
    )
    console.print(f"Quote basis used: {result.get('quote_basis_used', 'unknown')}")
    if result.get("spot_warning"):
        console.print(f"[yellow]{result['spot_warning']}[/yellow]")
    recommendations = result.get("recommendations", [])
    if not recommendations:
        if result.get("data_readiness") == "no_marketable_otm_puts":
            console.print("[yellow]No marketable OTM short-put quotes were available; pair limits were not reached.[/yellow]")
        elif result.get("data_readiness") == "no_marketable_otm_calls":
            console.print("[yellow]No marketable OTM long-call quotes were available; pair scoring could not run.[/yellow]")
        else:
            console.print("[yellow]No pair passed every downside, liquidity, call-delta, and funding limit.[/yellow]")
        return

    table = Table(title="Premium-funded bullish risk reversals", header_style="bold cyan")
    table.add_column("Rank", justify="right")
    table.add_column("Expiry")
    table.add_column("DTE", justify="right")
    table.add_column("Put / Call", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Net", justify="right")
    table.add_column("Put delta", justify="right")
    table.add_column("Put assign", justify="right")
    table.add_column("Call delta", justify="right")
    table.add_column("+10% P&L", justify="right")
    for recommendation in recommendations:
        net_text = (
            f"${recommendation['net_credit']:,.0f} cr"
            if recommendation["net_credit"] >= 0
            else f"${recommendation['net_debit']:,.0f} dr"
        )
        table.add_row(
            f"#{recommendation['rank']}",
            recommendation["expiration"],
            str(recommendation["dte"]),
            f"{recommendation['put_strike']:.0f}P / {recommendation['call_strike']:.0f}C",
            f"{recommendation['score']:.1f}",
            net_text,
            f"{recommendation['short_put_delta']:.3f}",
            f"{recommendation['estimated_assignment_probability'] * 100:.1f}%",
            f"{recommendation['long_call_delta']:.2f}",
            f"${recommendation['profit_at_up_10_pct']:,.0f}",
        )
    console.print(table)
    best = recommendations[0]
    console.print(
        f"Top pair: sell {best['put_strike']:.0f} put for ${best['put_credit']:,.0f}; "
        f"buy {best['call_strike']:.0f} call for ${best['call_cost']:,.0f}. "
        f"Cash-secured downside to zero: [bold]${best['max_loss']:,.0f}[/bold]."
    )
    console.print(
        "[dim]This bullish risk reversal has unlimited upside but substantial short-put downside. "
        "The +/- scenario is an expiration payoff, not a forecast.[/dim]"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="US Stocks Valuation Model CLI Tool")
    parser.add_argument("--ticker", required=True, help="US stock ticker symbol (e.g. AAPL, MSFT, KO)")
    parser.add_argument("--growth", type=float, default=None, help="Custom revenue growth rate (e.g. 0.12 for 12%%)")
    parser.add_argument("--terminal-growth", type=float, default=0.02, help="Custom terminal growth rate (default: 2%%)")
    parser.add_argument("--mrp", type=float, default=0.055, help="Market Risk Premium (default: 5.5%%)")
    parser.add_argument("--peers", type=str, default=None, help="Comma-separated custom peer tickers")
    parser.add_argument("--options", action="store_true", help="Rank cash-secured short puts for the ticker")
    parser.add_argument("--bull-combo", action="store_true", help="Rank put-credit-funded long-call pairs")
    parser.add_argument(
        "--quote-basis",
        choices=("auto", "live", "previous_session"),
        default="auto",
        help="Option data basis: auto, live, or the latest saved regular-session snapshot",
    )
    
    args = parser.parse_args()
    
    peers_list = None
    if args.peers:
        peers_list = [p.strip() for p in args.peers.split(",")]
        
    try:
        if args.options or args.bull_combo:
            if args.options:
                print_cli_options(args.ticker, args.quote_basis)
            if args.bull_combo:
                print_cli_bullish_combo(args.ticker, args.quote_basis)
        else:
            results = run_valuation(
                ticker_symbol=args.ticker,
                custom_growth=args.growth,
                custom_terminal_growth=args.terminal_growth,
                custom_mrp=args.mrp,
                custom_peers=peers_list
            )
            print_cli_report(results)

    except Exception as e:
        console.print(f"[bold red]Execution error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
