import pandas as pd
import numpy as np

def format_large_val(val: float, language: str = "en") -> str:
    if pd.isna(val) or val is None or not np.isfinite(val):
        return "N/A"
    sign = "-" if val < 0 else ""
    abs_val = abs(val)
    if language == "zh":
        if abs_val >= 1e12:
            return f"{sign}${abs_val/1e12:,.2f}万亿"
        elif abs_val >= 1e8:
            return f"{sign}${abs_val/1e8:,.2f}亿"
        elif abs_val >= 1e4:
            return f"{sign}${abs_val/1e4:,.2f}万"
        else:
            return f"{sign}${abs_val:,.2f}"
    else:
        if abs_val >= 1e12:
            return f"{sign}${abs_val/1e12:,.2f}T"
        elif abs_val >= 1e9:
            return f"{sign}${abs_val/1e9:,.2f}B"
        elif abs_val >= 1e6:
            return f"{sign}${abs_val/1e6:,.2f}M"
        else:
            return f"{sign}${abs_val:,.2f}"


class NLGAnalyst:
    def generate_report(
        self,
        ticker_symbol: str,
        ticker_info: dict,
        financials_history: pd.DataFrame,
        wacc_details: dict,
        dcf_results: dict,
        multiples_results: dict,
        graham_results: dict,
        ddm_results: dict,
        calibration_results: dict,
        trajectories: dict,
        language: str = "en"
    ) -> dict:
        """
        Generates a dynamic, highly contextual investment report with a professional analyst narrative in English or Chinese.
        """
        company_name = ticker_info.get("shortName", ticker_symbol)
        
        current_price = ticker_info.get("currentPrice") or ticker_info.get("previousClose") or 1.0
        
        profile_code = calibration_results.get("profile", "STANDARD")
        if profile_code == "ETF":
            # Return custom ETF report
            if language == "zh":
                profile_name = "交易所交易基金 (ETF)"
                recommendation = "持有 (HOLD)"
                overview_para = (
                    f"经过分析，{company_name} ({ticker_symbol}) 目前的价格为 ${current_price:.2f}。作为一个交易所交易基金 (ETF)，"
                    f"它代表了一揽子多元化的底层证券。诸如现金流折现 (DCF)、葛兰姆公式、同行乘数对比及股利折现模型 (DDM) 等企业估值模型均不适用。"
                    f"由于其价格通常紧密跟踪其成份股的净资产价值 (NAV)，模型对该基金发布“持有”评级。"
                )
                financials_para = (
                    f"作为交易所交易基金，{company_name} 不发布标准的净利润、营收或企业债务等财务报表。其表现直接取决于其底层持仓证券的集体财务状况和价格走势。"
                )
                valuation_para = (
                    f"此资产类别不适用企业估值方法。所有估值模型（DCF、葛兰姆、同行乘数、DDM）的权重均设为 0%。投资者应基于其管理费率、跟踪误差、流动性以及行业/地理风险敞口对其进行评估。"
                )
                scenarios_para = (
                    f"我们基于市场 Beta 和历史价格走势（而非企业增长率）进行情景展望。在基准情景下，价格预计将围绕 ${current_price:.2f} 跟踪当前市场走势。"
                    f"牛市和熊市情景将完全取决于该 ETF 所跟踪的底层指数或板块的表现。"
                )
                conclusion_para = (
                    f"**操作建议**: 建议对 {ticker_symbol} 采取 [ 继续持有 ] 策略。作为一只 ETF，它应当作为多元化资产配置的一部分进行持有。"
                    f"应密切关注指数成份股调整、组合重平衡、管理费率及宏观市场趋势，而非单一企业基本面。"
                )
            else:
                profile_name = "Exchange Traded Fund (ETF)"
                recommendation = "HOLD"
                overview_para = (
                    f"Following an analysis of the ETF, {company_name} ({ticker_symbol}) is currently priced at ${current_price:.2f}. "
                    f"As an Exchange Traded Fund (ETF), it represents a diversified basket of underlying securities. Corporate fundamental models such as "
                    f"DCF, Graham Formula, Peer Multiples, and DDM do not apply. We default to a HOLD recommendation as the price generally tracks the "
                    f"Net Asset Value (NAV) of its constituents."
                )
                financials_para = (
                    f"As an Exchange Traded Fund, {company_name} does not publish standard corporate financials like revenue, EBITDA, or corporate debt. "
                    f"Its performance is directly tied to the collective financial health and price action of its underlying holdings."
                )
                valuation_para = (
                    f"Corporate valuation methodologies are inactive for this security type. The weight distribution is set to 0% across all models "
                    f"(DCF, Graham, Multiples, DDM). Investors should evaluate this ETF based on its expense ratio, tracking error, liquidity, and sectoral/geographical exposures."
                )
                scenarios_para = (
                    f"We map scenarios based on market beta and price history rather than corporate growth rates. Under the base case scenario, "
                    f"the price is expected to track current market trends around ${current_price:.2f}. Bull and bear case scenarios will depend entirely "
                    f"on the performance of the underlying index or sectors tracked by the ETF."
                )
                conclusion_para = (
                    f"**Actionable Advice**: Hold positions on {ticker_symbol}. Since this is an ETF, it should be held as part of a diversified portfolio "
                    f"asset allocation. Monitor index constituents, portfolio rebalancing, expense ratio, and overall market trend rather than individual company fundamentals."
                )

            return {
                "recommendation": recommendation,
                "color": "yellow",
                "margin_of_safety": 0.0,
                "target_price": current_price,
                "overview": overview_para,
                "financials_analysis": financials_para,
                "valuation_analysis": valuation_para,
                "scenarios_analysis": scenarios_para,
                "action_plan": conclusion_para,
                "profile": profile_code,
                "profile_name": profile_name,
                "dcf_weight": 0.0,
                "graham_weight": 0.0,
                "multiples_weight": 0.0,
                "ddm_weight": 0.0
            }

        # 1. Determine final target price (weighted based on calibration)
        dcf_val = dcf_results["base"]["target_price"]
        graham_val = graham_results["revised_graham_price"]
        if pd.isna(graham_val):
            graham_val = graham_results["classic_graham_price"]
            
        # Prioritize forward PE for multiples valuation to capture current market growth expectations
        multiples_val = multiples_results["implied_prices"].get("pe_forward")
        if pd.isna(multiples_val) or multiples_val <= 0:
            multiples_val = multiples_results["implied_prices"].get("ev_ebitda")
        if pd.isna(multiples_val) or multiples_val <= 0:
            multiples_val = multiples_results["implied_prices"].get("pe_trailing")
        if pd.isna(multiples_val) or multiples_val <= 0:
            multiples_val = multiples_results["implied_prices"].get("ev_revenue")
            
        ddm_val = ddm_results["two_stage_ddm_price"]
        if pd.isna(ddm_val):
            ddm_val = ddm_results["gordon_growth_price"]

        # Get the optimized weights directly from calibration results
        dcf_wt = calibration_results.get("dcf_weight", 0.35)
        graham_wt = calibration_results.get("graham_weight", 0.30)
        multiples_wt = calibration_results.get("multiples_weight", 0.25)
        ddm_wt = calibration_results.get("ddm_weight", 0.10)
        
        # Heuristic: If DCF is extremely depressed relative to current price (e.g. < 20%),
        # and multiples is valid, we shift 50% of the DCF weight to multiples (market reference).
        if not pd.isna(dcf_val) and dcf_val < 0.20 * current_price and multiples_wt > 0:
            shift = dcf_wt * 0.5
            dcf_wt -= shift
            multiples_wt += shift
            
        valid_vals = []
        weights = []
        
        if not pd.isna(dcf_val) and dcf_val > 0:
            valid_vals.append(dcf_val)
            weights.append(dcf_wt)
        if not pd.isna(graham_val) and graham_val > 0:
            valid_vals.append(graham_val)
            weights.append(graham_wt)
        if not pd.isna(multiples_val) and multiples_val > 0:
            valid_vals.append(multiples_val)
            weights.append(multiples_wt)
        if not pd.isna(ddm_val) and ddm_val > 0:
            valid_vals.append(ddm_val)
            weights.append(ddm_wt)
            
        # Normalize weights
        sum_w = sum(weights)
        if sum_w > 0:
            weights = [w / sum_w for w in weights]
            weighted_target = sum(v * w for v, w in zip(valid_vals, weights))
        else:
            weighted_target = dcf_val or current_price

        # Keep track of final weights for display/narrative
        final_dcf_w = 0.0
        final_graham_w = 0.0
        final_multiples_w = 0.0
        final_ddm_w = 0.0
        
        if sum_w > 0:
            idx = 0
            if not pd.isna(dcf_val) and dcf_val > 0:
                final_dcf_w = weights[idx]
                idx += 1
            if not pd.isna(graham_val) and graham_val > 0:
                final_graham_w = weights[idx]
                idx += 1
            if not pd.isna(multiples_val) and multiples_val > 0:
                final_multiples_w = weights[idx]
                idx += 1
            if not pd.isna(ddm_val) and ddm_val > 0:
                final_ddm_w = weights[idx]
                idx += 1
        else:
            if not pd.isna(dcf_val) and dcf_val > 0:
                final_dcf_w = 1.0
            else:
                final_multiples_w = 1.0

        profile_code = calibration_results.get("profile", "STANDARD")
        profile_names_en = {
            "FINANCIAL": "Financial / Banking Institution",
            "HIGH_GROWTH_AI": "High-Growth / AI Sector Leader",
            "CYCLICAL": "Cyclical / Asset-Heavy Business",
            "CASH_COW_MATURE": "Stable Cash Cow / High Yield Mature",
            "STANDARD": "Standard Corporate Profile"
        }
        profile_names_zh = {
            "FINANCIAL": "金融与银行机构",
            "HIGH_GROWTH_AI": "高成长与AI领域领航者",
            "CYCLICAL": "周期性与重资产企业",
            "CASH_COW_MATURE": "稳定现金流与高分红成熟企业",
            "STANDARD": "标准企业估值剖面"
        }
        profile_name = profile_names_zh.get(profile_code, "标准企业估值剖面") if language == "zh" else profile_names_en.get(profile_code, "Standard Corporate Profile")

        # Recommendation and margin of safety
        margin_of_safety = (weighted_target - current_price) / current_price
        
        # Determine Recommendation Terminology
        if language == "zh":
            if margin_of_safety > 0.20:
                recommendation = "强力买入 (STRONG BUY)"
                rec_simple = "强力买入"
                valuation_status = "被显著低估"
                upside_text = "上涨空间"
            elif margin_of_safety > 0.05:
                recommendation = "买入 (BUY)"
                rec_simple = "买入"
                valuation_status = "被适度低估"
                upside_text = "上涨空间"
            elif margin_of_safety >= -0.05:
                recommendation = "持有 (HOLD)"
                rec_simple = "持有"
                valuation_status = "估值合理"
                upside_text = "上涨空间"
            elif margin_of_safety >= -0.20:
                recommendation = "卖出 (SELL)"
                rec_simple = "卖出"
                valuation_status = "被适度高估"
                upside_text = "下跌风险"
            else:
                recommendation = "强力卖出 (STRONG SELL)"
                rec_simple = "强力卖出"
                valuation_status = "被显著高估"
                upside_text = "下跌风险"
        else:
            if margin_of_safety > 0.20:
                recommendation = "STRONG BUY"
                rec_simple = "STRONG BUY"
                valuation_status = "undervalued"
                upside_text = "upside potential"
            elif margin_of_safety > 0.05:
                recommendation = "BUY"
                rec_simple = "BUY"
                valuation_status = "moderately undervalued"
                upside_text = "upside potential"
            elif margin_of_safety >= -0.05:
                recommendation = "HOLD"
                rec_simple = "HOLD"
                valuation_status = "fairly valued"
                upside_text = "upside potential"
            elif margin_of_safety >= -0.20:
                recommendation = "SELL"
                rec_simple = "SELL"
                valuation_status = "moderately overvalued"
                upside_text = "downside risk"
            else:
                recommendation = "STRONG SELL"
                rec_simple = "STRONG SELL"
                valuation_status = "significantly overvalued"
                upside_text = "downside risk"

        rec_color = "green" if "BUY" in rec_simple or "买" in rec_simple else "red" if "SELL" in rec_simple or "卖" in rec_simple else "yellow"

        # 2. Financial trends analysis
        rev_trend = ""
        margin_trend = ""
        fcf_trend = ""
        solvency_trend = ""
        
        if len(financials_history) >= 2:
            rev_history = financials_history["revenue"].dropna()
            if len(rev_history) >= 2:
                growth_rate = rev_history.pct_change().iloc[-1]
                if language == "zh":
                    if growth_rate > 0.15:
                        rev_trend = f"展现出同比 {growth_rate*100:.1f}% 的营收快速扩张"
                    elif growth_rate > 0.03:
                        rev_trend = f"展现出同比 {growth_rate*100:.1f}% 的稳定营收增长"
                    elif growth_rate >= -0.03:
                        rev_trend = f"营收基本持平，同比增长仅为 {growth_rate*100:.1f}%"
                    else:
                        rev_trend = f"面临同比 {growth_rate*100:.1f}% 的营收萎缩压力"
                else:
                    if growth_rate > 0.15:
                        rev_trend = f"demonstrating rapid revenue expansion of {growth_rate*100:.1f}% year-over-year"
                    elif growth_rate > 0.03:
                        rev_trend = f"showing steady topline growth of {growth_rate*100:.1f}% year-over-year"
                    elif growth_rate >= -0.03:
                        rev_trend = f"experiencing flat topline trajectory ({growth_rate*100:.1f}%)"
                    else:
                        rev_trend = f"facing a topline contraction of {growth_rate*100:.1f}% year-over-year"
                    
            ebitda_history = financials_history["ebitda"].dropna()
            rev_history_aligned = financials_history.loc[ebitda_history.index, "revenue"]
            margins = ebitda_history / rev_history_aligned
            if len(margins) >= 2:
                latest_m = margins.iloc[-1]
                prev_m = margins.iloc[-2]
                m_diff = latest_m - prev_m
                if language == "zh":
                    if m_diff > 0.02:
                        margin_trend = f"EBITDA利润率已扩大至 {latest_m*100:.1f}%（提升 {m_diff*100:.1f}% 个百分点），表明运营效率有所提高。"
                    elif m_diff >= -0.02:
                        margin_trend = f"EBITDA利润率保持在 {latest_m*100:.1f}% 左右，反映了稳定的成本结构。"
                    else:
                        margin_trend = f"EBITDA利润率已收缩至 {latest_m*100:.1f}%（下降 {abs(m_diff)*100:.1f}% 个百分点），这可能暗示着定价压力或投入成本上升。"
                else:
                    if m_diff > 0.02:
                        margin_trend = f"EBITDA margins have expanded to {latest_m*100:.1f}% (up {m_diff*100:.1f}% pts), indicating improving operational efficiency."
                    elif m_diff >= -0.02:
                        margin_trend = f"EBITDA margins remain stable at {latest_m*100:.1f}%, reflecting consistent cost structures."
                    else:
                        margin_trend = f"EBITDA margins have compressed to {latest_m*100:.1f}% (down {abs(m_diff)*100:.1f}% pts), suggesting pricing pressure or rising input costs."
                    
            fcf_history = financials_history["fcf"].dropna()
            if not fcf_history.empty:
                latest_fcf = fcf_history.iloc[-1]
                if latest_fcf > 0:
                    latest_ni = financials_history.loc[fcf_history.index[-1], "net_income"]
                    conversion = latest_fcf / latest_ni if latest_ni and latest_ni > 0 else 0.0
                    if language == "zh":
                        if conversion > 1.0:
                            fcf_trend = f"自由现金流生成效率极高，达到 {format_large_val(latest_fcf, language)}，且超过净利润，这标志着高品质的收益。"
                        else:
                            fcf_trend = f"自由现金流规模达 {format_large_val(latest_fcf, language)}，为资本配置提供了充足的流动性。"
                    else:
                        if conversion > 1.0:
                            fcf_trend = f"Free cash flow generation is highly efficient at {format_large_val(latest_fcf, language)}, exceeding net income, which signals high quality earnings."
                        else:
                            fcf_trend = f"Free cash flow stands at a healthy {format_large_val(latest_fcf, language)}, providing robust liquidity for capital allocation."
                else:
                    if language == "zh":
                        fcf_trend = f"自由现金流为负，达 {format_large_val(latest_fcf, language)}，如果持续如此，可能会给短期流动性带来压力。"
                    else:
                        fcf_trend = f"Free cash flow was negative at {format_large_val(latest_fcf, language)}, which could strain short-term liquidity if sustained."
                    
            # Debt analysis
            latest = financials_history.iloc[-1]
            debt = latest.get("total_debt", 0.0)
            cash = latest.get("total_cash", 0.0)
            ebitda = latest.get("ebitda", 0.0)
            
            if pd.isna(debt) or debt == 0:
                if language == "zh":
                    solvency_trend = f"公司维持极佳的资产负债表，几乎没有有息债务。"
                else:
                    solvency_trend = f"{company_name} maintains an exceptionally clean balance sheet with virtually zero interest-bearing debt."
            elif not pd.isna(ebitda) and ebitda > 0:
                net_debt = debt - cash
                leverage = net_debt / ebitda
                if language == "zh":
                    if leverage < 0:
                        solvency_trend = f"公司拥有 {format_large_val(abs(net_debt), language)} 的净现金头寸，为抵御经济下行创造了强有力的屏障。"
                    elif leverage < 2.0:
                        solvency_trend = f"杠杆水平保守，净债务与EBITDA的比率为 {leverage:.2f}倍，远处于安全区间内。"
                    elif leverage < 4.0:
                        solvency_trend = f"杠杆水平温和，净债务与EBITDA的比率为 {leverage:.2f}倍，需要关注但暂无迫切风险。"
                    else:
                        solvency_trend = f"杠杆水平偏高，净债务与EBITDA的比率高达 {leverage:.2f}倍，财务风险偏高。"
                else:
                    if leverage < 0:
                        solvency_trend = f"The company holds a net-cash position of {format_large_val(abs(net_debt), language)}, creating a strong buffer against economic downturns."
                    elif leverage < 2.0:
                        solvency_trend = f"Leverage is conservative with a Net Debt to EBITDA ratio of {leverage:.2f}x, well within safe parameters."
                    elif leverage < 4.0:
                        solvency_trend = f"Leverage is moderate at {leverage:.2f}x Net Debt/EBITDA, requiring monitoring but not posing immediate risks."
                    else:
                        solvency_trend = f"Leverage is elevated at {leverage:.2f}x Net Debt/EBITDA, suggesting a higher financial risk profile."
            else:
                if language == "zh":
                    solvency_trend = f"总债务为 {format_large_val(debt, language)}，而持有现金为 {format_large_val(cash, language)}。"
                else:
                    solvency_trend = f"Total debt stands at {format_large_val(debt, language)} against cash of {format_large_val(cash, language)}."
        else:
            if language == "zh":
                rev_trend = "有限的财务报告历史使得多年度趋势分析难以进行。"
                margin_trend = "由于缺乏足够报告，无法计算利润率趋势。"
                fcf_trend = "由于历史数据不足，未能分析现金流动态。"
                solvency_trend = "偿债能力指标不可用。"
            else:
                rev_trend = "limited financial statement history makes multi-year trend analysis difficult"
                margin_trend = "margin trends cannot be calculated without multi-year reports"
                fcf_trend = "cash flow dynamics are unanalyzed due to insufficient history"
                solvency_trend = "solvency metrics are unavailable"

        # 3. Dynamic Calibration Explanation
        cal_details = ""
        if calibration_results.get("calibrated"):
            gm = calibration_results["growth_multiplier"]
            wo = calibration_results["wacc_offset"]
            dw = calibration_results["dcf_weight"]
            pts = calibration_results["backtest_points"]
            
            if language == "zh":
                wacc_action = "上调惩罚" if wo > 0 else "下调折让"
                growth_action = "保守折扣" if gm < 1.0 else "增长溢价"
                cal_details = (
                    f"我们的机器学习校准引擎回测了该股过去 {pts} 个财务报告期，"
                    f"优化参数以最小化历史目标值与实际未来价格之间的差距。"
                    f"优化器对增长率假设应用了 {abs(1.0-gm)*100:.1f}% 的{growth_action}，"
                    f"对WACC应用了 {abs(wo)*100:.2f}% 的{wacc_action}。"
                    f"此校准将 {dw*100:.0f}% 的权重分配给DCF估值，将 {(1.0-dw)*100:.0f}% 分配给基于基本面的估值模型，"
                    f"有效地调整了模型参数，使其与市场过去对该股的定价方式保持一致。"
                )
            else:
                wacc_action = "upward premium" if wo > 0 else "downward discount"
                growth_action = "conservative haircut" if gm < 1.0 else "growth premium"
                cal_details = (
                    f"Our machine learning calibration engine backtested {pts} historical reporting periods for {ticker_symbol}. "
                    f"It optimized parameters to minimize the gap between historical target valuations and actual future prices. "
                    f"The optimizer applied a {growth_action} of {abs(1.0-gm)*100:.1f}% to growth assumptions and a WACC {wacc_action} of {abs(wo)*100:.2f}%. "
                    f"This calibration assigns a weight of {dw*100:.0f}% to the DCF valuation and {(1.0-dw)*100:.0f}% to fundamentals-based models, "
                    f"effectively adjusting the model parameters to align with how the market historically priced {ticker_symbol}."
                )
        else:
            if language == "zh":
                cal_details = (
                    f"没有足够的历史数据来校准 {ticker_symbol} 的参数。"
                    f"采用了基准校准（60% DCF权重，40% 葛兰姆估值）。"
                    f"增长和折现率仅反映了标准的行业计算，未使用自定义的历史市场乘数。"
                )
            else:
                cal_details = (
                    f"Insufficient historical data was available to calibrate parameters for {ticker_symbol}. "
                    f"A baseline calibration (60% DCF weight, 40% Graham valuation) was used. Growth and WACC rates "
                    f"reflect standard industry calculations without customized historical market multipliers."
                )

        # 4. Multiples narrative
        multiples_narrative = ""
        target_pe = multiples_results["target_multiples"].get("pe_trailing")
        peer_pe_median = multiples_results["stats"].get("pe_trailing", {}).get("median")
        
        if target_pe and peer_pe_median and not pd.isna(target_pe) and not pd.isna(peer_pe_median):
            diff = (target_pe - peer_pe_median) / peer_pe_median
            if language == "zh":
                if diff < -0.15:
                    multiples_narrative = f"相对估值显示，{ticker_symbol}目前以较低估值交易（P/E 为 {target_pe:.1f}x，而同行中位数为 {peer_pe_median:.1f}x），呈现出潜在的价值空间。"
                elif diff > 0.15:
                    multiples_narrative = f"相对估值显示，{ticker_symbol}目前以溢价交易（P/E 为 {target_pe:.1f}x，而同行中位数为 {peer_pe_median:.1f}x），反映出市场对其的高期望值。"
                else:
                    multiples_narrative = f"相对估值显示，{ticker_symbol}目前与同行一致（P/E 为 {target_pe:.1f}x，而同行中位数为 {peer_pe_median:.1f}x），表明估值符合行业定位。"
            else:
                if diff < -0.15:
                    multiples_narrative = f"Relative valuation shows {ticker_symbol} trading at a significant discount (P/E of {target_pe:.1f}x vs peer median of {peer_pe_median:.1f}x), representing a potential value play."
                elif diff > 0.15:
                    multiples_narrative = f"Relative valuation shows {ticker_symbol} trading at a premium (P/E of {target_pe:.1f}x vs peer median of {peer_pe_median:.1f}x), reflecting high market expectations or a quality premium."
                else:
                    multiples_narrative = f"Relative valuation shows {ticker_symbol} trading in-line with peers (P/E of {target_pe:.1f}x vs peer median of {peer_pe_median:.1f}x), indicating a fair pricing relative to its direct competitors."
        else:
            if language == "zh":
                multiples_narrative = "同行乘数对比不够清晰或噪音较大，这表明同行公司可能尚未盈利或资本结构极其不同。"
            else:
                multiples_narrative = "Peer multiples comparisons were incomplete or noisy, indicating peer companies are either unprofitable or have highly divergent capital structures."

        # 5. Build dynamic narrative sections
        if language == "zh":
            overview_para = (
                f"经过全面的基本面分析，{company_name} ({ticker_symbol}) 在当前股价为 ${current_price:.2f} 时，"
                f"被认为是 {valuation_status}。模型识别该股票符合「{profile_name}」特征，"
                f"并基于历史回测动态分配了如下估值模型权重：现金流折现 (DCF) {final_dcf_w*100:.1f}%、"
                f"葛兰姆公式 {final_graham_w*100:.1f}%、同行乘数对比 {final_multiples_w*100:.1f}%、股利折现 (DDM) {final_ddm_w*100:.1f}%。"
                f"模型计算出的最终内在目标价格为 ${weighted_target:.2f}，"
                f"意味着大约有 {abs(margin_of_safety)*100:.1f}% 的{upside_text}。基于此安全边际，"
                f"模型对该股发布了“{recommendation}”投资评级。"
            )
            financials_para = (
                f"在业务运营方面，{company_name} {rev_trend}。{margin_trend} {fcf_trend} {solvency_trend}"
            )
            valuation_para = (
                f"估值由多角度定量模型驱动。DCF基准模型测算 intrinsic value 为 ${dcf_val:.2f}（折现率WACC为 {wacc_details['wacc']*100:.2f}%）。"
                f" {multiples_narrative} {cal_details}"
            )
        else:
            overview_para = (
                f"Following a comprehensive fundamental analysis, {company_name} ({ticker_symbol}) is currently estimated to be {valuation_status} "
                f"at its current price of ${current_price:.2f}. The model classifies the stock under the '{profile_name}' profile. "
                f"Based on historical backtesting and dynamic weight calibration, the intrinsic value is computed using the following weights: "
                f"DCF {final_dcf_w*100:.1f}%, Graham Formula {final_graham_w*100:.1f}%, Peer Multiples {final_multiples_w*100:.1f}%, and DDM {final_ddm_w*100:.1f}%. "
                f"The model calculates an optimized intrinsic target price of ${weighted_target:.2f}, "
                f"representing a {abs(margin_of_safety)*100:.1f}% {upside_text}. "
                f"Based on this margin of safety, our model issues a {recommendation} recommendation for the stock."
            )
            financials_para = (
                f"Operationally, {company_name} is {rev_trend}. {margin_trend} {fcf_trend} {solvency_trend}"
            )
            valuation_para = (
                f"Valuation is modeled through multiple complementary methodologies. The DCF base case projects an intrinsic value of ${dcf_val:.2f} "
                f"using a WACC of {wacc_details['wacc']*100:.2f}%. {multiples_narrative} {cal_details}"
            )

        # Scenarios explanation
        bear_12m = trajectories["bear"][12]
        base_12m = trajectories["base"][12]
        bull_12m = trajectories["bull"][12]
        
        if language == "zh":
            scenarios_para = (
                f"我们对不同宏观情景下的多周期股价目标进行了预测：\n"
                f"- **基准情景 (12个月目标价: ${base_12m:.2f})**: 假设经济软着陆，保持 {dcf_results['base']['growth_rate']*100:.1f}% 的稳健收入增长。短期目标价：${trajectories['base'][1]:.2f} (1个月), ${trajectories['base'][3]:.2f} (3个月), 以及 ${trajectories['base'][6]:.2f} (6个月)。\n"
                f"- **牛市情景 (12个月目标价: ${bull_12m:.2f})**: 假设市场份额加速夺取，利润率继续扩张，且降息周期使得折现率下降到 {dcf_results['bull']['discount_rate']*100:.2f}%。短期目标价：${trajectories['bull'][1]:.2f} (1个月), ${trajectories['bull'][3]:.2f} (3个月), 以及 ${trajectories['bull'][6]:.2f} (6个月)。\n"
                f"- **熊市情景 (12个月目标价: ${bear_12m:.2f})**: 假设遭遇衰退逆风，营收萎缩，且面临溢价风险使折现率攀升至 {dcf_results['bear']['discount_rate']*100:.2f}%。短期目标价：${trajectories['bear'][1]:.2f} (1个月), ${trajectories['bear'][3]:.2f} (3个月), 以及 ${trajectories['bear'][6]:.2f} (6个月)。"
            )
            
            action_word = "分批建仓 / 增持" if "BUY" in rec_simple or "买" in rec_simple else "继续持有" if "HOLD" in rec_simple or "持" in rec_simple else "获利了结 / 减持"
            advice_detail = "充足的安全边际提供了强有力的抗跌保护和潜在超额回报。" if "BUY" in rec_simple or "买" in rec_simple else "当前股价已充分反映了基本面表现，短期超额回报有限。" if "HOLD" in rec_simple or "持" in rec_simple else "当前估值明显透支了增长预期，若业绩表现未达预期将面临估值双杀的风险。"
            
            conclusion_para = (
                f"**操作建议**: 建议对 {ticker_symbol} 采取 [ {action_word} ] 策略。{advice_detail} "
                f"未来需要密切关注的风险催化剂包括同行板块乘数的漂移以及WACC在 {wacc_details['wacc']*100:.1f}% 的敏感度红线。"
            )
        else:
            scenarios_para = (
                f"We map price targets over multiple horizons across three distinct macro scenarios: \n"
                f"- **Base Case (12-month target: ${base_12m:.2f})**: Assumes a soft landing, steady topline growth of {dcf_results['base']['growth_rate']*100:.1f}%, and successful execution. Shorter-term price targets are projected at ${trajectories['base'][1]:.2f} (1m), ${trajectories['base'][3]:.2f} (3m), and ${trajectories['base'][6]:.2f} (6m).\n"
                f"- **Bull Case (12-month target: ${bull_12m:.2f})**: Assumes accelerating market share capture, margins expansion, and a declining interest rate environment lowering WACC to {dcf_results['bull']['discount_rate']*100:.2f}%. Projected targets: ${trajectories['bull'][1]:.2f} (1m), ${trajectories['bull'][3]:.2f} (3m), and ${trajectories['bull'][6]:.2f} (6m).\n"
                f"- **Bear Case (12-month target: ${bear_12m:.2f})**: Assumes recessionary headwinds contracting revenue, pricing pressure, and an elevated discount rate of {dcf_results['bear']['discount_rate']*100:.2f}%. Projected targets: ${trajectories['bear'][1]:.2f} (1m), ${trajectories['bear'][3]:.2f} (3m), and ${trajectories['bear'][6]:.2f} (6m)."
            )
            
            conclusion_para = (
                f"**Actionable Advice**: "
                f"{'Accumulate shares' if rec_simple in ['BUY', 'STRONG BUY'] else 'Hold positions' if rec_simple == 'HOLD' else 'Reduce exposure or take profits'} "
                f"on {ticker_symbol}. "
                f"{'The substantial margin of safety offers downside protection and significant upside if growth target is met.' if rec_simple in ['BUY', 'STRONG BUY'] else 'The current stock price fully reflects fundamental value, leaving limited room for alpha.' if rec_simple == 'HOLD' else 'The valuation is stretched relative to fundamentals, making the stock highly sensitive to any earnings misses.'} "
                f"Key risk factors to monitor include peer sector multiples shifts and the WACC sensitivity threshold around {wacc_details['wacc']*100:.1f}%."
            )

        return {
            "recommendation": recommendation,
            "color": rec_color,
            "margin_of_safety": margin_of_safety,
            "target_price": weighted_target,
            "overview": overview_para,
            "financials_analysis": financials_para,
            "valuation_analysis": valuation_para,
            "scenarios_analysis": scenarios_para,
            "action_plan": conclusion_para,
            "profile": profile_code,
            "profile_name": profile_name,
            "dcf_weight": final_dcf_w,
            "graham_weight": final_graham_w,
            "multiples_weight": final_multiples_w,
            "ddm_weight": final_ddm_w
        }
