"""Northstar — decision-focused company valuation and portfolio risk UI."""

from __future__ import annotations

import html
import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from valuation.data_provider import DataProvider
from valuation.engine import ValuationEngine
from valuation.option_scoring import (
    COMBO_PUT_PROFILE_MAP,
    BullishComboPreferences,
    ShortPutPreferences,
)
from valuation.options import (
    QUOTE_BASIS_AUTO,
    QUOTE_BASIS_LIVE,
    QUOTE_BASIS_PREVIOUS_SESSION,
    OptionsAnalyzer,
)
from valuation.positions import PositionManager, get_latest_price, load_portfolio, normalize_portfolio


st.set_page_config(
    page_title="Northstar | Company valuation",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


PALETTE = {
    "ink": "#102A43",
    "muted": "#627D98",
    "teal": "#0F766E",
    "teal_soft": "#CCFBF1",
    "blue": "#2563EB",
    "amber": "#B45309",
    "red": "#B42318",
    "surface": "#FFFFFF",
    "canvas": "#F4F7FB",
    "line": "#D9E2EC",
}

OPTION_QUOTE_BASIS_LABELS = {
    "Auto — live or saved snapshot": QUOTE_BASIS_AUTO,
    "Live current session": QUOTE_BASIS_LIVE,
    "Latest saved regular session": QUOTE_BASIS_PREVIOUS_SESSION,
}

PORTFOLIO_SESSION_KEY = "northstar_portfolio"


def get_session_portfolio() -> dict:
    """Return an isolated portfolio for the current browser session."""
    if PORTFOLIO_SESSION_KEY not in st.session_state:
        st.session_state[PORTFOLIO_SESSION_KEY] = load_portfolio()
    return st.session_state[PORTFOLIO_SESSION_KEY]


def set_session_portfolio(portfolio: dict) -> None:
    st.session_state[PORTFOLIO_SESSION_KEY] = normalize_portfolio(portfolio)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700&display=swap');
    :root { color-scheme: light; }
    .stApp { background: #F4F7FB; color: #102A43; }
    .block-container { max-width: 1240px; padding-top: 1.25rem; padding-bottom: 4rem; }
    html, body, [class*="css"] { font-family: "DM Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    h1, h2, h3, h4 { font-family: "Manrope", "DM Sans", sans-serif; color: #102A43; letter-spacing: -0.025em; }
    h1 { font-size: clamp(2rem, 4vw, 3.25rem); line-height: 1.06; }
    p { color: #486581; }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { display: none; }
    [data-testid="stSidebar"] { background: #FFFFFF; }
    .brand-row { display:flex; align-items:center; gap:.75rem; padding:.35rem 0 1rem; }
    .brand-mark { width:2rem; height:2rem; border-radius:.65rem; background:#0F766E; color:white; display:grid; place-items:center; font-weight:800; }
    .brand-name { font-family:"Manrope"; font-size:1.05rem; font-weight:700; color:#102A43; }
    .brand-tag { color:#829AB1; font-size:.78rem; margin-left:.35rem; }
    .eyebrow { color:#0F766E; font-size:.76rem; font-weight:700; text-transform:uppercase; letter-spacing:.12em; margin-bottom:.55rem; }
    .hero-copy { max-width:760px; margin:1.4rem 0 1.2rem; }
    .hero-copy h1 { margin:0 0 .65rem; }
    .hero-copy p { font-size:1.08rem; max-width:680px; margin:0; }
    .company-card, .content-card, .insight-card {
        background:#FFFFFF; border:1px solid #D9E2EC; border-radius:18px; box-shadow:0 8px 26px rgba(16,42,67,.055);
    }
    .company-card { padding:1.25rem 1.4rem; margin:.4rem 0 1rem; }
    .company-top { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; }
    .company-name { font-family:"Manrope"; font-size:1.35rem; font-weight:700; color:#102A43; margin:0; }
    .company-meta { color:#627D98; font-size:.9rem; margin-top:.3rem; }
    .verdict { border-radius:999px; font-size:.78rem; font-weight:700; padding:.42rem .72rem; white-space:nowrap; }
    .verdict.positive { background:#CCFBF1; color:#0F766E; }
    .verdict.neutral { background:#E8EEF5; color:#486581; }
    .verdict.negative { background:#FEE4E2; color:#B42318; }
    .content-card { padding:1.1rem 1.2rem; margin:.5rem 0; }
    .insight-card { padding:1.1rem 1.2rem; min-height:180px; }
    .insight-card h4 { margin:.05rem 0 .75rem; font-size:1rem; }
    .insight-card ul { margin:0; padding-left:1.15rem; color:#486581; }
    .insight-card li { margin:.58rem 0; line-height:1.45; }
    .scenario { background:#FFFFFF; border:1px solid #D9E2EC; border-radius:14px; padding:1rem 1.05rem; }
    .scenario-label { color:#829AB1; text-transform:uppercase; letter-spacing:.08em; font-size:.72rem; font-weight:700; }
    .scenario-price { color:#102A43; font-family:"Manrope"; font-size:clamp(1.1rem,2.2vw,1.5rem); font-weight:700; margin:.25rem 0; white-space:nowrap; }
    .scenario-return { font-size:.86rem; color:#486581; }
    .recommendation { background:#ECFDF5; border:1px solid #A7F3D0; border-radius:16px; padding:1.1rem 1.25rem; margin:.75rem 0; }
    .recommendation strong { color:#065F46; font-family:"Manrope"; font-size:1.2rem; }
    .recommendation p { margin:.35rem 0 0; color:#047857; }
    .empty-state { text-align:center; background:#FFFFFF; border:1px dashed #BCCCDC; border-radius:18px; padding:3rem 1rem; color:#627D98; }
    .fine-print { color:#829AB1; font-size:.78rem; line-height:1.4; margin-top:1.5rem; }
    div[data-testid="stMetric"] { background:#FFFFFF; border:1px solid #D9E2EC; border-radius:15px; padding:1rem 1.05rem; box-shadow:0 5px 18px rgba(16,42,67,.04); }
    div[data-testid="stMetricLabel"] p { color:#627D98; font-weight:600; }
    div[data-testid="stMetricValue"] { color:#102A43; font-family:"Manrope"; overflow:visible; }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div, div[data-testid="stMetricValue"] p {
        font-size:clamp(1.35rem,2.1vw,2.05rem) !important; line-height:1.15 !important;
        overflow:visible !important; text-overflow:clip !important; white-space:nowrap !important;
    }
    div[data-testid="stMetricDelta"] { font-weight:600; }
    .stButton > button, .stFormSubmitButton > button {
        border-radius:11px; min-height:2.75rem; border:1px solid #0F766E; background:#0F766E; color:#FFFFFF; font-weight:700;
    }
    .stButton > button p, .stButton > button span,
    .stFormSubmitButton > button p, .stFormSubmitButton > button span { color:#FFFFFF !important; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { background:#115E59; color:#FFFFFF; border-color:#115E59; }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { border-radius:11px; border-color:#BCCCDC; background:#FFFFFF; }
    div[data-testid="stTabs"] button { color:#627D98; font-weight:700; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color:#0F766E; }
    div[data-testid="stDataFrame"] { border:1px solid #D9E2EC; border-radius:12px; overflow:hidden; }
    hr { border-color:#D9E2EC; }
    @media (max-width: 700px) {
        .brand-tag { display:none; }
        .company-top { display:block; }
        .verdict { display:inline-block; margin-top:.8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def esc(value: Any) -> str:
    return html.escape(str(value or ""))


def money(value: Any, currency: str = "USD", decimals: int = 2) -> str:
    number = float(value or 0.0)
    prefix = "$" if currency == "USD" else f"{currency} "
    return f"{prefix}{number:,.{decimals}f}"


def compact_price_number(value: Any, show_cents: bool = True) -> str:
    """Format a per-share value to fit a headline card without ellipsis."""
    number = float(value or 0.0)
    magnitude = abs(number)
    if magnitude >= 1e9:
        return f"{number / 1e9:,.2f}B"
    if magnitude >= 1e6:
        return f"{number / 1e6:,.2f}M"
    if magnitude >= 100_000:
        return f"{number / 1e3:,.0f}K"
    if magnitude >= 10_000:
        return f"{number / 1e3:,.1f}K"
    if magnitude >= 1_000:
        return f"{number:,.0f}"
    decimals = 2 if show_cents else 0
    return f"{number:,.{decimals}f}"


def headline_money(value: Any, currency: str = "USD") -> str:
    prefix = "$" if currency == "USD" else f"{currency} "
    return f"{prefix}{compact_price_number(value)}"


def headline_range(low: Any, high: Any, currency: str = "USD") -> str:
    low_number = float(low or 0.0)
    high_number = float(high or 0.0)
    prefix = "$" if currency == "USD" else f"{currency} "
    for scale, suffix, decimals in (
        (1e9, "B", 2),
        (1e6, "M", 2),
        (100_000, "K", 0),
        (10_000, "K", 1),
    ):
        if abs(low_number) >= scale and abs(high_number) >= scale:
            return (
                f"{prefix}{low_number / scale:,.{decimals}f}"
                f"–{high_number / scale:,.{decimals}f}{suffix}"
            )
    return (
        f"{prefix}{compact_price_number(low_number, show_cents=False)}"
        f"–{compact_price_number(high_number, show_cents=False)}"
    )


def percent(value: Any, signed: bool = False, decimals: int = 1) -> str:
    number = float(value or 0.0) * 100
    sign = "+" if signed and number > 0 else ""
    return f"{sign}{number:.{decimals}f}%"


def underlying_quote_caption(result: dict) -> str:
    source_labels = {
        "preMarketPrice": "pre-market quote",
        "regularMarketPrice": "regular-market quote",
        "postMarketPrice": "post-market quote",
        "currentPrice": "current provider quote",
        "latest_1m_close": "latest one-minute trade",
        "fast_info.last_price": "provider last price",
        "previousClose": "previous close fallback",
        "saved_regular_session_snapshot": "saved regular-session snapshot",
    }
    source = result.get("spot_source") or "unknown source"
    parts = [f"Underlying source: {source_labels.get(source, source)}"]
    as_of = result.get("spot_as_of")
    if as_of:
        try:
            timestamp = pd.Timestamp(as_of)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            market_timestamp = timestamp.tz_convert("America/New_York")
            parts.append(f"as of {market_timestamp.strftime('%Y-%m-%d %H:%M %Z')}")
        except (TypeError, ValueError):
            parts.append(f"as of {as_of}")
    parts.append(f"provider market state {result.get('market_state') or 'unknown'}")
    snapshot_captured_at = result.get("snapshot_captured_at") if result.get("is_snapshot") else None
    if snapshot_captured_at:
        try:
            capture_timestamp = pd.Timestamp(snapshot_captured_at)
            if capture_timestamp.tzinfo is None:
                capture_timestamp = capture_timestamp.tz_localize("UTC")
            capture_timestamp = capture_timestamp.tz_convert("America/New_York")
            parts.append(f"snapshot captured {capture_timestamp.strftime('%Y-%m-%d %H:%M %Z')}")
        except (TypeError, ValueError):
            parts.append(f"snapshot captured {snapshot_captured_at}")
    regular_price = float(result.get("regular_market_price") or 0.0)
    spot = float(result.get("spot") or 0.0)
    if regular_price > 0 and abs(regular_price - spot) >= 0.005:
        parts.append(f"regular-session reference {money(regular_price)}")
    return " Â· ".join(parts) + "."


def compact(value: Any, currency: str = "USD") -> str:
    number = float(value or 0.0)
    prefix = "$" if currency == "USD" else f"{currency} "
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= scale:
            return f"{prefix}{number / scale:,.1f}{suffix}"
    return f"{prefix}{number:,.0f}"


@st.cache_data(ttl=900, show_spinner=False)
def cached_analysis(symbol: str) -> dict:
    return ValuationEngine().analyze(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_resolve(query: str) -> tuple[str, str]:
    return ValuationEngine().resolve_symbol(query)


@st.cache_data(ttl=300, show_spinner=False)
def cached_price(symbol: str) -> float:
    return get_latest_price(symbol)


@st.cache_data(ttl=30, show_spinner=False)
def cached_short_put_scan(
    symbol: str,
    quote_basis: str,
    profile: str,
    min_dte: int,
    max_dte: int,
    max_assignment_probability: float,
    min_open_interest: int,
    max_spread_pct: float,
    max_cash_secured: float,
    limit: int,
) -> dict:
    preferences = ShortPutPreferences.for_profile(profile).with_dte_window(min_dte, max_dte).with_overrides(
        max_assignment_probability=max_assignment_probability,
        min_open_interest=min_open_interest,
        max_bid_ask_spread_pct=max_spread_pct,
        max_cash_secured=max_cash_secured if max_cash_secured > 0 else None,
        limit=limit,
    )
    provider = DataProvider()
    risk_free_rate = 0.0425 if quote_basis == QUOTE_BASIS_PREVIOUS_SESSION else provider.get_risk_free_rate()
    return OptionsAnalyzer(provider).recommend_short_puts(
        symbol,
        preferences,
        r_rate=risk_free_rate,
        quote_basis=quote_basis,
    )


@st.cache_data(ttl=30, show_spinner=False)
def cached_bullish_combo_scan(
    symbol: str,
    quote_basis: str,
    profile: str,
    min_dte: int,
    max_dte: int,
    max_assignment_probability: float,
    min_open_interest: int,
    max_spread_pct: float,
    max_cash_secured: float,
    min_call_delta: float,
    min_premium_utilization: float,
    max_extra_debit: float,
    limit: int,
) -> dict:
    put_profile = COMBO_PUT_PROFILE_MAP[profile]
    put_preferences = ShortPutPreferences.for_profile(put_profile).with_dte_window(min_dte, max_dte).with_overrides(
        max_assignment_probability=max_assignment_probability,
        min_open_interest=min_open_interest,
        max_bid_ask_spread_pct=max_spread_pct,
        max_cash_secured=max_cash_secured if max_cash_secured > 0 else None,
        limit=max(limit, 5),
    )
    combo_preferences = BullishComboPreferences.for_profile(
        profile,
        min_call_delta=min_call_delta,
        min_call_open_interest=min_open_interest,
        max_call_spread_pct=max_spread_pct,
        min_premium_utilization=min_premium_utilization,
        max_extra_debit=max_extra_debit,
        limit=limit,
    )
    provider = DataProvider()
    risk_free_rate = 0.0425 if quote_basis == QUOTE_BASIS_PREVIOUS_SESSION else provider.get_risk_free_rate()
    return OptionsAnalyzer(provider).recommend_premium_funded_bullish_combo(
        symbol,
        put_preferences,
        combo_preferences,
        r_rate=risk_free_rate,
        quote_basis=quote_basis,
    )


def plot_layout(figure: go.Figure, height: int = 360) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=PALETTE["muted"]),
        hoverlabel=dict(bgcolor="#102A43", font_color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    figure.update_xaxes(gridcolor="#E7EDF4", zeroline=False)
    figure.update_yaxes(gridcolor="#E7EDF4", zeroline=False)
    return figure


def valuation_range_chart(result: dict) -> go.Figure:
    valuation = result["valuation"]
    price = result["current_price"]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[valuation["bear"], valuation["bull"]], y=["Estimated range", "Estimated range"],
        mode="lines", line=dict(color="#99F6E4", width=18), hoverinfo="skip", showlegend=False,
    ))
    figure.add_trace(go.Scatter(
        x=[valuation["bear"], valuation["base"], valuation["bull"]],
        y=["Estimated range"] * 3, mode="markers+text",
        marker=dict(color=[PALETTE["amber"], PALETTE["teal"], PALETTE["blue"]], size=[12, 17, 12]),
        text=["Downside", "Fair value", "Upside"], textposition="top center",
        customdata=[[money(valuation["bear"])], [money(valuation["base"])], [money(valuation["bull"])]],
        hovertemplate="%{text}: %{customdata[0]}<extra></extra>", showlegend=False,
    ))
    figure.add_vline(x=price, line_color=PALETTE["ink"], line_width=2, line_dash="dot")
    figure.add_annotation(x=price, y="Estimated range", text=f"Market {headline_money(price)}", showarrow=True, arrowhead=2, ay=52, font=dict(color=PALETTE["ink"]))
    figure.update_yaxes(showticklabels=False)
    figure.update_xaxes(title="Price per share", tickprefix="$", tickformat="~s", rangemode="tozero")
    return plot_layout(figure, 285)


def method_chart(result: dict) -> go.Figure:
    methods = result["methods"]
    names = [method["name"] for method in methods]
    figure = go.Figure()
    figure.add_trace(go.Bar(
        y=names, x=[method["base"] for method in methods], orientation="h",
        marker_color=PALETTE["teal"], text=[headline_money(method["base"]) for method in methods],
        textposition="outside", hovertemplate="%{y}: $%{x:,.2f}<extra></extra>",
    ))
    figure.add_vline(x=result["current_price"], line_color=PALETTE["ink"], line_dash="dot", line_width=2)
    figure.update_xaxes(title="Base value per share", tickprefix="$", tickformat="~s", rangemode="tozero")
    figure.update_yaxes(autorange="reversed")
    return plot_layout(figure, max(280, 85 * len(methods)))


def financial_chart(result: dict) -> go.Figure:
    financials = result["financials"].copy().tail(6)
    labels = [pd.to_datetime(index).strftime("%Y") for index in financials.index]
    figure = go.Figure()
    for column, label, color in (("revenue", "Revenue", PALETTE["blue"]), ("fcf", "Free cash flow", PALETTE["teal"])):
        if column in financials:
            figure.add_trace(go.Bar(
                x=labels, y=financials[column], name=label, marker_color=color,
                hovertemplate=f"{label}: $%{{y:,.0f}}<extra></extra>",
            ))
    figure.update_yaxes(title="Reported amount", tickformat="~s")
    figure.update_layout(barmode="group")
    return plot_layout(figure, 390)


def price_chart(result: dict) -> go.Figure | None:
    history = result.get("price_history")
    if history is None or history.empty or "Close" not in history:
        return None
    figure = go.Figure(go.Scatter(
        x=history.index, y=history["Close"], mode="lines", fill="tozeroy",
        line=dict(color=PALETTE["teal"], width=2.5), fillcolor="rgba(15,118,110,.09)",
        hovertemplate="$%{y:,.2f}<extra></extra>",
    ))
    figure.update_yaxes(title="Share price", tickprefix="$", tickformat="~s", fixedrange=True)
    figure.update_xaxes(fixedrange=True)
    return plot_layout(figure, 310)


def top_navigation() -> str:
    left, right = st.columns([1.7, 1.55], vertical_alignment="center")
    with left:
        st.markdown(
            '<div class="brand-row"><div class="brand-mark">N</div><div><span class="brand-name">Northstar</span>'
            '<span class="brand-tag">Valuation without the spreadsheet</span></div></div>',
            unsafe_allow_html=True,
        )
    with right:
        return st.radio(
            "Workspace", ["Company value", "Portfolio", "Options"], horizontal=True,
            label_visibility="collapsed", key="workspace_navigation",
        )


def render_company_header(result: dict) -> None:
    valuation = result["valuation"]
    premium_label = " · Leadership premium included" if result.get("market_regime", {}).get("leadership_premium") else ""
    st.markdown(
        f"""
        <div class="company-card">
          <div class="company-top">
            <div>
              <div class="company-name">{esc(result['company_name'])} <span style="color:#829AB1;font-weight:600">{esc(result['symbol'])}</span></div>
              <div class="company-meta">{esc(result['sector'])} · {esc(result['profile_label'])} · {esc(result['exchange'])}{esc(premium_label)}</div>
            </div>
            <span class="verdict {esc(valuation['tone'])}">{esc(valuation['verdict'])}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_company_value() -> None:
    st.markdown(
        '<div class="hero-copy"><div class="eyebrow">Company value</div>'
        '<h1>Know what the market price is asking you to believe.</h1>'
        '<p>Search a listed US company. Northstar selects the suitable models, builds the inputs, and gives you a range—not false precision.</p></div>',
        unsafe_allow_html=True,
    )

    if "active_symbol" not in st.session_state:
        st.session_state.active_symbol = "AAPL"
    if "company_query" not in st.session_state:
        st.session_state.company_query = "AAPL"

    with st.form("company_search_form"):
        search_col, action_col = st.columns([5, 1.15], vertical_alignment="bottom")
        with search_col:
            query = st.text_input(
                "Company or ticker", key="company_query",
                placeholder="Search Apple, Microsoft, AAPL…",
                help="You can enter either a company name or its ticker symbol.",
            )
        with action_col:
            submitted = st.form_submit_button("Value company", width="stretch")

    if submitted:
        try:
            with st.spinner("Finding the company…"):
                symbol, matched_name = cached_resolve(query)
            st.session_state.active_symbol = symbol
            st.session_state.matched_company = matched_name
        except Exception as exc:
            st.error(str(exc))
            return

    symbol = st.session_state.active_symbol
    try:
        with st.spinner(f"Building a current valuation for {symbol}…"):
            result = cached_analysis(symbol)
    except Exception as exc:
        st.error(str(exc))
        st.info("Try another listed company. Funds and ETFs need a different NAV-and-holdings framework, so Northstar does not force them through a stock DCF.")
        return

    render_company_header(result)
    valuation = result["valuation"]
    value_label = (
        "Market-adjusted fair value"
        if result.get("market_regime", {}).get("leadership_premium")
        else "Estimated fair value"
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Market price",
        headline_money(result["current_price"], result["currency"]),
        help=f"Exact quote: {money(result['current_price'], result['currency'])}",
    )
    metric_cols[1].metric(
        value_label,
        headline_money(valuation["base"], result["currency"]),
        percent(valuation["upside"], signed=True),
        help=f"Exact base value: {money(valuation['base'], result['currency'])}",
    )
    metric_cols[2].metric(
        "Valuation range",
        headline_range(valuation["bear"], valuation["bull"], result["currency"]),
        help=(
            f"Exact range: {money(valuation['bear'], result['currency'])} to "
            f"{money(valuation['bull'], result['currency'])}"
        ),
    )
    metric_cols[3].metric("Confidence", valuation["confidence_label"], f"{valuation['confidence_score']}/100", delta_color="off")

    summary_tab, financial_tab, valuation_tab, method_tab = st.tabs([
        "Summary", "Business performance", "Valuation range", "How it works",
    ])

    with summary_tab:
        st.plotly_chart(valuation_range_chart(result), width="stretch", config={"displayModeBar": False})
        scenario_columns = st.columns(3)
        scenario_data = [
            ("Downside case", valuation["bear"], valuation["bear_return"]),
            ("Base case", valuation["base"], valuation["upside"]),
            ("Upside case", valuation["bull"], valuation["bull_return"]),
        ]
        for column, (label, value, change) in zip(scenario_columns, scenario_data):
            column.markdown(
                f'<div class="scenario"><div class="scenario-label">{label}</div>'
                f'<div class="scenario-price" title="{esc(money(value, result["currency"]))}">'
                f'{headline_money(value, result["currency"])}</div>'
                f'<div class="scenario-return">{percent(change, signed=True)} from today</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        driver_col, risk_col = st.columns(2)
        driver_items = "".join(f"<li>{esc(item)}</li>" for item in result["drivers"])
        risk_items = "".join(f"<li>{esc(item)}</li>" for item in result["risks"])
        driver_col.markdown(f'<div class="insight-card"><h4>What supports the value</h4><ul>{driver_items}</ul></div>', unsafe_allow_html=True)
        risk_col.markdown(f'<div class="insight-card"><h4>What could change the view</h4><ul>{risk_items}</ul></div>', unsafe_allow_html=True)

        chart = price_chart(result)
        if chart is not None:
            st.markdown("### The last 12 months")
            st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})

    with financial_tab:
        info = result["info"]
        health_cols = st.columns(4)
        health_cols[0].metric("Revenue growth", percent(info.get("revenueGrowth")))
        health_cols[1].metric("Profit margin", percent(info.get("profitMargins")))
        health_cols[2].metric("Return on equity", percent(info.get("returnOnEquity")))
        debt_to_equity = info.get("debtToEquity")
        health_cols[3].metric("Debt / equity", f"{float(debt_to_equity):.0f}%" if debt_to_equity is not None else "Not available")
        st.markdown("### Reported revenue and free cash flow")
        st.plotly_chart(financial_chart(result), width="stretch", config={"displayModeBar": False})

        financials = result["financials"].copy().tail(6)
        table = pd.DataFrame({
            "Year": [pd.to_datetime(index).strftime("%Y") for index in financials.index],
            "Revenue": [compact(value) for value in financials.get("revenue", pd.Series(index=financials.index, dtype=float))],
            "Free cash flow": [compact(value) for value in financials.get("fcf", pd.Series(index=financials.index, dtype=float))],
            "Net income": [compact(value) for value in financials.get("net_income", pd.Series(index=financials.index, dtype=float))],
            "Cash": [compact(value) for value in financials.get("total_cash", pd.Series(index=financials.index, dtype=float))],
            "Debt": [compact(value) for value in financials.get("total_debt", pd.Series(index=financials.index, dtype=float))],
        })
        st.dataframe(table, width="stretch", hide_index=True)

    with valuation_tab:
        st.markdown("### Independent views, combined carefully")
        st.caption(
            "Methods are chosen for this type of company and weighted by relevance and data quality. "
            "Dominant technology leaders also include a broad, present-value-adjusted institutional outlook."
        )
        st.plotly_chart(method_chart(result), width="stretch", config={"displayModeBar": False})

        method_rows = pd.DataFrame([{
            "Method": method["name"],
            "Downside": money(method["bear"]),
            "Base": money(method["base"]),
            "Upside": money(method["bull"]),
            "Weight": percent(method["weight"]),
            "What it captures": method["note"],
        } for method in result["methods"]])
        st.dataframe(method_rows, width="stretch", hide_index=True)

        implied_growth = result.get("implied_growth")
        if implied_growth is not None:
            st.markdown(
                f'<div class="content-card"><div class="eyebrow">Reverse DCF</div>'
                f'<strong style="font-size:1.15rem;color:#102A43">The current price implies about {percent(implied_growth)} initial annual growth.</strong>'
                f'<p style="margin:.35rem 0 0">Northstar’s evidence-based starting assumption is {percent(result["assumptions"]["initial_growth"])}. The gap shows how demanding the market price is.</p></div>',
                unsafe_allow_html=True,
            )

        dcf = result["models"].get("dcf", {})
        if dcf and "error" not in dcf and dcf.get("sensitivity"):
            sensitivity = pd.DataFrame(dcf["sensitivity"])
            figure = go.Figure(go.Heatmap(
                z=sensitivity.values, x=sensitivity.columns, y=sensitivity.index,
                colorscale=[[0, "#FEE4E2"], [0.5, "#FEF3C7"], [1, "#CCFBF1"]],
                text=np.vectorize(lambda value: f"${value:,.0f}")(sensitivity.values),
                texttemplate="%{text}", hovertemplate="WACC %{x}<br>Terminal growth %{y}<br>$%{z:,.2f}<extra></extra>",
            ))
            figure.update_xaxes(title="Discount rate")
            figure.update_yaxes(title="Terminal growth")
            st.markdown("### Cash-flow sensitivity")
            st.plotly_chart(plot_layout(figure, 410), width="stretch", config={"displayModeBar": False})

    with method_tab:
        st.markdown("### Automatic where it should be. Visible when you need it.")
        st.write(
            "Northstar identifies the company type first, then uses only suitable valuation methods. "
            "Financial institutions are valued from book value and residual income; cash-generating businesses use a ten-year DCF; "
            "dividend value is included only when distributions are meaningful. Large technology leaders also receive an explicit "
            "market-regime layer based on broad analyst targets, discounted back to today."
        )
        for method in result["methods"]:
            with st.expander(f"{method['name']} · {percent(method['weight'])} of the final range"):
                st.write(method["note"])
                st.write(f"Model confidence: {percent(method['confidence'])}.")

        with st.expander("Automatic model inputs"):
            assumptions = result["assumptions"]
            input_rows = pd.DataFrame([
                {"Input": "Company type", "Value": assumptions["profile_label"], "Basis": "Business model and reported fundamentals"},
                {"Input": "Market regime", "Value": result["market_regime"]["label"], "Basis": "Scale, sector, industry and institutional coverage"},
                {"Input": "Starting growth", "Value": percent(assumptions["initial_growth"]), "Basis": assumptions["sources"]["growth"]},
                {"Input": "Normalized FCF margin", "Value": percent(assumptions["target_fcf_margin"]), "Basis": assumptions["sources"]["margin"]},
                {"Input": "Equity risk premium", "Value": percent(result["market_risk_premium"]), "Basis": "Latest published US implied market premium"},
                {"Input": "Discount rate", "Value": percent(result["wacc"]["wacc"]), "Basis": "Market rates, beta, leverage, tax and size"},
                {"Input": "Long-run growth", "Value": percent(assumptions["terminal_growth"]), "Basis": assumptions["sources"]["terminal_growth"]},
            ])
            st.dataframe(input_rows, width="stretch", hide_index=True)

        with st.expander("Comparable companies used in the background"):
            peers = result["models"].get("multiples", {}).get("peers_table", [])
            if peers:
                peer_frame = pd.DataFrame(peers)
                selected = [column for column in ("symbol", "name", "market_cap", "pe_forward", "ev_ebitda", "ev_revenue", "pb_ratio") if column in peer_frame]
                peer_frame = peer_frame[selected].rename(columns={
                    "symbol": "Ticker", "name": "Company", "market_cap": "Market cap",
                    "pe_forward": "Forward P/E", "ev_ebitda": "EV / EBITDA", "ev_revenue": "EV / revenue", "pb_ratio": "Price / book",
                })
                if "Market cap" in peer_frame:
                    peer_frame["Market cap"] = peer_frame["Market cap"].apply(compact)
                st.dataframe(peer_frame, width="stretch", hide_index=True)
            else:
                st.write("A reliable peer set was not available.")

    st.markdown(
        '<div class="fine-print">Decision support, not investment advice. Prices and company data may be delayed. '
        'A valuation range is an estimate, not a promise of future market price.</div>',
        unsafe_allow_html=True,
    )


def portfolio_donut(diagnostics: dict) -> go.Figure | None:
    allocations = diagnostics["ticker_allocations_pct"]
    if not allocations:
        return None
    labels = list(allocations)
    values = [diagnostics["account_value"] * allocations[label] for label in labels]
    cash = diagnostics["available_cash"]
    if cash > 0:
        labels.append("Cash")
        values.append(cash)
    figure = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.64, sort=False,
        marker=dict(colors=["#0F766E", "#2563EB", "#7C3AED", "#D97706", "#DC2626", "#94A3B8"]),
        textinfo="label+percent", hovertemplate="%{label}<br>$%{value:,.0f}<extra></extra>",
    ))
    figure.add_annotation(text="Allocation", showarrow=False, font=dict(size=15, color=PALETTE["ink"]))
    return plot_layout(figure, 360)


def render_holdings(portfolio: dict, diagnostics: dict) -> None:
    if not diagnostics["positions"]:
        st.markdown('<div class="empty-state"><h3>No positions yet</h3><p>Plan your first position from the “Add position” tab. Northstar will size it from the risk you choose.</p></div>', unsafe_allow_html=True)
        return

    rows = pd.DataFrame([{
        "Ticker": item["ticker"],
        "Role": item.get("role", "Core"),
        "Shares": item["shares"],
        "Average cost": money(item["avg_price"]),
        "Last price": money(item["current_price"]),
        "Market value": money(item["market_value"]),
        "Return": percent(item["pnl_pct"], signed=True),
        "Account weight": percent(item["allocation_pct"]),
        "Stop": money(item["stop_loss"]) if item.get("stop_loss") else "Not set",
    } for item in diagnostics["positions"]])
    st.dataframe(rows, width="stretch", hide_index=True)

    chart_col, risk_col = st.columns([1.2, 1])
    chart = portfolio_donut(diagnostics)
    if chart is not None:
        chart_col.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
    with risk_col:
        st.markdown("### Risk checks")
        if diagnostics["warnings"]:
            for warning in diagnostics["warnings"]:
                st.warning(warning["message"])
        else:
            st.success("No position, sector, leverage, or stop-risk limits are currently breached.")

    with st.expander("Edit or remove a holding"):
        choices = [item["ticker"] for item in diagnostics["positions"]]
        selected = st.selectbox("Holding", choices, key="edit_position_ticker")
        position = next(item for item in diagnostics["positions"] if item["ticker"] == selected)
        with st.form("edit_position_form"):
            edit_cols = st.columns(4)
            shares = edit_cols[0].number_input("Shares", min_value=0.0001, value=float(position["shares"]), step=1.0)
            average = edit_cols[1].number_input("Average cost", min_value=0.01, value=float(position["avg_price"]), step=0.10)
            stop = edit_cols[2].number_input("Stop price", min_value=0.0, value=float(position.get("stop_loss", 0.0)), step=0.10)
            role_options = ["Starter", "Core", "Tactical", "Income"]
            current_role = str(position.get("role", "Core")).title()
            if current_role not in role_options:
                role_options.append(current_role)
            role = edit_cols[3].selectbox("Portfolio role", role_options, index=role_options.index(current_role))
            notes = st.text_input("Notes", value=position.get("notes", ""), placeholder="Optional thesis or review trigger")
            save_col, delete_col = st.columns(2)
            save_clicked = save_col.form_submit_button("Save changes", width="stretch")
            delete_clicked = delete_col.form_submit_button("Remove holding", width="stretch")
        try:
            if save_clicked:
                updated = PositionManager.set_position(
                    selected,
                    shares,
                    average,
                    stop,
                    role,
                    position.get("sector"),
                    notes,
                    portfolio=portfolio,
                    persist=False,
                )
                set_session_portfolio(updated)
                st.success(f"{selected} was updated.")
                st.cache_data.clear()
                st.rerun()
            if delete_clicked:
                updated = PositionManager.remove_position(selected, portfolio=portfolio, persist=False)
                set_session_portfolio(updated)
                st.success(f"{selected} was removed.")
                st.cache_data.clear()
                st.rerun()
        except Exception as exc:
            st.error(str(exc))


def render_add_position(portfolio: dict, diagnostics: dict) -> None:
    st.markdown("### Find a company")
    st.caption("Search by company name or ticker. The quote is filled automatically before position sizing.")
    with st.form("position_search_form"):
        query_col, role_col, action_col = st.columns([3.5, 1.5, 1.2], vertical_alignment="bottom")
        query = query_col.text_input("Company or ticker", placeholder="NVIDIA or NVDA", key="position_query")
        role = role_col.selectbox("Portfolio role", ["Starter", "Core", "Tactical", "Income"])
        find_clicked = action_col.form_submit_button("Get quote", width="stretch")
    if find_clicked:
        try:
            with st.spinner("Finding the company and latest quote…"):
                symbol, name = cached_resolve(query)
                quote = cached_price(symbol)
            if quote <= 0:
                raise ValueError(f"A reliable price for {symbol} is unavailable.")
            st.session_state.position_plan = {"symbol": symbol, "name": name, "price": quote, "role": role}
            st.session_state.pop("position_sizing", None)
        except Exception as exc:
            st.error(str(exc))

    plan = st.session_state.get("position_plan")
    if not plan:
        return

    existing = next((item for item in diagnostics["positions"] if item["ticker"] == plan["symbol"]), None)
    existing_value = existing["market_value"] if existing else 0.0
    st.markdown(
        f'<div class="company-card"><div class="company-name">{esc(plan["name"])} '
        f'<span style="color:#829AB1">{esc(plan["symbol"])}</span></div>'
        f'<div class="company-meta">Latest price {money(plan["price"])}'
        f'{" · Existing holding " + money(existing_value) if existing else ""}</div></div>',
        unsafe_allow_html=True,
    )

    with st.form("position_sizing_form"):
        size_cols = st.columns(4)
        entry = size_cols[0].number_input("Planned entry", min_value=0.01, value=float(plan["price"]), step=0.10)
        stop = size_cols[1].number_input("Exit if wrong", min_value=0.01, value=float(plan["price"] * 0.92), step=0.10, help="A price that invalidates the trade—not a guaranteed execution price.")
        risk_pct = size_cols[2].number_input("Account risk (%)", min_value=0.1, max_value=5.0, value=float(portfolio["risk_per_trade"] * 100), step=0.1) / 100.0
        max_position = size_cols[3].number_input("Max position (%)", min_value=3.0, max_value=50.0, value=float(portfolio["max_position_pct"] * 100), step=1.0) / 100.0
        size_clicked = st.form_submit_button("Calculate position", width="stretch")
    if size_clicked:
        try:
            st.session_state.position_sizing = PositionManager.calculate_position_size(
                entry, stop, portfolio["account_value"], risk_pct, max_position,
                existing_value, diagnostics["available_cash"],
            )
            st.session_state.position_sizing.update({"entry": entry, "stop": stop})
        except Exception as exc:
            st.error(str(exc))

    sizing = st.session_state.get("position_sizing")
    if not sizing:
        return
    if sizing["suggested_shares"] <= 0:
        st.warning(f"No additional shares fit the current limits. The binding constraint is {sizing['limiting_factor']}.")
        return

    st.markdown(
        f'<div class="recommendation"><strong>{sizing["suggested_shares"]:,} shares · {money(sizing["position_value"])}</strong>'
        f'<p>Estimated loss at the stop: {money(sizing["max_loss_at_stop"])} ({percent(sizing["max_loss_at_stop"] / portfolio["account_value"])}) '
        f'· Limited by {esc(sizing["limiting_factor"])}.</p></div>',
        unsafe_allow_html=True,
    )

    with st.form("confirm_position_form"):
        confirm_cols = st.columns(3)
        shares = confirm_cols[0].number_input("Shares to record", min_value=0.0001, value=float(sizing["suggested_shares"]), step=1.0)
        purchase_price = confirm_cols[1].number_input("Purchase price", min_value=0.01, value=float(sizing["entry"]), step=0.10)
        stop_price = confirm_cols[2].number_input("Stop price", min_value=0.0, value=float(sizing["stop"]), step=0.10)
        notes = st.text_input("Notes", placeholder="Optional thesis or review trigger")
        add_clicked = st.form_submit_button("Add to portfolio", width="stretch")
    if add_clicked:
        try:
            updated = PositionManager.add_or_update_position(
                plan["symbol"],
                shares,
                purchase_price,
                plan["role"],
                stop_price,
                notes=notes,
                portfolio=portfolio,
                persist=False,
            )
            set_session_portfolio(updated)
            st.success(f"{plan['symbol']} was added to the portfolio.")
            st.session_state.pop("position_sizing", None)
            st.session_state.pop("position_plan", None)
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def render_risk_setup(portfolio: dict) -> None:
    st.markdown("### Portfolio guardrails")
    st.write("These settings drive position sizing and warnings. They are portfolio rules, not valuation assumptions.")
    with st.form("portfolio_settings_form"):
        setting_cols = st.columns(4)
        account_value = setting_cols[0].number_input("Account equity", min_value=1.0, value=float(portfolio["account_value"]), step=1000.0)
        risk = setting_cols[1].number_input("Risk per position (%)", min_value=0.1, max_value=5.0, value=float(portfolio["risk_per_trade"] * 100), step=0.1) / 100.0
        max_position = setting_cols[2].number_input("Max single position (%)", min_value=3.0, max_value=50.0, value=float(portfolio["max_position_pct"] * 100), step=1.0) / 100.0
        max_sector = setting_cols[3].number_input("Max sector exposure (%)", min_value=10.0, max_value=100.0, value=float(portfolio["max_sector_pct"] * 100), step=5.0) / 100.0
        save_clicked = st.form_submit_button("Save guardrails", width="stretch")
    if save_clicked:
        try:
            updated = PositionManager.update_settings(
                account_value,
                risk,
                max_position,
                max_sector,
                portfolio=portfolio,
                persist=False,
            )
            set_session_portfolio(updated)
            st.success("Portfolio guardrails were saved.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.markdown("### Private session backup")
    st.caption(
        "Portfolio data is isolated to this browser session and is not stored on the public server. "
        "Download a backup before the session ends, then restore it on a later visit."
    )
    backup_col, restore_col = st.columns(2)
    backup_col.download_button(
        "Download portfolio backup",
        data=json.dumps(portfolio, indent=2, ensure_ascii=False),
        file_name="northstar-portfolio.json",
        mime="application/json",
        width="stretch",
    )
    uploaded = restore_col.file_uploader(
        "Restore portfolio backup",
        type=["json"],
        key="portfolio_restore_file",
        help="Choose a Northstar JSON backup (maximum 1 MB).",
    )
    if uploaded is not None and st.button("Restore selected backup", width="stretch"):
        try:
            raw = uploaded.getvalue()
            if len(raw) > 1_000_000:
                raise ValueError("The backup is larger than the 1 MB limit.")
            restored = normalize_portfolio(json.loads(raw.decode("utf-8")))
            set_session_portfolio(restored)
            st.success("The portfolio backup was restored for this session.")
            st.rerun()
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            st.error(f"This backup could not be restored: {exc}")

    st.markdown(
        '<div class="content-card"><strong style="color:#102A43">How sizing works</strong>'
        '<p>A position must pass three tests: loss at the planned stop, maximum account weight, and available cash. '
        'Northstar uses the smallest share count from those three limits.</p></div>',
        unsafe_allow_html=True,
    )


def render_portfolio() -> None:
    st.markdown(
        '<div class="hero-copy"><div class="eyebrow">Portfolio</div><h1>Size the risk before you take it.</h1>'
        '<p>Plan entries from a defined loss, see concentration clearly, and maintain holdings without ticker-entry friction.</p></div>',
        unsafe_allow_html=True,
    )
    portfolio = get_session_portfolio()
    current_prices = {}
    for position in portfolio["positions"]:
        try:
            price = cached_price(position["ticker"])
        except Exception:
            price = 0.0
        current_prices[position["ticker"]] = price or position["avg_price"]
    diagnostics = PositionManager.get_portfolio_diagnostics(portfolio, current_prices)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Account equity", headline_money(diagnostics["account_value"]), help=money(diagnostics["account_value"]))
    metric_cols[1].metric("Invested", headline_money(diagnostics["total_market_value"]), percent(diagnostics["invested_pct"]), help=money(diagnostics["total_market_value"]))
    metric_cols[2].metric("Unrealized P&L", headline_money(diagnostics["unrealized_pnl"]), percent(diagnostics["unrealized_pnl_pct"], signed=True), help=money(diagnostics["unrealized_pnl"]))
    metric_cols[3].metric("Risk to stops", percent(diagnostics["portfolio_heat"]), "Portfolio heat", delta_color="off")

    holding_tab, add_tab, settings_tab = st.tabs(["Holdings", "Add position", "Risk setup"])
    with holding_tab:
        render_holdings(portfolio, diagnostics)
    with add_tab:
        render_add_position(portfolio, diagnostics)
    with settings_tab:
        render_risk_setup(portfolio)

    st.markdown(
        '<div class="fine-print">Position sizing is a planning aid. Stops may execute away from the selected price during gaps or illiquid trading.</div>',
        unsafe_allow_html=True,
    )


def render_short_puts() -> None:
    st.markdown(
        '<div class="hero-copy"><div class="eyebrow">Options · phase one</div>'
        '<h1>Rank cash-secured puts by the risk you can actually accept.</h1>'
        '<p>Scan several expirations at once, reject contracts outside your guardrails, and inspect exactly why each surviving put earned its score.</p></div>',
        unsafe_allow_html=True,
    )
    st.info(
        "This first module assumes you already have a neutral-to-bullish thesis and enough cash to accept 100 shares. "
        "Higher IV can improve credit, but it also implies a larger expected move, so IV never overrides the risk gates."
    )

    profile_labels = {
        "Lowest risk": "lowest_risk",
        "Balanced": "balanced",
        "Income focused": "income_focused",
    }
    profile_label = st.segmented_control(
        "Score priority",
        list(profile_labels),
        default="Lowest risk",
        key="short_put_profile_v2",
        help=(
            "Also changes the default assignment ceiling and preferred short-delta band: "
            "lowest risk 15% and 0.01-0.18 targeting 0.08; balanced 28% and 0.10-0.35 targeting 0.22; "
            "income focused 50% and 0.25-0.50 targeting 0.40. You can override the ceiling below."
        ),
    )
    selected_profile = profile_labels[profile_label or "Lowest risk"]
    selected_preferences = ShortPutPreferences.for_profile(selected_profile)
    assignment_default_pct = round(selected_preferences.max_assignment_probability * 100)
    dte_default = (selected_preferences.min_dte, selected_preferences.max_dte)
    st.caption(
        f"{profile_label or 'Lowest risk'} starts with a {assignment_default_pct}% maximum assignment proxy "
        f"and a {dte_default[0]}–{dte_default[1]} DTE entry window targeting {selected_preferences.target_dte} DTE. "
        "Changing the mode loads its defaults; both limits remain editable."
    )

    with st.form("short_put_scan_form"):
        first_row = st.columns([1.35, 1.25, 0.75])
        ticker = first_row[0].text_input(
            "Underlying ticker",
            value="QQQ",
            key="short_put_ticker",
            help="The first release supports one cash-secured short put at a time.",
        ).strip().upper()
        quote_basis_label = first_row[1].selectbox(
            "Quote basis",
            list(OPTION_QUOTE_BASIS_LABELS),
            index=0,
            key="short_put_quote_basis_v1",
            help=(
                "Auto uses current quotes while the provider reports the regular session and otherwise replays "
                "the latest saved regular-session chain. A successful in-session scan saves the chain automatically."
            ),
        )
        selected_quote_basis = OPTION_QUOTE_BASIS_LABELS[quote_basis_label]
        result_limit = first_row[2].selectbox(
            "Recommendations", [3, 5, 8], index=1, key="short_put_limit"
        )

        second_row = st.columns([1.45, 1.0, 1.15])
        dte_window = second_row[0].slider(
            "DTE window",
            min_value=7,
            max_value=90,
            value=dte_default,
            key=f"short_put_dte_{selected_profile}_v2",
            help=(
                f"Entry window for new positions; this mode targets {selected_preferences.target_dte} DTE. "
                "The scanner compares up to 12 eligible expirations. A 21-DTE exit rule is a separate "
                "position-management convention and is not automated here."
            ),
        )
        max_assignment_pct = second_row[1].slider(
            "Maximum assignment proxy",
            min_value=5,
            max_value=60,
            value=int(assignment_default_pct),
            step=1,
            format="%d%%",
            key=f"short_put_assignment_{selected_profile}_v3",
            help=(
                "Hard ceiling for the model-estimated probability of expiring in the money. "
                "The mode supplies the starting value, but you can override it. It is not an exact assignment probability."
            ),
        )
        cash_available = second_row[2].number_input(
            "Cash available to secure",
            min_value=0.0,
            value=0.0,
            step=5_000.0,
            key="short_put_cash_v2",
            help=(
                "One put normally controls 100 shares. The default 0 disables only the capital filter; "
                "the strategy remains cash-secured and each result still shows the cash required."
            ),
        )

        third_row = st.columns([1.0, 1.0, 1.6])
        min_open_interest = third_row[0].number_input(
            "Minimum open interest", min_value=0, value=250, step=50
        )
        max_spread_pct = third_row[1].slider(
            "Maximum bid/ask spread",
            min_value=5,
            max_value=100,
            value=15,
            step=5,
            format="%d%%",
            help="Spread as a percentage of the quote midpoint. The score gives spread more weight than open interest.",
        )
        third_row[2].markdown(
            '<div style="padding-top:1.8rem;color:#627D98;font-size:.86rem">Premium and return use the displayed bid, '
            'which is more conservative than assuming a midpoint fill.</div>',
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button("Scan short puts", width="stretch")

    if submitted:
        if not ticker:
            st.error("Enter an underlying ticker.")
        else:
            try:
                with st.spinner(f"Scanning listed {ticker} puts across the selected DTE window…"):
                    st.session_state.short_put_scan_result = cached_short_put_scan(
                        ticker,
                        selected_quote_basis,
                        selected_profile,
                        int(dte_window[0]),
                        int(dte_window[1]),
                        max_assignment_pct / 100.0,
                        int(min_open_interest),
                        max_spread_pct / 100.0,
                        float(cash_available),
                        int(result_limit),
                    )
            except Exception as exc:
                st.session_state.short_put_scan_result = {"error": str(exc), "recommendations": []}

    result = st.session_state.get("short_put_scan_result")
    if result and not result.get("error") and "spot" in result and not result.get("spot_source"):
        st.session_state.pop("short_put_scan_result", None)
        result = None
    if not result:
        st.markdown(
            '<div class="empty-state"><strong>Ready to scan QQQ.</strong><br>'
            'Set the maximum risk and liquidity limits, then run the scanner.</div>',
            unsafe_allow_html=True,
        )
        return
    if result.get("error"):
        st.error(result["error"])
        return

    iv_context = result.get("iv_context", {})
    iv_rank = iv_context.get("iv_rank")
    summary = st.columns(4)
    summary[0].metric("Underlying", money(result.get("spot")), result.get("symbol", ""), delta_color="off")
    summary[1].metric(
        "52-week IV rank",
        f"{iv_rank * 100:.0f}/100" if iv_rank is not None else "Building history",
        help=iv_context.get("source", "No IV-rank source available"),
    )
    summary[2].metric("30-day realized vol", percent(result.get("historical_volatility")))
    summary[3].metric(
        "Eligible contracts",
        str(result.get("eligible_before_limit", len(result.get("recommendations", [])))),
        f"{result.get('input_count', 0)} scanned",
        delta_color="off",
    )
    st.caption(
        underlying_quote_caption(result) + " "
        f"IV context: {iv_context.get('source', 'unavailable')}. "
        "IV rank is range-based; IV percentile is a separate measure."
    )
    constraints = result.get("constraints", {})
    delta_range = constraints.get("short_delta", [])
    if len(delta_range) == 2:
        st.caption(
            f"{result.get('profile', 'Selected')} profile targets {constraints.get('target_short_delta', 0):.2f} "
            f"short delta and admits {delta_range[0]:.2f}-{delta_range[1]:.2f}. "
            "Maximum assignment proxy is a hard ceiling, not the target."
        )
    if result.get("spot_warning"):
        st.warning(result["spot_warning"])
    if result.get("snapshot_saved"):
        st.caption(
            f"Saved this aligned regular-session chain for later planning "
            f"({result.get('snapshot_marketable_otm_puts', 0)} marketable OTM puts and "
            f"{result.get('snapshot_marketable_otm_calls', 0)} marketable OTM calls)."
        )
    if result.get("snapshot_save_error"):
        st.warning("The live scan completed, but its planning snapshot could not be saved: " + result["snapshot_save_error"])
    elif result.get("snapshot_save_status") == "incomplete_expiration_fetch":
        st.warning("This live result was not saved for replay because at least one selected expiration failed to load.")
    if result.get("failed_expirations"):
        st.warning(
            "Some expirations could not be loaded: " + ", ".join(result["failed_expirations"])
        )

    recommendations = result.get("recommendations", [])
    if not recommendations:
        marketable_otm_puts = result.get("marketable_otm_put_count")
        rejected = result.get("rejected_counts", {})
        failed_preparation = int(rejected.get("invalid or non-marketable quote", 0)) + int(
            rejected.get("not out of the money", 0)
        )
        no_marketable_otm_puts = (
            int(marketable_otm_puts) <= 0
            if marketable_otm_puts is not None
            else bool(result.get("input_count", 0) and failed_preparation == result.get("input_count", 0))
        )
        if no_marketable_otm_puts:
            st.warning(
                f"No marketable OTM put quotes were available in this dataset "
                f"({result.get('marketable_put_count', 0)} marketable puts, "
                f"{result.get('marketable_otm_put_count', 0)} OTM). No contract reached the assignment, delta, "
                "OI, spread, or scoring checks. Retry after the option feed populates; do not relax risk guardrails."
            )
        else:
            st.warning("No contract passed every hard limit. Relax one guardrail at a time and review what was rejected.")
    else:
        best = recommendations[0]
        st.markdown(
            f'<div class="recommendation"><strong>#{best["rank"]} · Sell {esc(result.get("symbol"))} '
            f'{money(best["strike"])} put · {esc(best["expiration"])}</strong>'
            f'<p>Score {best["score"]:.1f}/100 · Bid credit {money(best["premium_per_contract"])} per contract · '
            f'Assignment proxy {percent(best["estimated_assignment_probability"])} · '
            f'Cash to secure {money(best["cash_secured"])}.</p></div>',
            unsafe_allow_html=True,
        )
        st.caption(best["reason"])

        ranking_rows = []
        for item in recommendations:
            ranking_rows.append(
                {
                    "Rank": item["rank"],
                    "Score": item["score"],
                    "Expiration": item["expiration"],
                    "DTE": item["dte"],
                    "Strike": item["strike"],
                    "Bid credit": item["premium_per_contract"],
                    "Assignment proxy": item["estimated_assignment_probability"] * 100,
                    "Break-even": item["break_even"],
                    "BE buffer": item["breakeven_cushion"] * 100,
                    "Annualized bid yield": item["annualized_return_on_cash"] * 100,
                    "IV": item["implied_volatility"] * 100,
                    "Short Δ": item["short_position_delta"],
                    "Θ / day": item["short_position_theta_per_day"],
                    "Open interest": item["open_interest"],
                    "Spread": item["bid_ask_spread_pct"] * 100,
                    "Cash required": item["cash_secured"],
                }
            )
        st.dataframe(
            pd.DataFrame(ranking_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Rank": st.column_config.NumberColumn(format="#%d", width="small"),
                "Score": st.column_config.NumberColumn(format="%.1f", width="small"),
                "Strike": st.column_config.NumberColumn(format="$%.2f"),
                "Bid credit": st.column_config.NumberColumn(format="$%.0f"),
                "Assignment proxy": st.column_config.NumberColumn(format="%.1f%%"),
                "Break-even": st.column_config.NumberColumn(format="$%.2f"),
                "BE buffer": st.column_config.NumberColumn(format="%.1f%%"),
                "Annualized bid yield": st.column_config.NumberColumn(format="%.1f%%"),
                "IV": st.column_config.NumberColumn(format="%.1f%%"),
                "Short Δ": st.column_config.NumberColumn(format="%.3f"),
                "Θ / day": st.column_config.NumberColumn(format="$%.3f"),
                "Spread": st.column_config.NumberColumn(format="%.1f%%"),
                "Cash required": st.column_config.NumberColumn(format="$%.0f"),
            },
        )

        with st.expander("Audit a recommendation", expanded=False):
            selected_index = st.selectbox(
                "Contract",
                range(len(recommendations)),
                format_func=lambda index: (
                    f"#{recommendations[index]['rank']} · {recommendations[index]['expiration']} · "
                    f"${recommendations[index]['strike']:.2f} put"
                ),
            )
            selected = recommendations[selected_index]
            component_labels = {
                "risk_target_fit": "Risk / short-delta target fit",
                "breakeven_safety": "Break-even safety",
                "liquidity": "Liquidity / execution",
                "greek_risk": "Gamma / vega risk",
                "dte_theta": "DTE / theta fit",
                "premium_efficiency": "Premium efficiency",
                "volatility_context": "IV context",
            }
            audit_rows = []
            for key, label in component_labels.items():
                component_score = selected["score_components"][key]
                weight = selected["score_weights"][key]
                audit_rows.append(
                    {
                        "Component": label,
                        "Component score": component_score,
                        "Weight": weight * 100,
                        "Weighted points": component_score * weight,
                    }
                )
            st.dataframe(
                pd.DataFrame(audit_rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "Component score": st.column_config.NumberColumn(format="%.1f / 100"),
                    "Weight": st.column_config.NumberColumn(format="%.0f%%"),
                    "Weighted points": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            greek_cols = st.columns(4)
            greek_cols[0].metric("Short delta", f"{selected['short_position_delta']:+.3f}")
            greek_cols[1].metric("Short theta / day", money(selected["short_position_theta_per_day"], decimals=3))
            greek_cols[2].metric("Short gamma", f"{selected['short_position_gamma']:+.4f}")
            greek_cols[3].metric("Short vega / vol point", f"{selected['short_position_vega_per_vol_point']:+.3f}")
            target_weight = 1.0 - selected.get("risk_target_safety_blend", 0.0)
            st.caption(
                f"Risk-target component: {target_weight:.0%} short-delta target fit and "
                f"{1.0 - target_weight:.0%} assignment safety. Delta-fit score "
                f"{selected.get('delta_target_score', 0.0):.1f}/100; standalone assignment-safety score "
                f"{selected.get('assignment_safety_score', 0.0):.1f}/100."
            )
            st.caption(
                f"Quote quality: {selected['data_quality']}. Max profit {money(selected['max_profit'])}; "
                f"modeled maximum loss if the underlying reaches zero {money(selected['max_loss'])}."
            )

    with st.expander("Filter audit and model limits", expanded=not recommendations):
        if result.get("marketable_otm_put_count") is not None:
            st.caption(
                f"Quote readiness: {result.get('marketable_put_count', 0)} marketable puts; "
                f"{result.get('marketable_otm_put_count', 0)} are OTM and eligible to enter the risk-filter stage."
            )
        rejected = result.get("rejected_counts", {})
        if rejected:
            st.dataframe(
                pd.DataFrame(
                    [{"Rejection reason": reason, "Contracts": count} for reason, count in rejected.items()]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.write("No contracts were rejected by the configured hard filters.")
        st.write(result.get("assignment_note", ""))
        st.write(
            "The score is a suitability ranking under your selected constraints, not a forecast of profit. "
            "It does not yet include trend confirmation, event risk, commissions, taxes, margin rules, early close management, or live order-book depth."
        )

    st.markdown(
        '<div class="fine-print">Research tool only. Quotes may be delayed; confirm the live NBBO and contract details with your broker. '
        'American-style ETF and equity puts can be assigned before expiration. Read the '
        '<a href="https://www.theocc.com/company-information/documents-and-archives/options-disclosure-document" target="_blank">OCC options disclosure document</a> '
        'before trading options.</div>',
        unsafe_allow_html=True,
    )


def render_bullish_combo() -> None:
    st.markdown(
        '<div class="hero-copy"><div class="eyebrow">Options · bullish extension</div>'
        '<h1>Use the put credit to buy upside.</h1>'
        '<p>Jointly rank an out-of-the-money short put and out-of-the-money long call with the same expiration. '
        'The call is paid at its ask from the put sold at its bid.</p></div>',
        unsafe_allow_html=True,
    )
    st.warning(
        "This is a bullish risk reversal, not free leverage. The call adds unlimited upside, but the short put still creates "
        "substantial downside and early-assignment risk. Between the two strikes, both options can expire worthless and only "
        "the small net credit or debit remains."
    )

    profile_labels = {
        "Downside aware": "downside_aware",
        "Balanced bull": "balanced",
        "Upside focused": "upside_focused",
    }
    profile_label = st.segmented_control(
        "Pair objective",
        list(profile_labels),
        default="Balanced bull",
        key="combo_profile_v2",
        help=(
            "Changes short-put safety, the default assignment ceiling, and long-call participation. "
            "Downside aware starts at 15%, balanced bull at 28%, and upside focused at 50%. "
            "You can override the ceiling below."
        ),
    )
    selected_profile = profile_labels[profile_label or "Balanced bull"]
    selected_put_profile = COMBO_PUT_PROFILE_MAP[selected_profile]
    selected_put_preferences = ShortPutPreferences.for_profile(selected_put_profile)
    assignment_default_pct = round(selected_put_preferences.max_assignment_probability * 100)
    dte_default = (selected_put_preferences.min_dte, selected_put_preferences.max_dte)
    st.caption(
        f"{profile_label or 'Balanced bull'} starts with a {assignment_default_pct}% maximum put assignment proxy "
        f"and a {dte_default[0]}–{dte_default[1]} DTE entry window targeting {selected_put_preferences.target_dte} DTE. "
        "Changing the mode loads its defaults; both limits remain editable."
    )

    with st.form("bullish_combo_scan_form"):
        first_row = st.columns([1.35, 1.25, 0.75])
        ticker = first_row[0].text_input("Underlying ticker", value="QQQ", key="combo_ticker").strip().upper()
        quote_basis_label = first_row[1].selectbox(
            "Quote basis",
            list(OPTION_QUOTE_BASIS_LABELS),
            index=0,
            key="combo_quote_basis_v1",
            help=(
                "Auto uses current quotes during the regular session and the latest saved aligned put/call chain outside it."
            ),
        )
        selected_quote_basis = OPTION_QUOTE_BASIS_LABELS[quote_basis_label]
        result_limit = first_row[2].selectbox("Recommendations", [3, 5, 8], index=1, key="combo_limit")

        second_row = st.columns([1.45, 1.0, 1.15])
        dte_window = second_row[0].slider(
            "DTE window",
            min_value=7,
            max_value=90,
            value=dte_default,
            key=f"combo_dte_{selected_profile}_v2",
            help=(
                f"Both legs use the same expiration; the mapped put mode targets {selected_put_preferences.target_dte} DTE. "
                "The 30–45 DTE range is an entry baseline, not an automated exit rule."
            ),
        )
        max_assignment_pct = second_row[1].slider(
            "Maximum put assignment proxy",
            min_value=5,
            max_value=60,
            value=int(assignment_default_pct),
            step=1,
            format="%d%%",
            key=f"combo_assignment_{selected_profile}_v3",
            help=(
                "Hard ceiling for the put's modeled probability of expiring in the money. "
                "The pair mode supplies the starting value, but you can override it."
            ),
        )
        cash_available = second_row[2].number_input(
            "Cash available to secure put",
            min_value=0.0,
            value=0.0,
            step=5_000.0,
            key="combo_cash_v2",
            help=(
                "The default 0 disables only the capital filter. The short put remains cash-secured, "
                "and each pair still shows the cash required if assigned."
            ),
        )

        third_row = st.columns(3)
        min_open_interest = third_row[0].number_input(
            "Minimum OI on both legs", min_value=0, value=100, step=50, key="combo_oi"
        )
        max_spread_pct = third_row[1].slider(
            "Maximum spread on each leg",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            format="%d%%",
            key="combo_spread",
        )
        min_call_delta_pct = third_row[2].slider(
            "Minimum long-call delta",
            min_value=5,
            max_value=45,
            value=20,
            step=5,
            format="%d%%",
            key="combo_call_delta",
            help="Prevents the premium from being spent on an extremely low-probability call.",
        )

        fourth_row = st.columns([1.0, 1.0, 1.5])
        min_utilization_pct = fourth_row[0].slider(
            "Minimum put credit used",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            format="%d%%",
            key="combo_utilization",
        )
        max_extra_debit = fourth_row[1].number_input(
            "Additional call cash",
            min_value=0.0,
            value=0.0,
            step=50.0,
            key="combo_extra_debit",
            help="Default 0 enforces a call fully funded by the short-put credit.",
        )
        fourth_row[2].markdown(
            '<div style="padding-top:1.8rem;color:#627D98;font-size:.86rem">Same expiration is mandatory. '
            'Put credit uses bid; call cost uses ask. This avoids assuming midpoint fills on either leg.</div>',
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button("Rank premium-funded bull pairs", width="stretch")

    if submitted:
        if not ticker:
            st.error("Enter an underlying ticker.")
        else:
            try:
                with st.spinner(f"Jointly scoring {ticker} short puts and long calls…"):
                    st.session_state.bullish_combo_result = cached_bullish_combo_scan(
                        ticker,
                        selected_quote_basis,
                        selected_profile,
                        int(dte_window[0]),
                        int(dte_window[1]),
                        max_assignment_pct / 100.0,
                        int(min_open_interest),
                        max_spread_pct / 100.0,
                        float(cash_available),
                        min_call_delta_pct / 100.0,
                        min_utilization_pct / 100.0,
                        float(max_extra_debit),
                        int(result_limit),
                    )
            except Exception as exc:
                st.session_state.bullish_combo_result = {"error": str(exc), "recommendations": []}

    result = st.session_state.get("bullish_combo_result")
    if result and not result.get("error") and "spot" in result and not result.get("spot_source"):
        st.session_state.pop("bullish_combo_result", None)
        result = None
    if not result:
        st.markdown(
            '<div class="empty-state"><strong>Ready to build a bullish pair.</strong><br>'
            'Keep additional call cash at zero to use only the short-put premium.</div>',
            unsafe_allow_html=True,
        )
        return
    if result.get("error"):
        st.error(result["error"])
        return

    iv_context = result.get("iv_context", {})
    iv_rank = iv_context.get("iv_rank")
    summary = st.columns(4)
    summary[0].metric("Underlying", money(result.get("spot")), result.get("symbol", ""), delta_color="off")
    summary[1].metric(
        "52-week IV rank",
        f"{iv_rank * 100:.0f}/100" if iv_rank is not None else "Building history",
        help=iv_context.get("source", "No IV-rank source available"),
    )
    summary[2].metric(
        "Eligible bull pairs",
        str(result.get("eligible_before_limit", 0)),
        f"{result.get('pair_input_count', 0)} put/call pairs tested",
        delta_color="off",
    )
    summary[3].metric(
        "Eligible short puts",
        str(result.get("eligible_short_put_count", 0)),
        f"{result.get('call_input_count', 0)} calls scanned",
        delta_color="off",
    )
    st.caption(underlying_quote_caption(result))
    put_constraints = result.get("short_put_constraints", {})
    put_delta_range = put_constraints.get("short_delta", [])
    if len(put_delta_range) == 2:
        st.caption(
            f"Short-put leg targets {put_constraints.get('target_short_delta', 0):.2f} delta within "
            f"{put_delta_range[0]:.2f}-{put_delta_range[1]:.2f}; the assignment proxy remains a hard ceiling."
        )
    if result.get("spot_warning"):
        st.warning(result["spot_warning"])
    if result.get("snapshot_saved"):
        st.caption(
            f"Saved this aligned regular-session put/call chain for later planning "
            f"({result.get('snapshot_marketable_otm_puts', 0)} marketable OTM puts and "
            f"{result.get('snapshot_marketable_otm_calls', 0)} marketable OTM calls)."
        )
    if result.get("snapshot_save_error"):
        st.warning("The live scan completed, but its planning snapshot could not be saved: " + result["snapshot_save_error"])
    elif result.get("snapshot_save_status") == "incomplete_expiration_fetch":
        st.warning("This live result was not saved for replay because at least one selected expiration failed to load.")

    recommendations = result.get("recommendations", [])
    if not recommendations:
        combo_readiness = result.get("data_readiness")
        if combo_readiness is None:
            put_rejections = result.get("short_put_rejected_counts", {})
            failed_put_preparation = int(put_rejections.get("invalid or non-marketable quote", 0)) + int(
                put_rejections.get("not out of the money", 0)
            )
            if result.get("short_put_input_count", 0) and failed_put_preparation == result.get(
                "short_put_input_count", 0
            ):
                combo_readiness = "no_marketable_otm_puts"
        if combo_readiness == "no_marketable_otm_puts":
            st.warning(
                f"No marketable OTM short-put quotes were available "
                f"({result.get('marketable_put_count', 0)} marketable puts, "
                f"{result.get('marketable_otm_put_count', 0)} OTM). Pair guardrails were not reached."
            )
        elif combo_readiness == "no_marketable_otm_calls":
            st.warning(
                f"No marketable OTM long-call quotes were available "
                f"({result.get('marketable_call_count', 0)} marketable calls, "
                f"{result.get('marketable_otm_call_count', 0)} OTM). Pair scoring cannot run on this dataset."
            )
        else:
            st.warning(
                "No same-expiry pair passed the put-risk, call-delta, liquidity, and funding limits. "
                "Review the audit before adding extra call cash."
            )
    else:
        best = recommendations[0]
        cash_flow_label = (
            f"net credit {money(best['net_credit'])}"
            if best["net_credit"] >= 0
            else f"net debit {money(best['net_debit'])}"
        )
        st.markdown(
            f'<div class="recommendation"><strong>#{best["rank"]} · Sell {esc(result.get("symbol"))} '
            f'{money(best["put_strike"])} put / Buy {money(best["call_strike"])} call · {esc(best["expiration"])}</strong>'
            f'<p>Put credit {money(best["put_credit"])} · Call cost {money(best["call_cost"])} · {esc(cash_flow_label)} · '
            f'Pair score {best["score"]:.1f}/100.</p></div>',
            unsafe_allow_html=True,
        )
        st.caption(best["reason"])

        risk_cols = st.columns(4)
        risk_cols[0].metric(
            "Put short delta",
            f"{best['short_put_delta']:.3f}",
            f"{percent(best['estimated_assignment_probability'])} assignment proxy",
            delta_color="off",
        )
        risk_cols[1].metric("Call expiry-ITM proxy", percent(best["long_call_probability_itm"]))
        risk_cols[2].metric("Combined delta", f"{best['net_delta']:+.3f}")
        risk_cols[3].metric("Modeled max loss", headline_money(best["max_loss"]), help=money(best["max_loss"]))

        ranking_rows = []
        for item in recommendations:
            ranking_rows.append(
                {
                    "Rank": item["rank"],
                    "Score": item["score"],
                    "Expiration": item["expiration"],
                    "DTE": item["dte"],
                    "Put strike": item["put_strike"],
                    "Put credit": item["put_credit"],
                    "Put delta": item["short_put_delta"],
                    "Put assign": item["estimated_assignment_probability"] * 100,
                    "Call strike": item["call_strike"],
                    "Call cost": item["call_cost"],
                    "Call delta": item["long_call_delta"],
                    "Call ITM": item["long_call_probability_itm"] * 100,
                    "Net credit": item["net_credit"],
                    "Net delta": item["net_delta"],
                    "+10% expiry P&L": item["profit_at_up_10_pct"],
                    "-10% expiry P&L": item["profit_at_down_10_pct"],
                    "Cash required": item["capital_required"],
                }
            )
        st.dataframe(
            pd.DataFrame(ranking_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Rank": st.column_config.NumberColumn(format="#%d", width="small"),
                "Score": st.column_config.NumberColumn(format="%.1f", width="small"),
                "Put strike": st.column_config.NumberColumn(format="$%.2f"),
                "Put credit": st.column_config.NumberColumn(format="$%.0f"),
                "Put delta": st.column_config.NumberColumn(format="%.3f"),
                "Put assign": st.column_config.NumberColumn(format="%.1f%%"),
                "Call strike": st.column_config.NumberColumn(format="$%.2f"),
                "Call cost": st.column_config.NumberColumn(format="$%.0f"),
                "Call delta": st.column_config.NumberColumn(format="%.3f"),
                "Call ITM": st.column_config.NumberColumn(format="%.1f%%"),
                "Net credit": st.column_config.NumberColumn(format="$%.0f"),
                "Net delta": st.column_config.NumberColumn(format="%.3f"),
                "+10% expiry P&L": st.column_config.NumberColumn(format="$%.0f"),
                "-10% expiry P&L": st.column_config.NumberColumn(format="$%.0f"),
                "Cash required": st.column_config.NumberColumn(format="$%.0f"),
            },
        )
        st.caption(
            "The ±10% columns are mechanical expiration payoffs from today's spot, not forecasts and not probability-weighted returns."
        )

        with st.expander("Audit a bullish pair", expanded=False):
            selected_index = st.selectbox(
                "Pair",
                range(len(recommendations)),
                format_func=lambda index: (
                    f"#{recommendations[index]['rank']} · P{recommendations[index]['put_strike']:.0f} / "
                    f"C{recommendations[index]['call_strike']:.0f} · {recommendations[index]['expiration']}"
                ),
                key="combo_audit_pair",
            )
            selected = recommendations[selected_index]
            component_labels = {
                "downside_safety": "Short-put downside safety",
                "upside_participation": "Long-call upside participation",
                "funding_efficiency": "Premium funding efficiency",
                "liquidity": "Two-leg liquidity",
                "net_greeks": "Combined Greek balance",
                "iv_skew": "Put-versus-call IV skew",
                "dte_fit": "DTE fit",
            }
            audit_rows = []
            for key, label in component_labels.items():
                component_score = selected["score_components"][key]
                weight = selected["score_weights"][key]
                audit_rows.append(
                    {
                        "Component": label,
                        "Component score": component_score,
                        "Weight": weight * 100,
                        "Weighted points": component_score * weight,
                    }
                )
            st.dataframe(
                pd.DataFrame(audit_rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "Component score": st.column_config.NumberColumn(format="%.1f / 100"),
                    "Weight": st.column_config.NumberColumn(format="%.0f%%"),
                    "Weighted points": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            greek_cols = st.columns(4)
            greek_cols[0].metric("Net delta", f"{selected['net_delta']:+.3f}")
            greek_cols[1].metric("Net theta / day", money(selected["net_theta_per_day"], decimals=3))
            greek_cols[2].metric("Net gamma", f"{selected['net_gamma']:+.4f}")
            greek_cols[3].metric("Net vega / vol point", f"{selected['net_vega_per_vol_point']:+.3f}")
            st.write(
                f"At expiration, downside break-even is {money(selected['lower_break_even'])}. "
                f"The long call begins adding intrinsic upside above {money(selected['call_activation_price'])}. "
                f"If the underlying reaches zero, modeled loss is {money(selected['max_loss'])}."
            )

    with st.expander("Pair filter audit and model limits", expanded=not recommendations):
        if result.get("marketable_otm_put_count") is not None:
            st.caption(
                f"Quote readiness: {result.get('marketable_otm_put_count', 0)} marketable OTM puts and "
                f"{result.get('marketable_otm_call_count', 0)} marketable OTM calls."
            )
        pair_rejections = result.get("rejected_counts", {})
        if pair_rejections:
            st.markdown("**Call and pair filters**")
            st.dataframe(
                pd.DataFrame(
                    [{"Rejection reason": reason, "Count": count} for reason, count in pair_rejections.items()]
                ),
                hide_index=True,
                width="stretch",
            )
        put_rejections = result.get("short_put_rejected_counts", {})
        if put_rejections:
            st.markdown("**Short-put filters**")
            st.dataframe(
                pd.DataFrame(
                    [{"Rejection reason": reason, "Contracts": count} for reason, count in put_rejections.items()]
                ),
                hide_index=True,
                width="stretch",
            )
        st.write(result.get("strategy_note", ""))
        st.write(result.get("assignment_note", ""))
        st.write(
            "The model does not yet confirm a bull regime, detect events, model early exits or rolls, include commissions and taxes, "
            "or estimate whether a multi-leg limit order will fill."
        )

    st.markdown(
        '<div class="fine-print">Research tool only. A long call plus a short put can behave like leveraged long exposure, '
        'and the short put can still require purchase of 100 shares. Confirm live quotes, buying-power treatment, and assignment procedures with your broker. '
        'Read the <a href="https://www.optionseducation.org/strategies/all-strategies/synthetic-long-stock" target="_blank">OIC synthetic-long risk overview</a> '
        'and the <a href="https://www.theocc.com/company-information/documents-and-archives/options-disclosure-document" target="_blank">OCC options disclosure document</a>.</div>',
        unsafe_allow_html=True,
    )


def render_options() -> None:
    short_put_tab, bullish_combo_tab = st.tabs(["Short put only", "Put premium → long call"])
    with short_put_tab:
        render_short_puts()
    with bullish_combo_tab:
        render_bullish_combo()


workspace = top_navigation()
if workspace == "Portfolio":
    render_portfolio()
elif workspace == "Options":
    render_options()
else:
    render_company_value()
