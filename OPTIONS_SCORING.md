# Options scoring model

## Phase 1 scope

The foundation ranks one **cash-secured short put** at a time. It assumes the
user already has a neutral-to-bullish view of the underlying and is willing and
able to buy 100 shares at the strike if assigned.

The Streamlit app exposes this model under **Options → Short put only**.

## Ranking flow

The model uses two stages:

1. Hard filters reject contracts outside the user's DTE window, assignment-risk
   ceiling, open-interest minimum, maximum bid/ask spread, cash limit, or OTM
   requirement. A zero or non-marketable bid is also rejected.
2. Surviving contracts receive a 0-100 suitability score. The result table and
   audit panel expose every component, weight, and weighted contribution.

The maximum assignment proxy is a hard ceiling, not a target. Changing the mode
loads its default ceiling, which remains editable. Each profile also defines an
eligible and preferred absolute short-put delta:

| Profile | Default maximum assignment proxy | Eligible short delta | Target short delta | Objective |
| --- | ---: | ---: | ---: | --- |
| Lowest risk | 15% | 0.01-0.18 | 0.08 | Prefer low directional exposure below the assignment ceiling |
| Balanced | 28% | 0.10-0.35 | 0.22 | Balance premium with a moderate probability of stock acquisition |
| Income focused | 50% | 0.25-0.50 | 0.40 | Seek materially higher carry while accepting near-ATM risk |

Contracts outside the profile's delta range are rejected. The risk-target score
rewards proximity to the target; it does not keep awarding an income-focused
contract merely for being far safer than the user's maximum.

### Entry DTE defaults

All three short-put profiles start with an editable **30-45 DTE entry window**.
Their scoring targets remain distinct: 35 DTE for lowest risk, 38 DTE for
balanced, and 30 DTE for income focused. If the user selects a custom window
that excludes the profile target, the target is clamped to the nearest edge of
that window instead of being replaced by the window midpoint.

The scanner ranks new entries; it does not implement a 21-DTE management rule.
Managing an existing position around 21 DTE is a separate exit convention and
should not make 21 DTE the default lower boundary for opening a position.

Cash available starts at `0`, which disables the capital filter rather than the
cash-secured obligation. The scanner still calculates and displays the cash
required for one 100-share put contract. Entering a positive amount turns the
capital filter on.

The executable premium assumption is the displayed **bid**, not the midpoint.
Annualized bid yield is:

```text
(bid / strike) * (365 / DTE)
```

It is a comparison rate, not an expected return. It excludes fill improvement,
fees, taxes, compounding, and position-management decisions.

## Underlying quote policy

Every contract in a scan uses one underlying-price snapshot. The selector uses
the provider's timestamped quote for the active session: pre-market,
regular-market, or post-market. If explicit session quotes are unavailable, it
tries the latest one-minute trade and the provider's last-price field.
`previousClose` is a flagged last resort only.

Results expose the selected provider field, session, market state, quote
timestamp, retrieval time, and regular/extended-hours reference prices. An
extended-hours warning is shown because the underlying may continue trading
after listed option quotes stop updating. The Streamlit scan cache lasts 30
seconds; it does not turn a delayed provider quote into exchange-direct data.

### Quote-basis modes and planning snapshots

Both option scanners expose three quote bases:

- **Auto** uses the provider's current chain while it reports the regular
  session. Outside that session it replays the latest saved regular-session
  snapshot.
- **Live current session** requires the provider to report a regular session;
  it will not combine a pre/post-market underlying with an old option chain.
- **Latest saved regular session** replays the newest local point-in-time
  snapshot without fetching a live chain.

A successful regular-session scan with at least one marketable **OTM** put and no
failed selected expirations saves the
spot, historical volatility, dividend yield, risk-free rate, IV context, and
both put and call chains under `.cache/options_snapshots/<symbol>/`. The daily
file is replaced by the newest successful scan so every saved dataset remains
internally aligned. Snapshot DTE is calculated from the captured market date,
not the replay date. If a later replay asks for an expiration that was not
captured by that scan, the tool requests a matching live-session scan instead
of silently ranking an incomplete window.

Snapshot recommendations are historical planning results, not executable
quotes. The UI displays the capture time and labels the bid/ask data as stale.
If a live or saved chain has marketable puts but none are OTM, the UI classifies
that as unavailable strategy data; it does not suggest relaxing assignment,
delta, liquidity, or scoring guardrails that no contract reached.
For a symbol that has never been scanned successfully during a regular session,
Auto and snapshot modes explain that a live scan is required rather than
fabricating premiums from `lastPrice` or zero bids.

## Score components

| Component | Lowest risk | Balanced | Income focused | Purpose |
| --- | ---: | ---: | ---: | --- |
| Risk / short-delta target fit | 28% | 22% | 18% | Blends target-delta proximity with assignment safety; income focused is 90% target fit |
| Break-even safety | 17% | 15% | 6% | Rewards a lower modeled probability of finishing below strike minus bid |
| Liquidity / execution | 20% | 18% | 14% | Prioritizes a tight spread, then OI and volume |
| Gamma / vega risk | 12% | 10% | 6% | Penalizes larger adverse short-gamma and short-vega exposure |
| DTE / theta fit | 10% | 12% | 16% | Balances the profile's target DTE with theta earned per secured dollar |
| Premium efficiency | 7% | 13% | 26% | Rewards annualized bid yield up to a profile-specific target |
| IV context | 6% | 10% | 14% | Uses IV rank and the contract IV-to-realized-volatility ratio |

Higher IV is therefore not treated as free income. It can raise the IV-context
and premium components, but it also increases the expected move, modeled ITM
probability, and mark-to-market vega risk.

The IV component is context, not a direct mean-reversion forecast. It blends a
52-week IV-rank measure with contract IV relative to realized volatility. A
future version should validate IV-minus-realized-volatility carry and subsequent
volatility changes historically rather than treating high IV alone as an edge.

## Phase 2: put premium funds a long call

The bullish extension jointly ranks two legs with the **same expiration**:

- sell one OTM cash-secured put at its displayed bid; and
- buy one OTM call at its displayed ask.

The short-put leg inherits the selected put profile's assignment ceiling and
delta band. Downside aware maps to lowest risk (15%), balanced bull maps to
balanced (28%), and upside focused maps to income focused (50%). In particular,
the upside-focused pair uses the income-focused 0.25-0.50 put-delta range and
targets 0.40 before the call leg is paired and scored. Its cash-available field
also starts at `0`, disabling only the capital filter.

Both legs start with the same 30-45 DTE entry window because the pair requires a
shared expiration. The pair inherits the mapped put profile's DTE target; this
is a baseline for the current model, not a claim that the long-call leg has
already been independently optimized for duration.

With the default additional-call budget of zero, the call ask cannot exceed the
put bid. This is a bullish risk reversal. A same-strike long call and short put
would replicate long stock; using a higher call strike creates a flat payoff
region between the put and call strikes. The OIC describes both the synthetic
long relationship and its potentially substantial downside in its
[synthetic-long stock overview](https://www.optionseducation.org/strategies/all-strategies/synthetic-long-stock).

The expiration payoff per pair is:

```text
100 * (max(expiry spot - call strike, 0)
       - max(put strike - expiry spot, 0)
       + put bid - call ask)
```

The call provides unlimited theoretical upside above its strike, but it does
not remove the short put's assignment obligation. If the underlying finishes
between the two strikes, both options expire worthless and only the initial net
credit or debit remains. Maximum modeled loss occurs if the underlying reaches
zero:

```text
100 * (put strike - net credit per share)
```

### Pair score components

| Component | Downside aware | Balanced bull | Upside focused | Purpose |
| --- | ---: | ---: | ---: | --- |
| Short-put downside safety | 32% | 24% | 18% | Preserves assignment and break-even discipline |
| Long-call upside participation | 23% | 32% | 40% | Rewards useful call delta, ITM proxy, and reachable strike |
| Premium funding efficiency | 15% | 15% | 14% | Rewards using the put credit without extra debit |
| Two-leg liquidity | 15% | 12% | 10% | Scores spread, OI, and volume on both legs |
| Combined Greek balance | 8% | 9% | 10% | Evaluates net theta, gamma, and vega rather than either leg alone |
| Put-versus-call IV skew | 4% | 5% | 5% | Rewards selling relatively richer put IV and buying relatively cheaper call IV |
| DTE fit | 3% | 3% | 3% | Retains the selected time-window preference |

Hard filters run before this score. They include the short-put assignment
ceiling, cash required, OI and spreads on both legs, minimum call delta, minimum
put-credit utilization, and maximum additional call cash.

## Assignment proxy and Greeks

Greeks use Black-Scholes-Merton with the risk-free rate, dividend yield, contract
IV, and calendar time to expiration. The displayed short-position signs are:

- positive delta;
- positive theta;
- negative gamma; and
- negative vega.

The assignment proxy is the Black-Scholes risk-neutral probability that the put
finishes in the money, `N(-d2)`. This is more explicit than treating absolute
delta as an exact probability, but it is still **not an actual assignment
probability**. Equity and ETF options are generally American style; an option
holder can exercise early, and expiration exercise instructions and after-hours
moves can produce outcomes the model does not predict. See the
[OIC assignment guide](https://www.optionseducation.org/optionsoverview/exercising-options)
and the [OCC options disclosure document](https://www.theocc.com/company-information/documents-and-archives/options-disclosure-document).

## IV rank

Range-based IV rank is:

```text
(current IV - 52-week IV low) / (52-week IV high - 52-week IV low)
```

IV percentile is calculated separately and is not substituted for IV rank.

- QQQ uses `^VXN` as a clearly labeled 52-week Nasdaq-100 volatility proxy.
  VXN measures Nasdaq-100 rather than QQQ option IV, so this is market context,
  not an exact QQQ IV history.
- SPY, IWM, and DIA use their named volatility-index proxies when available.
- Other symbols build a local daily ATM-IV history. IV rank remains unavailable
  until at least 20 daily snapshots exist rather than fabricating a result from
  historical realized volatility.

Each contract's quoted IV and its ratio to 30-day realized volatility remain
visible even when range-based IV rank is unavailable.

## Run and verify

```powershell
pip install -r requirements.txt
streamlit run app.py
python -m unittest discover -s tests -v
python main.py --ticker QQQ --bull-combo
python main.py --ticker QQQ --options --quote-basis previous_session
```

## Planned sequence

1. Add an explicit bullish-regime gate and event-risk calendar while preserving
   the current option-selection audit.
2. Add exit-management and roll rules, then backtest score buckets with
   point-in-time chains and realistic bid/ask fills.
3. Add broker-grade live quotes and account-aware contract sizing before using
   rankings for order preparation.
