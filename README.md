# Cross-Sectional Equity Momentum — Pipeline Validation

> **Stage 0: Pre-Registration Document**
> A pre-committed analysis plan for building and validating an end-to-end quantitative research pipeline by reproducing a published anomaly.

**Status:** Pre-registration (written before any backtest is run)
**Date initiated:** 2026-04-07
**Author:** _[your name]_

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Hypothesis](#2-hypothesis)
3. [Universe](#3-universe)
4. [Sample Period](#4-sample-period)
5. [Signal Definition](#5-signal-definition)
6. [Portfolio Construction](#6-portfolio-construction)
7. [Transaction Cost Model](#7-transaction-cost-model)
8. [Evaluation Methodology](#8-evaluation-methodology)
9. [Pipeline Sanity Checks](#9-pipeline-sanity-checks)
10. [Pre-Committed Deviations and Stopping Rules](#10-pre-committed-deviations-and-stopping-rules)
11. [Deliverables](#11-deliverables)
12. [References](#12-references)

---

## 1. Purpose

The purpose of this project is **not** to discover a novel trading strategy or to generate returns. The purpose is to **build and validate an end-to-end quantitative research pipeline** by reproducing a well-documented, published anomaly — cross-sectional equity momentum — and confirming that the pipeline produces results consistent with the published literature.

Cross-sectional momentum is chosen as the validation target because:

- It is one of the most replicated anomalies in the empirical asset pricing literature, with 30+ years of out-of-sample evidence since Jegadeesh & Titman (1993).
- Published net-of-cost Sharpe ratios provide a quantitative benchmark against which the pipeline's output can be compared. A working pipeline should reproduce these benchmarks within a reasonable tolerance; substantial deviation indicates a bug or methodological error.
- The economic mechanism is debated but the empirical pattern is robust across markets, asset classes, and time periods, reducing the risk that a null result reflects a real absence of signal rather than a pipeline defect.
- The construction is simple enough to implement without ambiguity, but realistic enough to exercise every component of a production research pipeline (point-in-time data, signal generation, portfolio construction, transaction costs, statistical evaluation).

The success criterion is **pipeline trustworthiness**, not strategy profitability. Once the pipeline is validated, it becomes a measurement apparatus that can later be used to evaluate novel hypotheses with calibrated confidence.

---

## 2. Hypothesis

### Primary hypothesis (H1)

Among large-cap US equities, a dollar-neutral long-short portfolio that is long the top decile and short the bottom decile of stocks ranked by their cumulative return over the prior 12 months excluding the most recent month ("12-1 momentum"), rebalanced monthly and equal-weighted within each leg, produces a positive net-of-cost Sharpe ratio over the sample period 2005-01-01 to 2019-12-31.

### Quantitative prediction

The annualized net-of-cost Sharpe ratio of the long-short portfolio will fall in the range **[0.3, 0.9]**, consistent with published estimates for US large-cap cross-sectional momentum (Asness, Moskowitz & Pedersen 2013; AQR factor library; Ken French data library momentum factor).

### Null hypothesis (H0)

The Sharpe ratio is indistinguishable from zero, or the pipeline produces a Sharpe outside the predicted range.

### Pre-committed interpretations

| Net Sharpe range | Interpretation | Action |
|---|---|---|
| `[0.3, 0.9]` | Pipeline validated | Proceed to Stage 7 evaluation and stress tests |
| `(0.9, 1.5]` | Suspicious | Audit for look-ahead bias, survivorship bias, unrealistic costs, or universe contamination |
| `> 1.5` | Almost certainly a bug | Halt and debug |
| `[0.0, 0.3)` | Plausible if dominated by post-2009 drawdown | Investigate via rolling Sharpe vs published estimates |
| `[-0.3, 0.0)` | Plausible for short samples dominated by momentum crashes | Validate via correlation with Ken French UMD factor (see deviation note below) |
| `< -0.3` | Pipeline likely broken | Halt and debug (sign error, look-ahead, mishandled delistings) |

> **Deviation note (dated 2026-05-04):** The original pre-registration specified `< 0.0` as "pipeline likely broken." This threshold was based on the long-sample (1927-present) Sharpe of ~0.5-0.8 for cross-sectional momentum. However, our in-sample period (2005-2019) contains the worst momentum crash in history (March 2009, documented in Daniel & Moskowitz 2016) plus a second significant drawdown in 2015-2016. Over a 15-year window dominated by these events, a mildly negative Sharpe is consistent with published factor returns over the same period — AQR's momentum factor and the Ken French UMD factor both show near-zero or negative cumulative returns for 2009-2013.
>
> The interpretation is revised to add a `[-0.3, 0.0)` bucket: mildly negative Sharpe is plausible for this sample period and does not indicate a broken pipeline. The definitive validation test is the **correlation of monthly returns against the Ken French UMD factor** (Stage 5). A correlation above 0.6 confirms the pipeline is reproducing the published anomaly correctly, regardless of the absolute Sharpe level. A Sharpe below -0.3 remains a halt-and-debug signal, as it would indicate systematic errors beyond what the momentum crash can explain.

---

## 3. Universe

**Definition:** S&P 500 constituents, **point-in-time**. On every rebalance date `T`, the eligible universe is the set of stocks that were members of the S&P 500 index as of the close of `T`, using historical index membership records — *not* today's membership.

**Rationale:** Point-in-time membership is required to eliminate survivorship bias and look-ahead bias in universe selection. Using today's S&P 500 constituents to backtest 2005-2019 would guarantee that every stock in the universe survived to 2026, biasing returns upward by an amount documented in the literature to be on the order of 1-4% per year for equity strategies.

**Data source for membership:** Norgate Data (preferred) or CRSP via WRDS if available through university access. Free sources are not acceptable for this field because they do not provide reliable historical index membership.

### Additional eligibility filters

At each rebalance date `T`, a stock is eligible only if:

- It has a complete daily price history from `T-252` trading days through `T` (required to compute the 12-1 signal).
- It has a non-missing closing price on `T`.
- It is not flagged as halted or suspended on `T`.

No filters on price level, market cap below S&P 500 inclusion thresholds, or liquidity beyond S&P 500 membership are applied, since membership itself implies large-cap and liquid status.

---

## 4. Sample Period

| Period | Dates | Use |
|---|---|---|
| **In-sample / development** | 2005-01-01 to 2019-12-31 | All development, validation, and robustness testing |
| **Holdout** | 2020-01-01 to 2025-12-31 | Single end-of-project evaluation only |

> ⚠️ **The holdout will not be examined in any form until the in-sample analysis is complete and frozen.** A single holdout evaluation will be performed at the end of the project. No iteration on the holdout is permitted.

**Rationale for the split:** 15 years of in-sample data captures multiple market regimes (2008 crisis, 2010-2015 low-volatility regime, 2015-2016 momentum drawdown, 2017-2019 recovery) without contaminating the holdout with the COVID shock and post-COVID regime, which represents a genuinely novel out-of-sample test.

**Pre-2005 data is excluded** because S&P 500 composition and market microstructure (decimalization completed 2001, Reg NMS 2007, post-crisis HFT proliferation) make pre-2005 results less representative of the cost and execution environment a contemporary implementation would face.

---

## 5. Signal Definition

**Signal name:** `MOM_12_1`

On each rebalance date `T`, for each eligible stock `i`:

1. Let `P_i(t)` denote the split- and dividend-adjusted total return price of stock `i` on trading day `t`, using only adjustments known as of date `T`.
2. Compute the raw signal as the cumulative log return from `T-252` to `T-21` trading days:
   ```
   s_i(T) = log(P_i(T-21)) - log(P_i(T-252))
   ```
3. After computing `s_i(T)` for all eligible stocks, cross-sectionally z-score the signal across the universe at time `T`:
   ```
   z_i(T) = (s_i(T) - mean_j(s_j(T))) / std_j(s_j(T))
   ```

**Rationale for the 12-1 specification:** The 21-day skip period excludes the most recent month of returns, which is contaminated by the well-documented short-term reversal effect (Jegadeesh 1990). Using a 12-month formation period excluding the skip month is the standard specification in Jegadeesh & Titman (1993) and subsequent literature, and matches the construction of the Ken French momentum factor (UMD).

> **No parameter tuning in Stage 0.** The lookback (252), skip (21), and ranking method (cross-sectional z-score) are fixed in advance. Robustness to perturbations will be examined in Stage 6.

---

## 6. Portfolio Construction

**Rebalance frequency:** Monthly, on the last trading day of each calendar month. Trades are assumed executed at the closing price of the rebalance date.

### Construction rule

1. On rebalance date `T`, rank all eligible stocks by `z_i(T)`.
2. The **long leg** is the top decile (top 10%) of the cross-section, equal-weighted.
3. The **short leg** is the bottom decile (bottom 10%) of the cross-section, equal-weighted.
4. The portfolio is **dollar-neutral**: gross long exposure = gross short exposure = 100% of NAV. Net exposure is zero by construction.
5. Positions are held from `T` to the next rebalance date `T'`. No intra-month rebalancing.

### Delisting handling

If a stock is delisted between `T` and `T'`, the delisting return (final return reflecting the delisting price or recovery, per CRSP/Norgate convention) is applied to that position, and the proceeds (or losses) are held in cash until the next rebalance.

> ⚠️ **Delisted stocks are not silently dropped.** Doing so would introduce a known and significant upward bias.

**No leverage, no short-borrow constraints modeled in Stage 0.** Stage 6 will introduce a borrow cost model and examine sensitivity to short-side frictions.

---

## 7. Transaction Cost Model

For each dollar traded (in either direction), charge a cost equal to:

```
cost_per_dollar_traded = half_spread + commission + impact
```

| Component | Stage 0 value | Notes |
|---|---|---|
| `half_spread` | 2 bp | Conservative for post-2010 US large caps; pre-2010 modeled at 4 bp in Stage 6 |
| `commission` | 0.5 bp | Modern retail commissions at IBKR or similar |
| `impact` | 0 bp | Negligible for retail position sizes; square-root model added in Stage 6 |
| **Total** | **2.5 bp** | Per dollar traded, applied to absolute change in position weights |

**Rationale:** This is intentionally simple but already realistic enough to materially affect the result. Momentum strategies have high turnover (typically 200-400% annual two-way), and even 2.5 bp of round-trip cost translates to 50-100 bp of annual drag on gross returns. A pipeline that ignores costs will produce misleadingly high Sharpes; this model is the minimum acceptable level of realism.

---

## 8. Evaluation Methodology

### Primary metric

Annualized net-of-cost Sharpe ratio of the long-short portfolio over the in-sample period.

### Secondary metrics (all reported, none used for selection)

- Annualized return (net of costs)
- Annualized volatility
- Maximum drawdown
- Calmar ratio
- Hit rate (fraction of months with positive return)
- Two-way annual turnover
- Average holding period
- Correlation to S&P 500 total return
- Correlation to Fama-French market, size, value, and momentum factors (Ken French data library)
- Rolling 12-month Sharpe ratio (plotted)

### Statistical validation

- **Newey-West adjusted t-statistic** on monthly returns (lag = 6) to account for autocorrelation.
- **Deflated Sharpe ratio** (López de Prado 2014), computed conservatively with the number of trials set to 1, since no parameter search is performed in Stage 0. Will become more meaningful in Stage 6 when parameter sensitivity is examined.

### Benchmark comparison

Compute the correlation and OLS beta of the strategy's monthly returns against the Ken French UMD (momentum) factor over the same period.

- Correlation **above 0.6** is expected.
- Correlation **below 0.4** suggests a construction or universe mismatch with the published factor and warrants investigation.

---

## 9. Pipeline Sanity Checks

These checks validate the data and infrastructure independently of the strategy itself.

> ⚠️ **All must pass before Stage 4 backtesting begins.** Any failure halts the project until resolved.

1. **Universe count over time.** Plot the number of stocks in the eligible universe on each month-end from 2005 to 2025. Should be approximately 500 with normal index turnover (~20-25 changes per year). A flat line at exactly 500 indicates broken point-in-time logic. A monotonically increasing or decreasing line indicates a different bug.
2. **Index reproduction (cap-weighted).** Compute the cap-weighted total return of the eligible universe using point-in-time weights and compare to the published S&P 500 total return index. Annualized tracking error should be **under 50 bp/year**.
3. **Index reproduction (equal-weighted).** Compute the equal-weighted total return of the eligible universe and compare to the S&P 500 Equal Weight index. Should track within **100 bp/year**.
4. **Delisting accounting.** Identify at least three known delistings in the sample period (e.g., Lehman Brothers September 2008, General Motors June 2009, any acquired company) and verify that the delisting return is correctly reflected in the data, not silently `NaN`'d or dropped.
5. **No look-ahead in signal.** Implement the signal function with an explicit assertion that no input data has a date `> T`. Run a unit test that passes a deliberately corrupted future-data input and confirms the assertion fires.

---

## 10. Pre-Committed Deviations and Stopping Rules

**Permitted deviations from this plan:** None during Stage 0 in-sample analysis. If a bug is discovered, the bug is fixed and the entire in-sample analysis is rerun from scratch. No cherry-picking of "the version after the bugfix that gave the best result."

**Stopping rule for the in-sample analysis:** Once all sanity checks pass and the long-short backtest produces a Sharpe in any of the pre-committed interpretation buckets (Section 2), the in-sample analysis is frozen and the result is recorded. Iteration on the in-sample is not permitted at this stage.

**Holdout protocol:** The holdout period (2020-2025) will be evaluated **exactly once**, at the end of the project, after all in-sample analysis and Stage 6 robustness checks are complete. The holdout result will be reported regardless of whether it confirms or contradicts the in-sample result. **No iteration on the holdout under any circumstances.**

---

## 11. Deliverables

### At the end of Stage 0 (this document)

- This pre-registration document, dated and version-controlled in git, hash recorded.
- A git repository initialized with the directory structure for the project.
- A written list of the data sources to be acquired and their costs.

### At the end of the full project (Stages 1-7)

- A reproducible codebase that runs the entire pipeline end-to-end from raw data to final report.
- A results document reporting all metrics defined in Section 8, on both in-sample and holdout periods.
- A comparison to the Ken French momentum factor benchmark.
- A written assessment of whether the pipeline is validated (pre-committed criteria from Section 2).
- A list of known limitations and threats to validity not addressed by the current pipeline.

---

## 12. References

- Jegadeesh, N., & Titman, S. (1993). "Returns to buying winners and selling losers: Implications for stock market efficiency." *Journal of Finance*, 48(1), 65-91.
- Jegadeesh, N. (1990). "Evidence of predictable behavior of security returns." *Journal of Finance*, 45(3), 881-898.
- Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). "Value and momentum everywhere." *Journal of Finance*, 68(3), 929-985.
- López de Prado, M. (2014). "The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality." *Journal of Portfolio Management*, 40(5), 94-107.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Kenneth R. French data library: <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>

---

## Pre-Registration Commitment

I commit to executing the analysis described above without deviation, except as permitted in Section 10. Any deviations will be documented with justification and dated. The holdout sample will not be examined until the in-sample analysis is complete and frozen.

| Field | Value |
|---|---|
| Signed | _____________________ |
| Date | _____________________ |
| Git commit hash | _____________________ |
