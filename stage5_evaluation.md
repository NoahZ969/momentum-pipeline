# Stage 5: Evaluation

> **Goal:** Validate the pipeline by comparing our momentum strategy's returns to published benchmarks, running factor regressions, and computing all metrics specified in the Stage 0 pre-registration.
>
> **This is the definitive pipeline validation step.** The key test is the correlation of our monthly returns against the Ken French UMD factor. A correlation above 0.6 confirms we are reproducing the published momentum anomaly correctly.
>
> **Estimated time:** 3-5 days part-time.

---

## Table of Contents

1. [Key Concepts](#1-key-concepts)
2. [Overview](#2-overview)
3. [Implementation](#3-implementation)
4. [Deliverables Checklist](#4-deliverables-checklist)

---

## 1. Key Concepts

### Ken French UMD Factor

Kenneth French (Dartmouth) maintains a freely available [data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) of factor returns computed from the full universe of US stocks. The **UMD** ("Up Minus Down") factor is his momentum factor: each month, it goes long stocks with high prior returns and shorts stocks with low prior returns, using a 12-1 month lookback — essentially the same signal we're computing. If our strategy is built correctly, its monthly returns should be highly correlated with UMD. This is our ground truth.

### Fama-French 4-Factor Model

The standard model in empirical asset pricing for explaining what drives a portfolio's returns. It decomposes any portfolio's excess return (return minus the risk-free rate) into exposure to four systematic factors:

- **Mkt-RF (Market):** The return of the broad stock market minus the risk-free rate. A loading of 0 means the portfolio is market-neutral (which ours should be, since it's dollar-neutral).
- **SMB (Small Minus Big):** The return difference between small-cap and large-cap stocks. Tells us if our strategy is inadvertently betting on company size.
- **HML (High Minus Low):** The return difference between value stocks (high book-to-market) and growth stocks. Tells us if momentum is really just a value bet in disguise (it isn't — momentum and value are actually negatively correlated).
- **UMD (Up Minus Down):** The momentum factor. Our strategy should have a strong positive loading here, confirming we are capturing the momentum anomaly.

The **alpha** (intercept) is whatever return is left after accounting for all four factor exposures. A positive alpha means our strategy earns more than its factor exposures predict; a negative alpha means it earns less.

### Newey-West Standard Errors

Standard OLS assumes each observation is independent. But monthly portfolio returns are autocorrelated — this month's return partly predicts next month's. Newey-West standard errors correct for this autocorrelation (and heteroskedasticity), giving us honest t-statistics and p-values. Without the correction, the t-statistics would be inflated and we'd overstate the significance of our results. We use 6 lags, which is standard for monthly data.

### Deflated Sharpe Ratio

Proposed by López de Prado (2014). When you test N different strategy variants and report the best one, the maximum Sharpe ratio is biased upward — the more strategies you try, the more likely one of them looks good by chance. The deflated Sharpe corrects for this multiple-testing bias (analogous to the trials factor / look-elsewhere effect in LIGO searches). In Stage 0, we pre-committed to a single strategy specification (no parameter search), so the number of trials = 1 and the deflated Sharpe equals the naive Sharpe. This metric becomes meaningful in Stage 6 when we test parameter variations.

### Rolling Sharpe Ratio

Instead of computing a single Sharpe over the full 15-year sample, we compute it on a rolling 12-month window. This shows *when* the strategy worked and when it didn't. Momentum strategies have characteristic patterns: positive Sharpe during trending markets, deeply negative during market reversals (2009, 2015-2016). A strategy that shows constant Sharpe across all regimes would actually be suspicious — real strategies have regime dependence.

### Calmar Ratio

Annual return divided by maximum drawdown. While Sharpe measures return per unit of volatility, Calmar measures return per unit of worst-case pain. A Sharpe of 0.5 with a -20% max drawdown is much more livable than the same Sharpe with a -80% drawdown. For our strategy, the Calmar will be terrible because of the 2009 crash — this is expected.

---

## 2. Overview

Stage 3 produced a negative in-sample Sharpe ratio (-0.17). The revised pre-registration (deviation note, 2026-05-04) designates the **Ken French UMD correlation** as the definitive validation test, since absolute Sharpe is unreliable over short samples dominated by momentum crashes.

This stage computes everything specified in Stage 0 Section 8:

- **Primary metric:** Net Sharpe ratio (already computed in Stage 3)
- **Benchmark comparison:** Correlation and OLS beta vs Ken French UMD
- **Factor regression:** Fama-French 4-factor regression (alpha, loadings on Mkt-RF, SMB, HML, UMD)
- **Statistical validation:** Newey-West t-statistic, deflated Sharpe ratio
- **Secondary metrics:** Max drawdown, Calmar, hit rate, turnover, rolling Sharpe
- **Rolling analysis:** 12-month rolling Sharpe plot

---

## 3. Implementation

### `src/evaluation/evaluate.py`

```python
"""
evaluate.py — Full evaluation of the momentum strategy against published benchmarks.

Computes all metrics from Stage 0 Section 8 and runs the definitive
pipeline validation test (correlation with Ken French UMD factor).

Usage:
    python -m src.evaluation.evaluate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED, DATA_EXTERNAL, INSAMPLE_START

PLOT_DIR = PROJECT_ROOT / "notebooks"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

INSAMPLE_END = pd.Timestamp('2019-12-31')


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    """Load portfolio returns and Ken French factors."""
    # Portfolio returns
    ret_path = DATA_PROCESSED / "portfolio_returns.parquet"
    if not ret_path.exists():
        raise FileNotFoundError(
            f"Missing {ret_path}. Run Stage 3 first."
        )
    returns = pd.read_parquet(ret_path)
    returns['date'] = pd.to_datetime(returns['date'])

    # Filter to in-sample
    returns = returns[
        (returns['date'] >= pd.Timestamp(INSAMPLE_START)) &
        (returns['date'] <= INSAMPLE_END)
    ].copy()

    # Ken French factors (daily)
    ff_daily = pd.read_parquet(DATA_EXTERNAL / "ff_factors_daily.parquet")
    ff_daily['date'] = pd.to_datetime(ff_daily['date'])

    # Ken French factors (monthly)
    ff_monthly = pd.read_parquet(DATA_EXTERNAL / "ff_factors_monthly.parquet")
    # Ken French monthly data uses PeriodIndex — convert to timestamp
    if hasattr(ff_monthly['date'].dtype, 'freq') or 'period' in str(ff_monthly['date'].dtype).lower():
        ff_monthly['date'] = ff_monthly['date'].dt.to_timestamp()
    else:
        ff_monthly['date'] = pd.to_datetime(ff_monthly['date'])

    print(f"Portfolio returns: {len(returns)} daily observations")
    print(f"  Date range: {returns['date'].min().date()} to "
          f"{returns['date'].max().date()}")

    return returns, ff_daily, ff_monthly


# =============================================================================
# 1. Monthly aggregation and UMD comparison
# =============================================================================

def compute_monthly_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily returns to monthly."""
    returns = returns.copy()
    returns['ym'] = returns['date'].dt.to_period('M')

    monthly = returns.groupby('ym').agg(
        gross_ret=('gross_ret', lambda x: (1 + x).prod() - 1),
        net_ret=('net_ret', lambda x: (1 + x).prod() - 1),
        n_days=('date', 'count'),
    ).reset_index()

    monthly['date'] = monthly['ym'].dt.to_timestamp()
    return monthly


def benchmark_comparison(monthly: pd.DataFrame, ff_monthly: pd.DataFrame):
    """
    Compare monthly returns to Ken French UMD factor.
    This is the DEFINITIVE pipeline validation test.
    """
    print("\n" + "=" * 60)
    print("1. BENCHMARK COMPARISON: Ken French UMD Factor")
    print("=" * 60)

    # Align dates: Ken French monthly dates are month-end
    # Our monthly dates are first-of-month from to_period
    # Match by year-month
    monthly = monthly.copy()
    monthly['ym_str'] = monthly['ym'].astype(str)

    ff = ff_monthly.copy()
    ff['ym'] = ff['date'].dt.to_period('M')
    ff['ym_str'] = ff['ym'].astype(str)

    merged = monthly.merge(ff[['ym_str', 'UMD', 'Mkt-RF', 'SMB', 'HML', 'RF']],
                           on='ym_str', how='inner')

    if len(merged) == 0:
        print("  ✗ ERROR: No overlapping months found. Check date alignment.")
        return None

    print(f"  Overlapping months: {len(merged)}")

    # Correlation
    corr_gross = merged['gross_ret'].corr(merged['UMD'])
    corr_net = merged['net_ret'].corr(merged['UMD'])

    print(f"\n  Correlation with UMD (gross): {corr_gross:.4f}")
    print(f"  Correlation with UMD (net):   {corr_net:.4f}")

    # OLS regression: our returns = alpha + beta * UMD
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        merged['UMD'], merged['net_ret']
    )
    print(f"\n  OLS: net_ret = {intercept:.6f} + {slope:.4f} * UMD")
    print(f"  Beta to UMD:     {slope:.4f}")
    print(f"  Alpha (monthly): {intercept:.6f} ({intercept * 12 * 100:.2f}% annualized)")
    print(f"  R-squared:       {r_value**2:.4f}")
    print(f"  p-value (beta):  {p_value:.6f}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(merged['UMD'], merged['net_ret'], alpha=0.5, s=20)
    # Regression line
    x_range = np.linspace(merged['UMD'].min(), merged['UMD'].max(), 100)
    ax.plot(x_range, intercept + slope * x_range, 'r-', linewidth=2,
            label=f'β={slope:.2f}, R²={r_value**2:.2f}')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Ken French UMD (monthly)')
    ax.set_ylabel('Our strategy (monthly, net)')
    ax.set_title('Pipeline Validation: Strategy vs Published Momentum Factor')
    ax.legend()
    plt.tight_layout()
    path = PLOT_DIR / "stage5_umd_scatter.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Scatter plot saved to {path}")

    # Time series comparison
    fig, ax = plt.subplots(figsize=(14, 5))
    cum_ours = (1 + merged['net_ret']).cumprod()
    cum_umd = (1 + merged['UMD']).cumprod()
    ax.plot(merged['date'], cum_ours.values, label='Our strategy (net)', linewidth=1)
    ax.plot(merged['date'], cum_umd.values, label='Ken French UMD', linewidth=1)
    ax.set_ylabel('Cumulative return')
    ax.set_title('Cumulative Returns: Our Strategy vs Ken French UMD')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = PLOT_DIR / "stage5_umd_cumulative.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Cumulative plot saved to {path}")

    # VALIDATION DECISION
    print(f"\n  {'=' * 40}")
    if corr_net > 0.6:
        print(f"  ✓ PIPELINE VALIDATED")
        print(f"    Correlation {corr_net:.4f} > 0.6 threshold")
        print(f"    Our strategy reproduces the published momentum anomaly.")
    elif corr_net > 0.4:
        print(f"  ⚠ PARTIAL VALIDATION")
        print(f"    Correlation {corr_net:.4f} is between 0.4 and 0.6")
        print(f"    Results are directionally correct but weaker than expected.")
        print(f"    Investigate universe or construction differences.")
    else:
        print(f"  ✗ VALIDATION FAILED")
        print(f"    Correlation {corr_net:.4f} < 0.4")
        print(f"    Strategy does not reproduce the published momentum factor.")
        print(f"    Debug the pipeline before proceeding.")
    print(f"  {'=' * 40}")

    return merged


# =============================================================================
# 2. Fama-French 4-factor regression
# =============================================================================

def factor_regression(merged: pd.DataFrame):
    """
    Regress monthly strategy returns on Fama-French 4 factors.
    This tells us what our strategy is actually exposed to.
    """
    print("\n" + "=" * 60)
    print("2. FAMA-FRENCH 4-FACTOR REGRESSION")
    print("=" * 60)

    # Dependent variable: excess return (net_ret - RF)
    y = merged['net_ret'] - merged['RF']
    X = merged[['Mkt-RF', 'SMB', 'HML', 'UMD']].copy()

    # Add constant for intercept (alpha)
    X_with_const = X.copy()
    X_with_const['const'] = 1.0

    # OLS via numpy (to avoid statsmodels dependency, though it's installed)
    try:
        import statsmodels.api as sm
        model = sm.OLS(y, sm.add_constant(X)).fit(
            cov_type='HAC', cov_kwds={'maxlags': 6}
        )
        print(f"\n  Newey-West adjusted (6 lags):")
        print(f"  {'Factor':<10} {'Coeff':>10} {'t-stat':>10} {'p-value':>10}")
        print(f"  {'-'*42}")
        print(f"  {'Alpha':<10} {model.params['const']:>10.6f} "
              f"{model.tvalues['const']:>10.2f} "
              f"{model.pvalues['const']:>10.4f}")
        for factor in ['Mkt-RF', 'SMB', 'HML', 'UMD']:
            print(f"  {factor:<10} {model.params[factor]:>10.4f} "
                  f"{model.tvalues[factor]:>10.2f} "
                  f"{model.pvalues[factor]:>10.4f}")
        print(f"\n  R-squared:     {model.rsquared:.4f}")
        print(f"  Adj R-squared: {model.rsquared_adj:.4f}")

        # Annualized alpha
        alpha_ann = model.params['const'] * 12
        print(f"  Alpha (ann.):  {alpha_ann:.4f} ({alpha_ann*100:.2f}%)")

        # Expected UMD loading interpretation
        umd_loading = model.params['UMD']
        print(f"\n  UMD loading: {umd_loading:.4f}")
        if umd_loading > 0.5:
            print(f"  ✓ Strong positive UMD loading — confirms this IS a momentum strategy")
        elif umd_loading > 0.2:
            print(f"  ⚠ Moderate UMD loading — momentum signal present but diluted")
        else:
            print(f"  ✗ Weak UMD loading — strategy may not be capturing momentum")

    except ImportError:
        # Fallback: simple OLS without Newey-West
        from numpy.linalg import lstsq
        X_mat = X_with_const.values
        betas, _, _, _ = lstsq(X_mat, y.values, rcond=None)
        factor_names = ['Mkt-RF', 'SMB', 'HML', 'UMD', 'Alpha']
        print(f"\n  Simple OLS (no Newey-West):")
        for name, beta in zip(factor_names, betas):
            print(f"    {name}: {beta:.4f}")


# =============================================================================
# 3. Secondary metrics (Stage 0 Section 8)
# =============================================================================

def secondary_metrics(returns: pd.DataFrame):
    """Compute all secondary metrics from the pre-registration."""
    print("\n" + "=" * 60)
    print("3. SECONDARY METRICS (Stage 0, Section 8)")
    print("=" * 60)

    r = returns['net_ret']

    # Basic stats
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    cum = (1 + r).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Hit rate
    hit_rate_daily = (r > 0).mean()

    # Monthly hit rate
    returns_copy = returns.copy()
    returns_copy['ym'] = returns_copy['date'].dt.to_period('M')
    monthly_ret = returns_copy.groupby('ym')['net_ret'].apply(
        lambda x: (1 + x).prod() - 1
    )
    hit_rate_monthly = (monthly_ret > 0).mean()

    # Turnover (from costs)
    total_cost = returns['cost'].sum()
    n_years = len(returns) / 252
    ann_cost = total_cost / n_years

    print(f"\n  Annualized return (net): {ann_ret:.4f} ({ann_ret*100:.1f}%)")
    print(f"  Annualized volatility:  {ann_vol:.4f} ({ann_vol*100:.1f}%)")
    print(f"  Sharpe ratio (net):     {sharpe:.2f}")
    print(f"  Max drawdown:           {max_dd:.4f} ({max_dd*100:.1f}%)")
    print(f"  Calmar ratio:           {calmar:.2f}")
    print(f"  Hit rate (daily):       {hit_rate_daily:.3f} ({hit_rate_daily*100:.1f}%)")
    print(f"  Hit rate (monthly):     {hit_rate_monthly:.3f} ({hit_rate_monthly*100:.1f}%)")
    print(f"  Annualized cost drag:   {ann_cost:.4f} ({ann_cost*100:.2f}%)")

    return {
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
    }


# =============================================================================
# 4. Rolling Sharpe ratio
# =============================================================================

def rolling_sharpe(returns: pd.DataFrame):
    """Plot 12-month rolling Sharpe ratio."""
    print("\n" + "=" * 60)
    print("4. ROLLING 12-MONTH SHARPE RATIO")
    print("=" * 60)

    r = returns.set_index('date')['net_ret']

    # 252-day rolling window
    rolling_mean = r.rolling(252).mean() * 252
    rolling_vol = r.rolling(252).std() * np.sqrt(252)
    rolling_sr = rolling_mean / rolling_vol

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rolling_sr.index, rolling_sr.values, linewidth=1)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.3, label='Published long-run Sharpe ~0.5')
    ax.set_ylabel('Rolling 12-month Sharpe ratio')
    ax.set_title('Rolling Sharpe Ratio (Net of Costs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = PLOT_DIR / "stage5_rolling_sharpe.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved to {path}")

    # Report periods of positive and negative Sharpe
    positive_pct = (rolling_sr.dropna() > 0).mean()
    print(f"  Fraction of time with positive rolling Sharpe: "
          f"{positive_pct:.1%}")
    print(f"  Rolling Sharpe range: [{rolling_sr.min():.2f}, {rolling_sr.max():.2f}]")


# =============================================================================
# 5. Market correlation
# =============================================================================

def market_correlation(returns: pd.DataFrame, ff_daily: pd.DataFrame):
    """Compute correlation to market return (should be near zero for L/S)."""
    print("\n" + "=" * 60)
    print("5. MARKET CORRELATION")
    print("=" * 60)

    merged = returns.merge(ff_daily[['date', 'Mkt-RF']], on='date', how='inner')
    corr = merged['net_ret'].corr(merged['Mkt-RF'])
    print(f"  Daily correlation with market: {corr:.4f}")

    if abs(corr) < 0.3:
        print(f"  ✓ Low market correlation — consistent with dollar-neutral construction")
    else:
        print(f"  ⚠ Higher than expected market correlation for a dollar-neutral strategy")


# =============================================================================
# 6. Deflated Sharpe ratio
# =============================================================================

def deflated_sharpe(returns: pd.DataFrame):
    """
    Compute the deflated Sharpe ratio (López de Prado 2014).
    Since no parameter search was performed (Stage 0 specification),
    the number of trials = 1 and the deflated Sharpe ≈ the naive Sharpe.
    """
    print("\n" + "=" * 60)
    print("6. DEFLATED SHARPE RATIO")
    print("=" * 60)

    r = returns['net_ret']
    T = len(r)
    sr = r.mean() / r.std()  # daily Sharpe (not annualized)
    sr_ann = sr * np.sqrt(252)
    skew = r.skew()
    kurt = r.kurtosis()  # excess kurtosis

    # Standard error of the Sharpe ratio
    # SE(SR) = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt/4)*SR^2) / T)
    se_sr = np.sqrt(
        (1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / T
    )

    # For 1 trial, deflated Sharpe ≈ naive Sharpe
    # (the correction is meaningful only when N_trials > 1)
    n_trials = 1
    print(f"  Number of trials (pre-registered):  {n_trials}")
    print(f"  Daily Sharpe ratio:                 {sr:.4f}")
    print(f"  Annualized Sharpe ratio:            {sr_ann:.2f}")
    print(f"  Standard error of SR:               {se_sr:.4f}")
    print(f"  Skewness:                           {skew:.4f}")
    print(f"  Excess kurtosis:                    {kurt:.4f}")
    print(f"\n  With 1 trial, the deflated Sharpe ≈ naive Sharpe.")
    print(f"  The deflated Sharpe becomes meaningful in Stage 6")
    print(f"  when parameter variations are tested.")

    # t-statistic for SR != 0
    t_stat = sr / se_sr
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    print(f"\n  t-stat (SR != 0): {t_stat:.2f}")
    print(f"  p-value:          {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Sharpe is statistically significantly different from zero")
    else:
        print(f"  Sharpe is NOT statistically significantly different from zero")


# =============================================================================
# Main
# =============================================================================

def run_evaluation():
    """Run the full evaluation suite."""
    print("=" * 60)
    print("STAGE 5: EVALUATION")
    print("=" * 60)

    returns, ff_daily, ff_monthly = load_data()

    # Monthly aggregation
    monthly = compute_monthly_returns(returns)
    print(f"Monthly returns: {len(monthly)} months")

    # 1. Benchmark comparison (DEFINITIVE TEST)
    merged = benchmark_comparison(monthly, ff_monthly)

    # 2. Factor regression
    if merged is not None:
        factor_regression(merged)

    # 3. Secondary metrics
    secondary_metrics(returns)

    # 4. Rolling Sharpe
    rolling_sharpe(returns)

    # 5. Market correlation
    market_correlation(returns, ff_daily)

    # 6. Deflated Sharpe
    deflated_sharpe(returns)

    # Summary
    print("\n" + "=" * 60)
    print("STAGE 5 COMPLETE")
    print("=" * 60)
    print("Review all plots in notebooks/ and the metrics above.")
    print("The UMD correlation is the definitive pipeline validation.")
    print("Do NOT examine holdout results — that's Stage 7.")


if __name__ == "__main__":
    run_evaluation()
```

---

## 4. Deliverables Checklist

Before moving to Stage 6, confirm every item:

- [ ] `src/evaluation/evaluate.py` — evaluation module
- [ ] **UMD correlation > 0.6** — definitive pipeline validation
- [ ] UMD scatter plot and cumulative return comparison
- [ ] Fama-French 4-factor regression with Newey-West standard errors
- [ ] All secondary metrics from Stage 0 Section 8 reported
- [ ] Rolling 12-month Sharpe plot
- [ ] Market correlation check (should be near zero)
- [ ] Deflated Sharpe ratio computed
- [ ] All code committed to git
- [ ] Holdout results NOT examined

> **Only proceed to Stage 6 when all boxes are checked.**
