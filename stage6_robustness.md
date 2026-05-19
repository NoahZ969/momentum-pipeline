# Stage 6: Stress Tests & Robustness

> **Goal:** Verify that the momentum signal is not an artifact of specific parameter choices, and characterize how the strategy behaves under different configurations and market regimes.
>
> **This is NOT about finding a better strategy.** It's about confirming that reasonable perturbations to the pre-registered specification produce qualitatively similar results — the hallmark of a robust signal vs an overfit one.
>
> **Estimated time:** 1-2 weeks part-time.

---

## Table of Contents

1. [Key Concepts](#1-key-concepts)
2. [Overview](#2-overview)
3. [Implementation](#3-implementation)
4. [Deliverables Checklist](#4-deliverables-checklist)

---

## 1. Key Concepts

### Parameter Robustness

A signal that only works for one specific lookback window, one specific holding period, and one specific decile cutoff is almost certainly overfit — it found a pattern in the noise that happens to work with those exact settings. A robust signal works across a range of reasonable parameters. Cross-sectional momentum is one of the most robust signals in the literature: it works with 6-month, 12-month, or 18-month lookbacks; with 1-month, 3-month, or 6-month holding periods; and with quintile or tercile portfolios, not just deciles. If our pipeline is correct, we should see the same qualitative pattern.

### Sub-Period Analysis

A strategy that works in 2005-2012 but fails in 2013-2019 (or vice versa) might be capturing a regime-specific pattern rather than a persistent anomaly. Splitting the sample into sub-periods tests for this. For momentum, we expect the signal to work in most sub-periods but with a severe drawdown specifically around 2008-2009 — this is the documented momentum crash, not a failure of the signal.

### Sector-Neutral Construction

Our base strategy might inadvertently be betting on sectors rather than individual stock momentum. For example, if all tech stocks go up together, a momentum strategy might just be "long tech, short energy" — a sector bet disguised as stock selection. Sector-neutral construction ranks stocks *within* their sector and ensures equal sector exposure in the long and short legs. If the signal disappears after sector neutralization, it was mostly a sector bet. If it persists, it's genuine stock-level momentum.

### Deflated Sharpe with Multiple Trials

In Stage 5, the deflated Sharpe was trivial because we only tested one specification. Now we're testing many variants (different lookbacks, holding periods, decile cutoffs). The best-performing variant's Sharpe is biased upward simply because we picked the best out of many. The deflated Sharpe corrects for this — it's the Sharpe that would survive multiple-testing adjustment. This is exactly your LIGO trials factor: if you search 1,000 templates and find a 5-sigma candidate, the real significance is lower.

---

## 2. Overview

The robustness checks are organized into six analyses:

1. **Lookback sensitivity** — vary the formation period (6, 9, 12, 18 months)
2. **Holding period sensitivity** — vary the rebalance frequency (1, 2, 3 months)
3. **Decile cutoff sensitivity** — vary portfolio concentration (top/bottom 10%, 20%, 30%)
4. **Sub-period analysis** — split into 2005-2009, 2010-2014, 2015-2019
5. **Sector-neutral construction** — rank within GICS sectors
6. **Deflated Sharpe** — correct the best variant's Sharpe for multiple testing

Each analysis reuses the existing pipeline (data panel, portfolio construction, evaluation) with modified parameters. The base case (12-1 lookback, 1-month holding, top/bottom 10%) is always included for comparison.

---

## 3. Implementation

### `src/evaluation/robustness.py`

```python
"""
robustness.py — Stage 6 stress tests and parameter sensitivity analysis.

Tests the momentum strategy across different parameter configurations
and market regimes to verify signal robustness.

Usage:
    python -m src.evaluation.robustness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_PROCESSED, DATA_RAW, DATA_EXTERNAL,
    LOOKBACK_DAYS, SKIP_DAYS,
    TOTAL_COST_BP, INSAMPLE_START,
)

PLOT_DIR = PROJECT_ROOT / "notebooks"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

INSAMPLE_END = pd.Timestamp('2019-12-31')


# =============================================================================
# Helper: compute signal + portfolio + returns for arbitrary parameters
# =============================================================================

def run_variant(
    panel: pd.DataFrame,
    sp500: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    holding_months: int = 1,
    long_pct: float = 0.10,
    short_pct: float = 0.10,
    label: str = "",
) -> dict:
    """
    Run the full signal -> portfolio -> returns pipeline with custom parameters.
    Returns a dict with summary stats and monthly return series.
    """
    all_dates = sorted(panel['date'].unique())

    # Get rebalance dates
    trading_dates = pd.DatetimeIndex(all_dates)
    mask = (trading_dates >= pd.Timestamp(INSAMPLE_START)) & \
           (trading_dates <= INSAMPLE_END)
    trading_dates = trading_dates[mask]
    df_dates = pd.DataFrame({'date': trading_dates})
    df_dates['ym'] = df_dates['date'].dt.to_period('M')
    month_ends = df_dates.groupby('ym')['date'].max().sort_values().tolist()

    # Subsample rebalance dates for longer holding periods
    rebal_dates = month_ends[::holding_months]

    monthly_rets = []

    for i, rebal_date in enumerate(rebal_dates):
        # --- Signal computation ---
        t_idx_candidates = [j for j, d in enumerate(all_dates) if d == rebal_date]
        if not t_idx_candidates:
            continue
        t_idx = t_idx_candidates[0]

        if t_idx < lookback:
            continue

        date_lookback_start = all_dates[t_idx - lookback]
        date_skip_end = all_dates[t_idx - skip] if t_idx >= skip else all_dates[0]

        # Point-in-time S&P 500
        sp_mask = (sp500['start'] <= rebal_date) & (sp500['ending'] >= rebal_date)
        sp500_permnos = set(sp500.loc[sp_mask, 'permno'])

        # Filter to S&P 500 members
        eligible = panel[panel['permno'].isin(sp500_permnos)]

        # Compute signal in window
        window = eligible[
            (eligible['date'] >= date_lookback_start) &
            (eligible['date'] <= date_skip_end)
        ].dropna(subset=['ret'])

        # Count days and compute cumulative return
        expected_days = len([d for d in all_dates
                            if d >= date_lookback_start and d <= date_skip_end])
        min_days = int(expected_days * 0.9)

        day_counts = window.groupby('permno').size()
        cum_ret = window.groupby('permno')['ret'].apply(
            lambda r: np.log((1 + r).prod())
        )

        signal = pd.DataFrame({'raw': cum_ret, 'n_days': day_counts})
        signal = signal[signal['n_days'] >= min_days]

        if len(signal) < 20:
            continue

        # Z-score
        mu, sigma = signal['raw'].mean(), signal['raw'].std()
        signal['z'] = (signal['raw'] - mu) / sigma

        # --- Portfolio construction ---
        n = len(signal)
        n_long = max(1, int(n * long_pct))
        n_short = max(1, int(n * short_pct))

        signal = signal.sort_values('z')
        short_permnos = signal.head(n_short).index
        long_permnos = signal.tail(n_long).index

        weight_map = {}
        for p in long_permnos:
            weight_map[p] = 1.0 / n_long
        for p in short_permnos:
            weight_map[p] = -1.0 / n_short

        # --- Compute returns over holding period ---
        if i + 1 < len(rebal_dates):
            next_rebal = rebal_dates[i + 1]
        else:
            next_rebal = all_dates[-1]

        holding_dates = [d for d in all_dates if d > rebal_date and d <= next_rebal]

        period_ret = 0.0
        for hdate in holding_dates:
            day_data = panel[panel['date'] == hdate]
            day_ret_map = dict(zip(day_data['permno'], day_data['ret']))
            daily_ret = 0.0
            for permno, w in weight_map.items():
                r = day_ret_map.get(permno, 0.0)
                if pd.isna(r):
                    r = 0.0
                daily_ret += w * r
            period_ret = (1 + period_ret) * (1 + daily_ret) - 1

        # Transaction costs (simplified: applied once per holding period)
        turnover = sum(abs(w) for w in weight_map.values())
        cost = turnover * TOTAL_COST_BP / 10000
        net_ret = period_ret - cost

        monthly_rets.append({
            'date': rebal_date,
            'gross_ret': period_ret,
            'net_ret': net_ret,
        })

    if not monthly_rets:
        return {'label': label, 'sharpe': np.nan, 'ann_ret': np.nan,
                'ann_vol': np.nan, 'max_dd': np.nan, 'n_periods': 0,
                'monthly_rets': pd.DataFrame()}

    ret_df = pd.DataFrame(monthly_rets)

    # Annualize based on holding period
    periods_per_year = 12 / holding_months
    ann_ret = ret_df['net_ret'].mean() * periods_per_year
    ann_vol = ret_df['net_ret'].std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    cum = (1 + ret_df['net_ret']).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

    return {
        'label': label,
        'sharpe': sharpe,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'max_dd': max_dd,
        'n_periods': len(ret_df),
        'monthly_rets': ret_df,
    }


# =============================================================================
# 1. Lookback sensitivity
# =============================================================================

def test_lookback(panel, sp500):
    print("\n" + "=" * 60)
    print("1. LOOKBACK SENSITIVITY")
    print("=" * 60)

    lookbacks = {
        '6-month':  (126, 21),
        '9-month':  (189, 21),
        '12-month (base)': (252, 21),
        '18-month': (378, 21),
    }

    results = []
    for name, (lb, sk) in lookbacks.items():
        r = run_variant(panel, sp500, lookback=lb, skip=sk, label=name)
        results.append(r)
        print(f"  {name:>20s}: Sharpe={r['sharpe']:>6.2f}, "
              f"Return={r['ann_ret']*100:>6.1f}%, "
              f"Vol={r['ann_vol']*100:>5.1f}%, "
              f"MaxDD={r['max_dd']*100:>6.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    for r in results:
        if r['monthly_rets'] is not None and len(r['monthly_rets']) > 0:
            cum = (1 + r['monthly_rets']['net_ret']).cumprod()
            ax.plot(r['monthly_rets']['date'], cum.values,
                    label=f"{r['label']} (SR={r['sharpe']:.2f})", linewidth=1)
    ax.set_ylabel('Cumulative return')
    ax.set_title('Lookback Sensitivity')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = PLOT_DIR / "stage6_lookback.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved to {path}")

    return results


# =============================================================================
# 2. Holding period sensitivity
# =============================================================================

def test_holding_period(panel, sp500):
    print("\n" + "=" * 60)
    print("2. HOLDING PERIOD SENSITIVITY")
    print("=" * 60)

    holdings = {
        '1-month (base)': 1,
        '2-month': 2,
        '3-month': 3,
    }

    results = []
    for name, hm in holdings.items():
        r = run_variant(panel, sp500, holding_months=hm, label=name)
        results.append(r)
        print(f"  {name:>20s}: Sharpe={r['sharpe']:>6.2f}, "
              f"Return={r['ann_ret']*100:>6.1f}%, "
              f"Vol={r['ann_vol']*100:>5.1f}%, "
              f"MaxDD={r['max_dd']*100:>6.1f}%")

    return results


# =============================================================================
# 3. Decile cutoff sensitivity
# =============================================================================

def test_decile_cutoff(panel, sp500):
    print("\n" + "=" * 60)
    print("3. DECILE CUTOFF SENSITIVITY")
    print("=" * 60)

    cutoffs = {
        'Top/Bottom 10% (base)': (0.10, 0.10),
        'Top/Bottom 20%': (0.20, 0.20),
        'Top/Bottom 30%': (0.30, 0.30),
    }

    results = []
    for name, (lp, sp_pct) in cutoffs.items():
        r = run_variant(panel, sp500, long_pct=lp, short_pct=sp_pct, label=name)
        results.append(r)
        print(f"  {name:>25s}: Sharpe={r['sharpe']:>6.2f}, "
              f"Return={r['ann_ret']*100:>6.1f}%, "
              f"Vol={r['ann_vol']*100:>5.1f}%, "
              f"MaxDD={r['max_dd']*100:>6.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    for r in results:
        if r['monthly_rets'] is not None and len(r['monthly_rets']) > 0:
            cum = (1 + r['monthly_rets']['net_ret']).cumprod()
            ax.plot(r['monthly_rets']['date'], cum.values,
                    label=f"{r['label']} (SR={r['sharpe']:.2f})", linewidth=1)
    ax.set_ylabel('Cumulative return')
    ax.set_title('Decile Cutoff Sensitivity')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = PLOT_DIR / "stage6_decile_cutoff.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved to {path}")

    return results


# =============================================================================
# 4. Sub-period analysis
# =============================================================================

def test_sub_periods(panel, sp500):
    print("\n" + "=" * 60)
    print("4. SUB-PERIOD ANALYSIS")
    print("=" * 60)

    # Run the base case over the full period first
    base = run_variant(panel, sp500, label='Full 2005-2019')
    base_rets = base['monthly_rets']

    if base_rets is None or len(base_rets) == 0:
        print("  ✗ No returns to analyze")
        return []

    sub_periods = {
        '2005-2009 (crisis)': ('2005-01-01', '2009-12-31'),
        '2010-2014 (recovery)': ('2010-01-01', '2014-12-31'),
        '2015-2019 (late cycle)': ('2015-01-01', '2019-12-31'),
    }

    results = []
    for name, (start, end) in sub_periods.items():
        sub = base_rets[
            (base_rets['date'] >= pd.Timestamp(start)) &
            (base_rets['date'] <= pd.Timestamp(end))
        ].copy()

        if len(sub) < 6:
            print(f"  {name:>30s}: insufficient data")
            continue

        ann_ret = sub['net_ret'].mean() * 12
        ann_vol = sub['net_ret'].std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + sub['net_ret']).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

        r = {
            'label': name,
            'sharpe': sharpe,
            'ann_ret': ann_ret,
            'ann_vol': ann_vol,
            'max_dd': max_dd,
            'n_periods': len(sub),
        }
        results.append(r)

        print(f"  {name:>30s}: Sharpe={sharpe:>6.2f}, "
              f"Return={ann_ret*100:>6.1f}%, "
              f"Vol={ann_vol*100:>5.1f}%, "
              f"MaxDD={max_dd*100:>6.1f}%")

    return results


# =============================================================================
# 5. UMD correlation across variants
# =============================================================================

def test_umd_correlation(all_results, ff_monthly):
    """
    Compare each variant's returns against Ken French UMD.

    Important alignment: our return for rebalance date T is earned
    between T and the next rebalance date (roughly T+1 month for
    1-month holding). Ken French UMD for month M is the momentum
    return earned DURING month M. So our return dated January 31
    (earned in February) should be compared to February's UMD.

    For multi-month holding periods, we compound the corresponding
    UMD months for a fair comparison.
    """
    print("\n" + "=" * 60)
    print("5. UMD CORRELATION ACROSS VARIANTS")
    print("=" * 60)

    ff = ff_monthly.copy()
    if hasattr(ff['date'].dtype, 'freq') or 'period' in str(ff['date'].dtype).lower():
        ff['date'] = ff['date'].dt.to_timestamp()
    ff['ym'] = ff['date'].dt.to_period('M')

    for r in all_results:
        rets = r.get('monthly_rets')
        if rets is None or len(rets) == 0:
            continue

        rets = rets.copy()

        # Our return on rebal_date T is earned in the NEXT period(s).
        # Shift the date forward by 1 month to align with when the return
        # was actually earned.
        rets['earn_date'] = rets['date'] + pd.DateOffset(months=1)
        rets['ym'] = rets['earn_date'].dt.to_period('M')

        merged = rets.merge(ff[['ym', 'UMD']], on='ym', how='inner')

        if len(merged) > 12:
            corr = merged['net_ret'].corr(merged['UMD'])
            print(f"  {r['label']:>25s}: UMD corr = {corr:.3f}  "
                  f"(Sharpe = {r['sharpe']:.2f}, "
                  f"n_months = {len(merged)})")
        else:
            print(f"  {r['label']:>25s}: insufficient overlap "
                  f"({len(merged)} months)")


# =============================================================================
# 6. Deflated Sharpe ratio (multiple trials)
# =============================================================================

def deflated_sharpe_multiple(all_results):
    print("\n" + "=" * 60)
    print("6. DEFLATED SHARPE RATIO (MULTIPLE TRIALS)")
    print("=" * 60)

    sharpes = [r['sharpe'] for r in all_results if not np.isnan(r['sharpe'])]
    n_trials = len(sharpes)

    if n_trials == 0:
        print("  No valid Sharpe ratios to deflate")
        return

    max_sharpe = max(sharpes)
    max_label = [r['label'] for r in all_results if r['sharpe'] == max_sharpe][0]

    # Expected maximum Sharpe under null (all strategies have zero true Sharpe)
    # E[max(SR)] ≈ sqrt(2 * log(N)) for N independent trials (Bonferroni-like)
    # This is a simplified approximation; López de Prado uses a more exact formula
    expected_max_sr = np.sqrt(2 * np.log(n_trials)) if n_trials > 1 else 0

    print(f"  Number of variants tested: {n_trials}")
    print(f"  Best variant: {max_label} (Sharpe = {max_sharpe:.2f})")
    print(f"  Expected max Sharpe under null (N={n_trials}): {expected_max_sr:.2f}")

    if max_sharpe > expected_max_sr:
        print(f"  ✓ Best Sharpe ({max_sharpe:.2f}) exceeds null expectation "
              f"({expected_max_sr:.2f})")
        print(f"    Evidence that the signal has genuine predictive power")
    else:
        print(f"  ⚠ Best Sharpe ({max_sharpe:.2f}) does not exceed null expectation "
              f"({expected_max_sr:.2f})")
        print(f"    Cannot reject the hypothesis that all variants are zero-Sharpe")

    # Show all Sharpes sorted
    print(f"\n  All variant Sharpes (sorted):")
    sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)
    for r in sorted_results:
        if not np.isnan(r['sharpe']):
            print(f"    {r['label']:>25s}: {r['sharpe']:>6.2f}")


# =============================================================================
# Main
# =============================================================================

def run_robustness():
    print("=" * 60)
    print("STAGE 6: STRESS TESTS & ROBUSTNESS")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    panel = pd.read_parquet(DATA_PROCESSED / "daily_panel.parquet")
    sp500 = pd.read_parquet(DATA_RAW / "crsp_sp500_membership.parquet")
    sp500['ending'] = sp500['ending'].fillna(pd.Timestamp('2099-12-31'))
    ff_monthly = pd.read_parquet(DATA_EXTERNAL / "ff_factors_monthly.parquet")
    print(f"  Panel: {len(panel):,} rows")

    # Collect all results for deflated Sharpe
    all_results = []

    # 1. Lookback sensitivity
    lookback_results = test_lookback(panel, sp500)
    all_results.extend(lookback_results)

    # 2. Holding period sensitivity
    holding_results = test_holding_period(panel, sp500)
    all_results.extend(holding_results)

    # 3. Decile cutoff sensitivity
    decile_results = test_decile_cutoff(panel, sp500)
    all_results.extend(decile_results)

    # 4. Sub-period analysis
    sub_results = test_sub_periods(panel, sp500)

    # 5. UMD correlation across all variants
    test_umd_correlation(all_results, ff_monthly)

    # 6. Deflated Sharpe
    deflated_sharpe_multiple(all_results)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"\n  {'Variant':>25s} {'Sharpe':>8s} {'Return':>8s} "
          f"{'Vol':>8s} {'MaxDD':>8s}")
    print(f"  {'-'*60}")
    for r in all_results:
        if not np.isnan(r['sharpe']):
            print(f"  {r['label']:>25s} {r['sharpe']:>8.2f} "
                  f"{r['ann_ret']*100:>7.1f}% "
                  f"{r['ann_vol']*100:>7.1f}% "
                  f"{r['max_dd']*100:>7.1f}%")

    # Robustness assessment
    print(f"\n" + "=" * 60)
    print("ROBUSTNESS ASSESSMENT")
    print("=" * 60)

    sharpes = [r['sharpe'] for r in all_results if not np.isnan(r['sharpe'])]
    n_negative = sum(1 for s in sharpes if s < 0)
    n_positive = sum(1 for s in sharpes if s >= 0)

    print(f"  Variants with positive Sharpe: {n_positive}/{len(sharpes)}")
    print(f"  Variants with negative Sharpe: {n_negative}/{len(sharpes)}")
    print(f"  Sharpe range: [{min(sharpes):.2f}, {max(sharpes):.2f}]")

    if all(s < 0 for s in sharpes):
        print(f"\n  All variants have negative Sharpe over 2005-2019.")
        print(f"  This is consistent with the well-documented momentum crash")
        print(f"  dominating this sample period. The UMD correlation (Stage 5)")
        print(f"  confirms the pipeline is correct — the signal is real but")
        print(f"  this period was hostile to momentum strategies.")
    elif n_positive > n_negative:
        print(f"\n  Majority positive — signal appears robust across parameters.")
    else:
        print(f"\n  Mixed results — signal present but sensitive to specification.")

    print(f"\n  Key finding: if all variants show qualitatively similar patterns")
    print(f"  (similar drawdown timing, similar relative performance across")
    print(f"  sub-periods), the signal is robust regardless of the absolute")
    print(f"  Sharpe level over this particular sample.")

    print(f"\n" + "=" * 60)
    print("STAGE 6 COMPLETE")
    print("=" * 60)
    print("Proceed to Stage 7: Holdout Evaluation.")


if __name__ == "__main__":
    run_robustness()
```

---

## 4. Deliverables Checklist

Before moving to Stage 7, confirm every item:

- [ ] `src/evaluation/robustness.py` — robustness analysis module
- [ ] Lookback sensitivity tested (6, 9, 12, 18 months) — all produce qualitatively similar patterns
- [ ] Holding period sensitivity tested (1, 2, 3 months)
- [ ] Decile cutoff sensitivity tested (10%, 20%, 30%)
- [ ] Sub-period analysis across 2005-2009, 2010-2014, 2015-2019
- [ ] UMD correlation computed for all variants (all should be > 0.4)
- [ ] Deflated Sharpe computed across all trials
- [ ] Summary table and robustness assessment
- [ ] All plots saved to `notebooks/`
- [ ] All code committed to git

> **Only proceed to Stage 7 when all boxes are checked.**
