# Stage 3: Portfolio Construction

> **Goal:** Convert the MOM_12_1 signal into portfolio weights — deciding which stocks to go long, which to short, and how much of each — exactly as specified in the Stage 0 pre-registration.
>
> **Estimated time:** 3-5 days part-time.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Construction Rule Recap](#2-construction-rule-recap)
3. [Implementation](#3-implementation)
4. [Deliverables Checklist](#4-deliverables-checklist)

---

## 1. Overview

Stage 2 produced a signal — a z-score for each stock on each rebalance date telling us how strong its momentum is. Stage 3 turns that signal into a portfolio: a set of weights saying "put X% of capital in stock A, short Y% of stock B."

The construction rule is deliberately simple (equal-weighted top/bottom decile, dollar-neutral). This is the standard textbook construction used in published momentum research, which lets us compare our results to published benchmarks. Fancier construction (sector-neutral, volatility-weighted, optimized) comes in Stage 6 robustness checks, not here.

---

## 2. Construction Rule Recap

From Stage 0 pre-registration, Section 6:

- **Rebalance frequency:** Monthly, last trading day of each calendar month.
- **Long leg:** Top decile (top 10%) of stocks ranked by `z_signal`, equal-weighted.
- **Short leg:** Bottom decile (bottom 10%), equal-weighted.
- **Dollar-neutral:** Gross long exposure = gross short exposure = 100% of NAV. Net exposure = 0.
- **Holding period:** From rebalance date T to next rebalance date T'. No intra-month rebalancing.
- **Delisting handling:** If a stock delists between T and T', its delisting return is applied (already handled in the daily panel from Stage 1). The proceeds are held in cash until the next rebalance.

---

## 3. Implementation

### `src/portfolio/construction.py`

```python
"""
construction.py — Build long-short momentum portfolios from signals.

Takes the signal panel from Stage 2 and produces portfolio weights
for each rebalance date, then computes daily portfolio returns by
carrying the weights forward between rebalances.

Usage:
    python -m src.portfolio.construction
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_PROCESSED, DECILE_LONG, DECILE_SHORT,
    TOTAL_COST_BP, INSAMPLE_START, HOLDOUT_END,
)


def compute_weights(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Compute portfolio weights on each rebalance date.

    For each rebalance date:
      1. Rank stocks by z_signal
      2. Long leg = top decile, equal-weighted, summing to +1
      3. Short leg = bottom decile, equal-weighted, summing to -1
      4. All other stocks get weight 0

    Returns DataFrame with columns: date, permno, weight, leg
    """
    print("Computing portfolio weights...")

    all_weights = []
    rebal_dates = sorted(signals['date'].unique())

    for rebal_date in rebal_dates:
        day_signals = signals[signals['date'] == rebal_date].copy()
        n = len(day_signals)
        if n < 20:
            # Too few stocks for meaningful deciles
            continue

        decile_size = n // 10

        # Sort by z_signal
        day_signals = day_signals.sort_values('z_signal')

        # Bottom decile (short leg)
        short_permnos = day_signals.head(decile_size)['permno'].values
        # Top decile (long leg)
        long_permnos = day_signals.tail(decile_size)['permno'].values

        # Equal weights within each leg
        long_weight = 1.0 / len(long_permnos)    # sums to +1
        short_weight = -1.0 / len(short_permnos)  # sums to -1

        for p in long_permnos:
            all_weights.append({
                'date': rebal_date,
                'permno': p,
                'weight': long_weight,
                'leg': 'long',
            })
        for p in short_permnos:
            all_weights.append({
                'date': rebal_date,
                'permno': p,
                'weight': short_weight,
                'leg': 'short',
            })

    weights = pd.DataFrame(all_weights)
    print(f"  {len(rebal_dates)} rebalance dates")
    print(f"  Stocks per leg (typical): {weights.groupby(['date', 'leg']).size().mean():.0f}")

    return weights


def compute_portfolio_returns(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily portfolio returns from monthly rebalance weights.

    Between rebalances, weights drift with daily returns (buy-and-hold
    within each month). The portfolio return on day t is:

        r_portfolio(t) = sum_i [ w_i(t-1) * r_i(t) ]

    where w_i(t-1) is the weight of stock i at close of day t-1,
    which drifts from the target weights as stocks move.

    For simplicity in Stage 3, we use fixed weights within each month
    (rebalance at the start, ignore drift). This slightly overstates
    turnover costs but simplifies the implementation. Stage 4 will
    implement the full drift-aware backtester.

    Transaction costs are applied at each rebalance based on the
    absolute change in weights.

    Returns DataFrame with columns: date, gross_ret, cost, net_ret
    """
    print("Computing daily portfolio returns...")

    rebal_dates = sorted(weights['date'].unique())
    panel = panel.sort_values('date')
    all_trading_dates = sorted(panel['date'].unique())

    daily_returns = []

    for i, rebal_date in enumerate(rebal_dates):
        # Get weights for this rebalance
        month_weights = weights[weights['date'] == rebal_date].copy()
        weight_map = dict(zip(month_weights['permno'], month_weights['weight']))

        # Determine holding period: from rebal_date to next rebal_date
        if i + 1 < len(rebal_dates):
            next_rebal = rebal_dates[i + 1]
        else:
            # Last rebalance: hold to end of data
            next_rebal = all_trading_dates[-1]

        # Get all trading dates in the holding period (exclusive of rebal_date,
        # inclusive of next_rebal — we earn returns the day AFTER rebalancing)
        holding_dates = [d for d in all_trading_dates
                         if d > rebal_date and d <= next_rebal]

        # Compute transaction costs at this rebalance
        if i == 0:
            # First rebalance: cost of establishing the full position
            turnover = sum(abs(w) for w in weight_map.values())
        else:
            # Subsequent rebalances: cost of changing weights
            prev_weights = weights[weights['date'] == rebal_dates[i - 1]]
            prev_map = dict(zip(prev_weights['permno'], prev_weights['weight']))

            # All permnos that appear in either old or new weights
            all_permnos = set(prev_map.keys()) | set(weight_map.keys())
            turnover = sum(
                abs(weight_map.get(p, 0) - prev_map.get(p, 0))
                for p in all_permnos
            )

        rebal_cost = turnover * TOTAL_COST_BP / 10000

        for j, hdate in enumerate(holding_dates):
            # Get returns for all stocks in the portfolio on this date
            day_data = panel[panel['date'] == hdate]
            day_ret_map = dict(zip(day_data['permno'], day_data['ret']))

            # Portfolio gross return: weighted sum of stock returns
            gross_ret = 0.0
            for permno, w in weight_map.items():
                stock_ret = day_ret_map.get(permno, 0.0)
                if pd.isna(stock_ret):
                    stock_ret = 0.0
                gross_ret += w * stock_ret

            # Apply transaction cost only on the first day of the holding period
            if j == 0:
                cost = rebal_cost
            else:
                cost = 0.0

            net_ret = gross_ret - cost

            daily_returns.append({
                'date': hdate,
                'gross_ret': gross_ret,
                'cost': cost,
                'net_ret': net_ret,
            })

    result = pd.DataFrame(daily_returns)
    return result


def compute_summary_stats(returns: pd.DataFrame, label: str = ""):
    """Print summary statistics for a return series."""
    for col in ['gross_ret', 'net_ret']:
        r = returns[col]
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        # Maximum drawdown
        cum = (1 + r).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_dd = drawdown.min()

        # Calmar ratio
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        # Hit rate
        hit_rate = (r > 0).mean()

        label_col = f"{label} ({col})" if label else col
        print(f"\n  {label_col}:")
        print(f"    Annualized return:  {ann_ret:.4f} ({ann_ret*100:.1f}%)")
        print(f"    Annualized vol:     {ann_vol:.4f} ({ann_vol*100:.1f}%)")
        print(f"    Sharpe ratio:       {sharpe:.2f}")
        print(f"    Max drawdown:       {max_dd:.4f} ({max_dd*100:.1f}%)")
        print(f"    Calmar ratio:       {calmar:.2f}")
        print(f"    Hit rate (daily):   {hit_rate:.3f} ({hit_rate*100:.1f}%)")

    # Turnover stats
    total_costs = returns['cost'].sum()
    ann_cost = total_costs / (len(returns) / 252)
    print(f"\n  Transaction costs:")
    print(f"    Total cost drag:    {total_costs:.4f} ({total_costs*100:.2f}%)")
    print(f"    Annualized cost:    {ann_cost:.4f} ({ann_cost*100:.2f}%)")


def run_portfolio_construction():
    """Full pipeline: signals -> weights -> returns."""

    print("=" * 60)
    print("STAGE 3: PORTFOLIO CONSTRUCTION")
    print("=" * 60)

    # Load signals
    signals_path = DATA_PROCESSED / "signals.parquet"
    if not signals_path.exists():
        raise FileNotFoundError(
            f"Missing {signals_path}. Run momentum.py first."
        )
    signals = pd.read_parquet(signals_path)
    print(f"\nLoaded signals: {len(signals):,} rows, "
          f"{signals['date'].nunique()} rebalance dates")

    # Load daily panel
    panel_path = DATA_PROCESSED / "daily_panel.parquet"
    panel = pd.read_parquet(panel_path)
    print(f"Loaded panel: {len(panel):,} rows")

    # --- Compute weights ---
    weights = compute_weights(signals)

    # Save weights
    weights_path = DATA_PROCESSED / "weights.parquet"
    weights.to_parquet(weights_path, index=False)
    print(f"  Saved weights to {weights_path}")

    # --- Compute returns ---
    returns = compute_portfolio_returns(weights, panel)

    # Split into in-sample and holdout
    insample = returns[
        (returns['date'] >= pd.Timestamp(INSAMPLE_START)) &
        (returns['date'] <= pd.Timestamp('2019-12-31'))
    ]
    holdout = returns[
        (returns['date'] >= pd.Timestamp('2020-01-01')) &
        (returns['date'] <= pd.Timestamp(HOLDOUT_END))
    ]

    # Save full returns
    returns_path = DATA_PROCESSED / "portfolio_returns.parquet"
    returns.to_parquet(returns_path, index=False)
    print(f"  Saved returns to {returns_path}")

    # --- Summary stats (in-sample only for now) ---
    print("\n" + "=" * 60)
    print("IN-SAMPLE RESULTS (2005-2019)")
    print("=" * 60)
    compute_summary_stats(insample, "In-sample")

    print(f"\n  Date range: {insample['date'].min().date()} to "
          f"{insample['date'].max().date()}")
    print(f"  Trading days: {len(insample)}")

    # --- Equity curve plot ---
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Cumulative returns
    cum_gross = (1 + insample['gross_ret']).cumprod()
    cum_net = (1 + insample['net_ret']).cumprod()
    axes[0].plot(insample['date'].values, cum_gross.values,
                 label='Gross', linewidth=1)
    axes[0].plot(insample['date'].values, cum_net.values,
                 label='Net of costs', linewidth=1)
    axes[0].set_ylabel('Cumulative return')
    axes[0].set_title('In-Sample Equity Curve (2005-2019)')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Drawdown
    cum = (1 + insample['net_ret']).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    axes[1].fill_between(insample['date'].values, drawdown.values, 0,
                         alpha=0.5, color='red')
    axes[1].set_ylabel('Drawdown')
    axes[1].set_title('Drawdown (Net of Costs)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_dir = PROJECT_ROOT / "notebooks"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / "stage3_equity_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Equity curve saved to {path}")

    # --- Pre-registration check ---
    net_sharpe = (insample['net_ret'].mean() * 252) / \
                 (insample['net_ret'].std() * np.sqrt(252))
    print(f"\n" + "=" * 60)
    print("PRE-REGISTRATION CHECK (Stage 0, Section 2)")
    print("=" * 60)
    print(f"  Net Sharpe ratio: {net_sharpe:.2f}")

    if 0.3 <= net_sharpe <= 0.9:
        print("  ✓ In predicted range [0.3, 0.9] — pipeline validated")
    elif 0.0 <= net_sharpe < 0.3:
        print("  ⚠ Below predicted range — plausible if dominated by "
              "momentum drawdown periods. Investigate rolling Sharpe.")
    elif 0.9 < net_sharpe <= 1.5:
        print("  ⚠ Above predicted range — suspicious. Audit for "
              "look-ahead bias, unrealistic costs, or universe contamination.")
    elif net_sharpe > 1.5:
        print("  ✗ Almost certainly a bug. Halt and debug.")
    else:
        print("  ✗ Negative Sharpe — pipeline likely broken. "
              "Check for sign errors, look-ahead, mishandled delistings.")

    print("\n" + "=" * 60)
    print("STAGE 3 COMPLETE")
    print("=" * 60)
    print("Review the equity curve plot and summary stats.")
    print("Do NOT examine holdout results yet — that's Stage 7.")


if __name__ == "__main__":
    run_portfolio_construction()
```

### Running the portfolio construction

```bash
python -m src.portfolio.construction
```

---

## 4. Deliverables Checklist

Before moving to Stage 4, confirm every item:

- [ ] `src/portfolio/construction.py` — portfolio construction module
- [ ] `data/processed/weights.parquet` — portfolio weights at each rebalance
- [ ] `data/processed/portfolio_returns.parquet` — daily gross and net returns
- [ ] `notebooks/stage3_equity_curve.png` — in-sample equity curve plot
- [ ] In-sample net Sharpe ratio falls in pre-registered range (see Stage 0 Section 2)
- [ ] Summary stats reported: return, vol, Sharpe, max drawdown, Calmar, hit rate, costs
- [ ] Holdout results NOT examined
- [ ] All code committed to git

> **Only proceed to Stage 4 when all boxes are checked.**

---

## Appendix A: What This Stage Does and Doesn't Do

**Does:**
- Converts signals to portfolio weights (top/bottom decile, equal-weighted, dollar-neutral)
- Computes daily gross and net portfolio returns
- Applies a flat transaction cost model at each rebalance
- Reports summary statistics and the pre-registration Sharpe check
- Plots the equity curve and drawdown

**Doesn't (yet):**
- Weight drift between rebalances (simplified to fixed weights within each month — Stage 4 will implement drift-aware returns)
- Stock-specific costs (uses flat 2.5 bp per dollar traded for all stocks)
- Sector-neutral or beta-neutral construction (Stage 6 robustness)
- Short-borrow costs (Stage 6 robustness)
- Holdout evaluation (Stage 7 only)

The simplified fixed-weight approach slightly overstates turnover (because in reality weights drift toward the new target during the month), but this is conservative — it overcharges transaction costs rather than undercharging them.
