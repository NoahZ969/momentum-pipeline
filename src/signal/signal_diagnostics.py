"""
signal_diagnostics.py — Verify that the computed signals have sensible properties.

Run after momentum.py has produced data/processed/signals.parquet.

Usage:
    python -m src.signal.signal_diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED, DATA_EXTERNAL

PLOT_DIR = PROJECT_ROOT / "notebooks"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_signals() -> pd.DataFrame:
    path = DATA_PROCESSED / "signals.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run momentum.py first: python -m src.signal.momentum"
        )
    signals = pd.read_parquet(path)
    print(f"Loaded signals: {len(signals):,} rows, "
          f"{signals['permno'].nunique()} unique permnos, "
          f"{signals['date'].nunique()} rebalance dates")
    return signals


# =============================================================================
# Diagnostic 1: Signal coverage over time
# =============================================================================

def check_coverage(signals: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Diagnostic 1: Signal coverage over time")
    print("=" * 60)

    coverage = signals.groupby('date')['permno'].count().rename('n_stocks')

    print(f"  Min:    {coverage.min()}")
    print(f"  Max:    {coverage.max()}")
    print(f"  Mean:   {coverage.mean():.0f}")
    print(f"  Median: {coverage.median():.0f}")

    fig, ax = plt.subplots(figsize=(14, 5))
    coverage.plot(ax=ax, linewidth=1)
    ax.axhline(y=coverage.mean(), color='red', linestyle='--', alpha=0.5,
               label=f'Mean = {coverage.mean():.0f}')
    ax.set_ylabel('Number of stocks scored')
    ax.set_title('Signal Coverage Over Time')
    ax.legend()
    plt.tight_layout()
    path = PLOT_DIR / "diag1_signal_coverage.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved to {path}")

    if coverage.min() > 350:
        print("  ✓ Coverage looks reasonable")
    else:
        print(f"  ⚠ Some dates have low coverage ({coverage.min()}). Investigate.")


# =============================================================================
# Diagnostic 2: Signal distribution
# =============================================================================

def check_distribution(signals: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Diagnostic 2: Signal distribution")
    print("=" * 60)

    print(f"  z_signal mean:   {signals['z_signal'].mean():.4f}  (expected ≈ 0)")
    print(f"  z_signal std:    {signals['z_signal'].std():.4f}  (expected ≈ 1)")
    print(f"  z_signal skew:   {signals['z_signal'].skew():.4f}")
    print(f"  z_signal kurt:   {signals['z_signal'].kurtosis():.4f}")

    extremes = signals[signals['z_signal'].abs() > 4]
    print(f"  |z| > 4:         {len(extremes)} observations "
          f"({len(extremes)/len(signals)*100:.3f}%)")

    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of z-scores
    axes[0].hist(signals['z_signal'], bins=100, edgecolor='none', alpha=0.7)
    axes[0].set_xlabel('z_signal')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Cross-Sectional Z-Score Distribution (All Dates)')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # Histogram of raw signal
    axes[1].hist(signals['raw_signal'], bins=100, edgecolor='none', alpha=0.7)
    axes[1].set_xlabel('raw_signal (log return)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Raw Momentum Signal Distribution (All Dates)')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = PLOT_DIR / "diag2_signal_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved to {path}")

    if abs(signals['z_signal'].mean()) < 0.05 and abs(signals['z_signal'].std() - 1.0) < 0.1:
        print("  ✓ Distribution looks correct")
    else:
        print("  ⚠ Distribution stats outside expected range. Investigate.")


# =============================================================================
# Diagnostic 3: Signal autocorrelation
# =============================================================================

def check_autocorrelation(signals: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Diagnostic 3: Month-to-month signal autocorrelation")
    print("=" * 60)

    rebal_dates = sorted(signals['date'].unique())
    autocorrs = []

    for i in range(1, len(rebal_dates)):
        prev = signals[signals['date'] == rebal_dates[i-1]].set_index('permno')['z_signal']
        curr = signals[signals['date'] == rebal_dates[i]].set_index('permno')['z_signal']
        common = prev.index.intersection(curr.index)
        if len(common) > 30:
            corr, _ = spearmanr(prev.loc[common], curr.loc[common])
            autocorrs.append({'date': rebal_dates[i], 'corr': corr})

    ac_df = pd.DataFrame(autocorrs)
    mean_ac = ac_df['corr'].mean()
    std_ac = ac_df['corr'].std()

    print(f"  Mean rank correlation:   {mean_ac:.3f}  (expected ~0.85-0.95)")
    print(f"  Std rank correlation:    {std_ac:.3f}")
    print(f"  Min:                     {ac_df['corr'].min():.3f}")
    print(f"  Max:                     {ac_df['corr'].max():.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ac_df['date'], ac_df['corr'], linewidth=1)
    ax.axhline(y=mean_ac, color='red', linestyle='--', alpha=0.5,
               label=f'Mean = {mean_ac:.3f}')
    ax.set_ylabel('Spearman rank correlation')
    ax.set_title('Month-to-Month Signal Rank Autocorrelation')
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = PLOT_DIR / "diag3_signal_autocorrelation.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved to {path}")

    if 0.80 < mean_ac < 0.98:
        print("  ✓ Autocorrelation in expected range (momentum is slow-moving)")
    else:
        print(f"  ⚠ Autocorrelation {mean_ac:.3f} outside expected range. Investigate.")


# =============================================================================
# Diagnostic 4: Top/bottom decile preview
# =============================================================================

def preview_deciles(signals: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Diagnostic 4: Top/bottom decile preview")
    print("=" * 60)

    # Pick a sample rebalance date near the middle of the in-sample period
    rebal_dates = sorted(signals['date'].unique())
    mid_date = rebal_dates[len(rebal_dates) // 2]
    sample = signals[signals['date'] == mid_date].sort_values('z_signal')

    n = len(sample)
    decile_size = n // 10

    bottom = sample.head(decile_size)
    top = sample.tail(decile_size)

    print(f"\n  Sample date: {mid_date.date()} ({n} stocks)")
    print(f"  Decile size: {decile_size}")

    print(f"\n  Bottom decile (short leg):")
    print(f"    z_signal range: [{bottom['z_signal'].min():.2f}, "
          f"{bottom['z_signal'].max():.2f}]")
    print(f"    raw_signal range: [{bottom['raw_signal'].min():.3f}, "
          f"{bottom['raw_signal'].max():.3f}]")
    print(f"    raw_signal means losers with ~"
          f"{(np.exp(bottom['raw_signal'].mean()) - 1) * 100:.0f}% "
          f"return over past 12-1 months")

    print(f"\n  Top decile (long leg):")
    print(f"    z_signal range: [{top['z_signal'].min():.2f}, "
          f"{top['z_signal'].max():.2f}]")
    print(f"    raw_signal range: [{top['raw_signal'].min():.3f}, "
          f"{top['raw_signal'].max():.3f}]")
    print(f"    raw_signal means winners with ~"
          f"{(np.exp(top['raw_signal'].mean()) - 1) * 100:.0f}% "
          f"return over past 12-1 months")

    print(f"\n  Spread (long - short raw signal): "
          f"{top['raw_signal'].mean() - bottom['raw_signal'].mean():.3f}")


# =============================================================================
# Main
# =============================================================================

def run_all_diagnostics():
    print("=" * 60)
    print("STAGE 2 SIGNAL DIAGNOSTICS")
    print("=" * 60)

    signals = load_signals()

    check_coverage(signals)
    check_distribution(signals)
    check_autocorrelation(signals)
    preview_deciles(signals)

    print("\n" + "=" * 60)
    print("ALL DIAGNOSTICS COMPLETE")
    print("=" * 60)
    print("Review the plots in notebooks/ and confirm all diagnostics look reasonable.")
    print("If everything checks out, proceed to Stage 3: Portfolio Construction.")


if __name__ == "__main__":
    run_all_diagnostics()
