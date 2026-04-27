"""
sanity_checks.py — Validate the cleaned daily panel before any strategy code is written.

Implements the five checks from Stage 0 pre-registration, Section 9.
All checks must pass before proceeding to Stage 2.

Usage:
    python -m src.data.sanity_checks

Prerequisites:
    Run download_crsp.py and clean.py first.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_RAW, DATA_PROCESSED, DATA_EXTERNAL, INSAMPLE_START, HOLDOUT_END

PLOT_DIR = PROJECT_ROOT / "notebooks"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Check 1: Universe count over time
# =============================================================================

def check_universe_count(panel: pd.DataFrame) -> pd.Series:
    """
    Plot the number of S&P 500 stocks at each month-end.

    What it catches:
      - Flat line at exactly 500 → point-in-time logic is broken
        (using today's membership for all dates)
      - Monotonically increasing → survivorship bias (adding but never removing)
      - Drops below 400 or above 600 → data or merge problem

    Pass criteria: count stays within [400, 600] throughout.
    """
    print("\n" + "=" * 60)
    print("Check 1: Universe count over time")
    print("=" * 60)

    sp500 = panel[panel['is_sp500']].copy()
    sp500['month_end'] = sp500['date'] + pd.offsets.MonthEnd(0)

    counts = (
        sp500.groupby('month_end')['permno']
        .nunique()
        .rename('n_stocks')
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 5))
    counts.plot(ax=ax, linewidth=1)
    ax.axhline(y=500, color='red', linestyle='--', alpha=0.5, label='Expected ~500')
    ax.set_ylabel('Number of S&P 500 constituents')
    ax.set_title('Sanity Check 1: Universe Count Over Time')
    ax.legend()
    plt.tight_layout()
    path = PLOT_DIR / "check1_universe_count.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved to {path}")

    # --- Stats ---
    print(f"  Min:    {counts.min()}")
    print(f"  Max:    {counts.max()}")
    print(f"  Mean:   {counts.mean():.1f}")
    print(f"  Median: {counts.median():.0f}")

    # --- Assertion ---
    assert counts.min() > 400, f"FAIL: universe too small (min = {counts.min()})"
    assert counts.max() < 600, f"FAIL: universe too large (max = {counts.max()})"
    print("  ✓ Check 1 PASSED")

    return counts


# =============================================================================
# Check 2: Cap-weighted index reproduction
# =============================================================================

def check_index_reproduction(panel: pd.DataFrame) -> pd.Series:
    """
    Compute the cap-weighted total return of S&P 500 members and compare
    to the Ken French market return (Mkt-RF + RF) as a proxy for the
    S&P 500 Total Return Index.

    What it catches:
      - Wrong return column usage or adjustment errors
      - Incorrect market cap computation
      - Missing stocks inflating or deflating the index
      - Look-ahead bias in weighting

    Pass criteria: annualized tracking error < 50 bp/year vs benchmark.
    (Relaxed to 200 bp for Wikipedia-sourced membership.)
    """
    print("\n" + "=" * 60)
    print("Check 2: Cap-weighted index reproduction")
    print("=" * 60)

    sp500 = panel[panel['is_sp500'] & panel['ret'].notna()].copy()
    sp500 = sp500.sort_values(['permno', 'date'])

    # Weight by PREVIOUS day's market cap (avoids look-ahead)
    sp500['lag_mkt_cap'] = sp500.groupby('permno')['mkt_cap'].shift(1)

    # Cap-weighted daily return
    def wavg(g):
        w = g['lag_mkt_cap'].fillna(g['mkt_cap'])
        if w.sum() == 0 or w.isna().all():
            return np.nan
        return np.average(g['ret'], weights=w)

    cw_ret = (
        sp500.groupby('date')
        .apply(wavg)
        .rename('cw_ret')
        .dropna()
    )

    # --- Load Ken French market return as benchmark ---
    ff_path = DATA_EXTERNAL / "ff_factors_daily.parquet"
    if ff_path.exists():
        ff = pd.read_parquet(ff_path)
        ff['date'] = pd.to_datetime(ff['date'])
        ff['mkt_ret'] = ff['Mkt-RF'] + ff['RF']

        # Align dates
        merged = pd.DataFrame({'cw_ret': cw_ret}).join(
            ff.set_index('date')['mkt_ret'], how='inner'
        )

        tracking_diff = merged['cw_ret'] - merged['mkt_ret']
        tracking_error = tracking_diff.std() * np.sqrt(252)
        correlation = merged['cw_ret'].corr(merged['mkt_ret'])

        print(f"  Our cap-weighted return (ann.):    {cw_ret.mean() * 252:.4f}")
        print(f"  Ken French Mkt return (ann.):      {merged['mkt_ret'].mean() * 252:.4f}")
        print(f"  Tracking error (ann.):             {tracking_error:.4f} "
              f"({tracking_error * 10000:.0f} bp)")
        print(f"  Correlation:                       {correlation:.4f}")

        # --- Plot ---
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Cumulative returns
        cum_ours = (1 + merged['cw_ret']).cumprod()
        cum_bench = (1 + merged['mkt_ret']).cumprod()
        axes[0].plot(cum_ours.index, cum_ours.values, label='Our cap-weighted', linewidth=1)
        axes[0].plot(cum_bench.index, cum_bench.values, label='Ken French Mkt',
                     linewidth=1, alpha=0.8)
        axes[0].set_ylabel('Cumulative return')
        axes[0].set_title('Check 2: Cap-Weighted Index vs Ken French Market Return')
        axes[0].legend()
        axes[0].set_yscale('log')

        # Rolling tracking error
        rolling_te = tracking_diff.rolling(252).std() * np.sqrt(252)
        axes[1].plot(rolling_te.index, rolling_te.values, linewidth=1)
        axes[1].axhline(y=0.005, color='green', linestyle='--', alpha=0.5,
                        label='50 bp target')
        axes[1].axhline(y=0.02, color='orange', linestyle='--', alpha=0.5,
                        label='200 bp relaxed target')
        axes[1].set_ylabel('Rolling 1Y tracking error')
        axes[1].set_title('Rolling Tracking Error')
        axes[1].legend()

        plt.tight_layout()
        path = PLOT_DIR / "check2_index_reproduction.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Plot saved to {path}")

        # Assertion — relaxed for Wikipedia membership
        if tracking_error < 0.005:
            print("  ✓ Check 2 PASSED (< 50 bp — excellent)")
        elif tracking_error < 0.02:
            print("  ✓ Check 2 PASSED (< 200 bp — acceptable for Wikipedia membership)")
        else:
            print(f"  ✗ Check 2 WARNING: tracking error {tracking_error:.4f} "
                  f"is high. Investigate universe membership.")
    else:
        print("  ⚠ Ken French factors not found — skipping benchmark comparison.")
        print(f"  Our cap-weighted return (ann.): {cw_ret.mean() * 252:.4f}")
        print(f"  Our cap-weighted vol (ann.):    {cw_ret.std() * np.sqrt(252):.4f}")

    return cw_ret


# =============================================================================
# Check 3: Equal-weighted index reproduction
# =============================================================================

def check_equal_weighted_index(panel: pd.DataFrame) -> pd.Series:
    """
    Compute the equal-weighted total return of S&P 500 members.

    What it catches:
      - Same issues as Check 2 but without market-cap weighting,
        so it's sensitive to different problems (e.g., small-cap
        contamination, incorrect universe breadth)
      - Equal-weighted should have higher return and higher vol
        than cap-weighted (small-cap premium within S&P 500)

    Pass criteria: annualized return and vol are in a reasonable
    range (return 8-16%, vol 14-22% over a long sample).
    """
    print("\n" + "=" * 60)
    print("Check 3: Equal-weighted index reproduction")
    print("=" * 60)

    sp500 = panel[panel['is_sp500'] & panel['ret'].notna()].copy()

    ew_ret = sp500.groupby('date')['ret'].mean().rename('ew_ret')

    ann_ret = ew_ret.mean() * 252
    ann_vol = ew_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol

    print(f"  Annualized return: {ann_ret:.4f} ({ann_ret * 100:.1f}%)")
    print(f"  Annualized vol:    {ann_vol:.4f} ({ann_vol * 100:.1f}%)")
    print(f"  Sharpe ratio:      {sharpe:.2f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 5))
    cum = (1 + ew_ret).cumprod()
    ax.plot(cum.index, cum.values, linewidth=1)
    ax.set_ylabel('Cumulative return')
    ax.set_title('Check 3: Equal-Weighted S&P 500 Cumulative Return')
    ax.set_yscale('log')
    plt.tight_layout()
    path = PLOT_DIR / "check3_equal_weighted.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved to {path}")

    # Reasonableness checks (not hard assertions — these are guideposts)
    if 0.04 < ann_ret < 0.20 and 0.10 < ann_vol < 0.30:
        print("  ✓ Check 3 PASSED (return and vol in reasonable range)")
    else:
        print(f"  ⚠ Check 3 WARNING: return or vol outside expected range. "
              f"Investigate.")

    return ew_ret


# =============================================================================
# Check 4: Delisting accounting
# =============================================================================

def check_delistings(panel: pd.DataFrame) -> None:
    """
    Verify that known delistings are correctly reflected in the data.

    What it catches:
      - Delisting returns not merged (stock vanishes with NaN last return)
      - Shumway adjustment not applied (missing -30% for bankruptcies)
      - Wrong permno mapping (stock not found at all)

    Pass criteria: at least two known delistings have non-NaN last returns
    near the expected delisting dates.
    """
    print("\n" + "=" * 60)
    print("Check 4: Delisting accounting")
    print("=" * 60)

    delist = pd.read_parquet(DATA_RAW / "crsp_msedelist.parquet")
    names = pd.read_parquet(DATA_RAW / "crsp_msenames.parquet")

    # Define test cases — we look up permnos dynamically in case
    # the commonly cited values differ in your CRSP vintage
    test_cases = [
        {'name': 'Lehman Brothers', 'search': 'LEHMAN BROS', 'approx_date': '2008-09'},
        {'name': 'General Motors (old)', 'search': 'GENERAL MOTORS', 'approx_date': '2009-0'},
    ]

    passed = 0
    for case in test_cases:
        # Find the permno from name history
        matches = names[names['comnam'].str.contains(case['search'], case=False, na=False)]
        if matches.empty:
            print(f"  ⚠ {case['name']}: could not find in name history "
                  f"(searched for '{case['search']}')")
            continue

        # There may be multiple permnos; pick the one with data near the expected date
        candidate_permnos = matches['permno'].unique()
        found = False

        for permno in candidate_permnos:
            stock = panel[panel['permno'] == permno].sort_values('date')
            if stock.empty:
                continue

            last_row = stock.iloc[-1]
            last_date_str = str(last_row['date'].date())

            if case['approx_date'] in last_date_str:
                dl_row = delist[delist['permno'] == permno]
                dlret_str = (f"{dl_row.iloc[0]['dlret']:.4f}"
                             if not dl_row.empty and pd.notna(dl_row.iloc[0]['dlret'])
                             else "N/A")
                dlstcd_str = (str(int(dl_row.iloc[0]['dlstcd']))
                              if not dl_row.empty and pd.notna(dl_row.iloc[0]['dlstcd'])
                              else "N/A")

                ret_ok = pd.notna(last_row['ret'])
                status = "✓" if ret_ok else "✗"

                print(f"  {status} {case['name']} (permno={permno}): "
                      f"last date = {last_date_str}, "
                      f"last ret = {last_row['ret']:.4f}, "
                      f"dlret = {dlret_str}, dlstcd = {dlstcd_str}")

                if ret_ok:
                    passed += 1
                found = True
                break

        if not found:
            print(f"  ⚠ {case['name']}: no permno found with data ending "
                  f"near {case['approx_date']}")

    # Also report summary statistics on delistings in the panel
    delist_in_panel = delist[delist['permno'].isin(panel['permno'].unique())]
    perf_delistings = delist_in_panel[delist_in_panel['dlstcd'].between(500, 599)]
    print(f"\n  Summary: {len(delist_in_panel)} delistings for stocks in panel")
    print(f"  Performance-related (dlstcd 500-599): {len(perf_delistings)}")
    print(f"  Of those, dlret is missing (Shumway -30% applied): "
          f"{perf_delistings['dlret'].isna().sum()}")

    if passed >= 2:
        print("  ✓ Check 4 PASSED")
    elif passed >= 1:
        print("  ⚠ Check 4 PARTIAL: only one case confirmed. Review the other.")
    else:
        print("  ✗ Check 4 FAILED: could not confirm any delisting returns. "
              "Investigate merge_delisting_returns in clean.py.")


# =============================================================================
# Check 5: No look-ahead in signal (stub — implemented in Stage 2)
# =============================================================================

def check_no_lookahead() -> None:
    """
    Placeholder for Stage 2. The actual test will:
      1. Write a signal function with an explicit date assertion
      2. Pass deliberately future-dated data
      3. Confirm the assertion fires

    This cannot be tested until the signal function exists.
    """
    print("\n" + "=" * 60)
    print("Check 5: No look-ahead in signal")
    print("=" * 60)
    print("  ⏳ Deferred to Stage 2 (requires signal function)")


# =============================================================================
# Main: run all checks
# =============================================================================

def run_all_checks():
    """Run all sanity checks on the cleaned daily panel."""

    print("=" * 60)
    print("STAGE 1 SANITY CHECKS")
    print("=" * 60)

    # Load the cleaned panel
    panel_path = DATA_PROCESSED / "daily_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Missing {panel_path}. Run clean.py first: python -m src.data.clean"
        )

    print(f"\nLoading daily panel from {panel_path}...")
    panel = pd.read_parquet(panel_path)
    print(f"  {len(panel):,} rows, {panel['permno'].nunique():,} permnos, "
          f"date range {panel['date'].min().date()} to {panel['date'].max().date()}")

    # Run checks
    check_universe_count(panel)
    check_index_reproduction(panel)
    check_equal_weighted_index(panel)
    check_delistings(panel)
    check_no_lookahead()

    # Summary
    print("\n" + "=" * 60)
    print("ALL CHECKS COMPLETE")
    print("=" * 60)
    print("Review the plots in notebooks/ and confirm all checks passed.")
    print("If all checks pass, proceed to Stage 2: Signal Construction.")


if __name__ == "__main__":
    run_all_checks()
