"""
clean.py — Clean raw CRSP data, merge delisting returns, build the daily panel.

Takes the raw Parquet files produced by download_crsp.py and outputs a single
clean daily panel at data/processed/daily_panel.parquet, ready for signal
computation in Stage 2.

Usage:
    python -m src.data.clean

Prerequisites:
    Run download_crsp.py first to populate data/raw/.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_RAW, DATA_PROCESSED,
    VALID_SHARE_CODES, VALID_EXCHANGE_CODES,
)


# =============================================================================
# Step 1: Load raw data
# =============================================================================

def load_raw_data() -> dict:
    """
    Load all raw Parquet files into a dict of DataFrames.
    Raises FileNotFoundError if any are missing (run download_crsp.py first).
    """
    files = {
        'dsf':        DATA_RAW / "crsp_dsf.parquet",
        'names':      DATA_RAW / "crsp_msenames.parquet",
        'delist':     DATA_RAW / "crsp_msedelist.parquet",
        'sp500':      DATA_RAW / "crsp_sp500_membership.parquet",
    }

    data = {}
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run download_crsp.py first."
            )
        data[key] = pd.read_parquet(path)
        print(f"Loaded {key}: {len(data[key]):,} rows from {path.name}")

    return data


# =============================================================================
# Step 2: Filter to common stocks on major exchanges (point-in-time)
# =============================================================================

def filter_common_stocks(
    dsf: pd.DataFrame,
    names: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only daily records where the stock was a common domestic stock
    (shrcd in [10, 11]) trading on a major exchange (exchcd in [1, 2, 3])
    on that date.

    This is a point-in-time filter: a stock that moved from NYSE to OTC
    mid-sample is included only for its NYSE days.

    The join uses the name history's [namedt, nameendt] validity window
    to ensure we apply the share code and exchange code that were current
    on each trading date — not a future or retroactive classification.
    """
    print("\nFiltering to common stocks on major exchanges...")

    # Keep only name records with valid share/exchange codes
    valid = names[
        names['shrcd'].isin(VALID_SHARE_CODES) &
        names['exchcd'].isin(VALID_EXCHANGE_CODES)
    ][['permno', 'namedt', 'nameendt']].copy()

    n_before = len(dsf)

    # Inner merge: drops dsf rows whose permno has no valid name record at all
    merged = dsf.merge(valid, on='permno', how='inner')

    # Keep only rows where the trading date falls within the name validity window
    merged = merged[
        (merged['date'] >= merged['namedt']) &
        (merged['date'] <= merged['nameendt'])
    ]

    # A permno-date can appear multiple times if name records overlap
    # (rare but possible at transitions). Keep first occurrence.
    merged = merged.drop_duplicates(subset=['permno', 'date'], keep='first')

    # Drop the name window columns — no longer needed
    merged = merged.drop(columns=['namedt', 'nameendt'])

    n_after = len(merged)
    print(f"  {n_before:,} -> {n_after:,} rows "
          f"({n_before - n_after:,} removed, "
          f"{(n_before - n_after) / n_before:.1%} of total)")

    return merged


# =============================================================================
# Step 3: Fix CRSP price convention
# =============================================================================

def fix_prices(dsf: pd.DataFrame) -> pd.DataFrame:
    """
    CRSP convention: when the closing price is missing, prc is stored as the
    negative of the bid-ask midpoint. Take absolute value for all downstream use.
    """
    print("\nFixing negative prices (bid-ask midpoint convention)...")
    n_negative = (dsf['prc'] < 0).sum()
    dsf['prc'] = dsf['prc'].abs()
    print(f"  {n_negative:,} negative prices converted to absolute value")
    return dsf


# =============================================================================
# Step 4: Compute market cap
# =============================================================================

def compute_market_cap(dsf: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market capitalization in millions of dollars.
    
    CRSP fields:
      - prc:    price in dollars (absolute value after fix_prices)
      - shrout: shares outstanding in thousands
      
    mkt_cap = prc * shrout / 1000  (in millions)
    """
    print("\nComputing market cap...")
    dsf['mkt_cap'] = dsf['prc'] * dsf['shrout'] / 1000.0

    n_missing = dsf['mkt_cap'].isna().sum()
    if n_missing > 0:
        print(f"  ⚠ {n_missing:,} rows with missing market cap "
              f"(missing price or shares outstanding)")

    return dsf


# =============================================================================
# Step 5: Merge delisting returns
# =============================================================================

def merge_delisting_returns(
    dsf: pd.DataFrame,
    delist: pd.DataFrame,
) -> pd.DataFrame:
    """
    Incorporate delisting returns into the daily return series.

    Logic (standard treatment per Shumway 1997, Beaver et al. 2007):

    1. If a stock has a delisting return (dlret) on its delisting date,
       compound it with the regular return on that date:
         adjusted_ret = (1 + ret) * (1 + dlret) - 1

    2. If the delisting date has no regular return (stock stopped trading
       before the formal delisting), use the delisting return alone:
         adjusted_ret = dlret

    3. If dlret is missing for a performance-related delisting
       (dlstcd 500-599, which includes bankruptcies and failures),
       assume dlret = -0.30 (Shumway 1997 convention).
       This is conservative but standard. Without this adjustment,
       these stocks silently vanish with an implicit return of 0%,
       which biases backtests upward — especially for the short leg
       of a momentum strategy, where bankrupt stocks cluster.

    4. For non-performance delistings with missing dlret (mergers,
       exchange switches, etc.), we do not impute a return. These
       are typically benign events where the shareholder receives
       fair value.
    """
    print("\nMerging delisting returns...")

    dl = delist[['permno', 'dlstdt', 'dlret', 'dlstcd']].copy()
    dl = dl.rename(columns={'dlstdt': 'date'})

    # --- Shumway adjustment for missing performance delistings ---
    performance_delist = dl['dlstcd'].between(500, 599)
    n_shumway = (performance_delist & dl['dlret'].isna()).sum()
    dl.loc[performance_delist & dl['dlret'].isna(), 'dlret'] = -0.30
    print(f"  Applied Shumway -30% to {n_shumway} performance delistings "
          f"with missing dlret")

    # --- Merge onto daily file ---
    n_before = dsf['ret'].notna().sum()
    dsf = dsf.merge(
        dl[['permno', 'date', 'dlret']],
        on=['permno', 'date'],
        how='left',
    )

    # --- Compound delisting return with regular return ---
    has_dlret = dsf['dlret'].notna()
    has_ret = dsf['ret'].notna()

    # Case 1: both ret and dlret exist — compound them
    both = has_dlret & has_ret
    dsf.loc[both, 'ret'] = (
        (1 + dsf.loc[both, 'ret']) *
        (1 + dsf.loc[both, 'dlret']) - 1
    )

    # Case 2: dlret exists but ret is NaN — use dlret as the return
    dlret_only = has_dlret & ~has_ret
    dsf.loc[dlret_only, 'ret'] = dsf.loc[dlret_only, 'dlret']

    n_affected = both.sum() + dlret_only.sum()
    print(f"  {n_affected:,} returns adjusted "
          f"({both.sum()} compounded, {dlret_only.sum()} filled from dlret)")

    dsf = dsf.drop(columns=['dlret'])
    return dsf


# =============================================================================
# Step 6: Add S&P 500 point-in-time membership flag
# =============================================================================

def add_sp500_flag(
    dsf: pd.DataFrame,
    sp500: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a boolean column 'is_sp500' indicating whether each permno was
    a member of the S&P 500 on each trading date.

    Uses the CRSP msp500list table, which provides (permno, start, ending)
    intervals. A permno can have multiple stints (leave and rejoin).

    Implementation: for each membership interval, flag all dsf rows
    for that permno within the interval. Then collapse to one row per
    permno-date, with is_sp500 = True if any interval covers that date.

    Note: This is an O(N * M) operation in the worst case, but with
    ~1,500 membership intervals and ~2M dsf rows, it runs in seconds
    using vectorized pandas operations via a merge-and-filter approach.
    """
    print("\nAdding S&P 500 membership flag...")

    sp = sp500[['permno', 'start', 'ending']].copy()
    # Fill missing end dates — stock is still in the index
    sp['ending'] = sp['ending'].fillna(pd.Timestamp('2099-12-31'))

    # Strategy: merge membership intervals onto dsf by permno,
    # then flag rows where date falls within [start, ending].
    # Finally, collapse duplicates (multiple stints).

    # To avoid a massive cross-join, we iterate over membership
    # records and build a set of (permno, date) pairs that are in the S&P 500.
    # This is more memory-efficient for our data size.

    # Get all unique (permno, date) pairs in dsf for permnos that were ever in S&P 500
    sp500_permnos = set(sp['permno'].unique())
    dsf_sp = dsf[dsf['permno'].isin(sp500_permnos)][['permno', 'date']].copy()

    # Merge on permno to get all candidate intervals for each daily row
    dsf_sp = dsf_sp.merge(sp, on='permno', how='left')

    # Flag rows where date is within the membership interval
    dsf_sp['in_interval'] = (
        (dsf_sp['date'] >= dsf_sp['start']) &
        (dsf_sp['date'] <= dsf_sp['ending'])
    )

    # Collapse: a permno-date is an S&P 500 member if ANY interval covers it
    sp500_flags = (
        dsf_sp
        .groupby(['permno', 'date'])['in_interval']
        .any()
        .rename('is_sp500')
        .reset_index()
    )

    # Merge the flag back onto the full dsf
    dsf = dsf.merge(sp500_flags, on=['permno', 'date'], how='left')
    dsf['is_sp500'] = dsf['is_sp500'].fillna(False)

    n_sp500_rows = dsf['is_sp500'].sum()
    n_sp500_permnos = dsf.loc[dsf['is_sp500'], 'permno'].nunique()
    print(f"  {n_sp500_rows:,} daily rows flagged as S&P 500 members")
    print(f"  {n_sp500_permnos:,} unique permnos were in the S&P 500 at some point")

    return dsf


# =============================================================================
# Step 7: Final assembly and save
# =============================================================================

def assemble_and_save(dsf: pd.DataFrame) -> pd.DataFrame:
    """
    Sort, validate, and save the final daily panel.

    Output columns:
      - permno:   int, permanent security identifier
      - date:     datetime, trading date
      - ret:      float, holding-period total return (with delisting adjustment)
      - prc:      float, closing price (absolute value)
      - vol:      float, trading volume (shares)
      - shrout:   float, shares outstanding (thousands)
      - mkt_cap:  float, market cap in millions of dollars
      - cfacpr:   float, cumulative price adjustment factor
      - cfacshr:  float, cumulative share adjustment factor
      - is_sp500: bool, S&P 500 member on this date (point-in-time)
    """
    print("\nFinal assembly...")

    # Sort by permno, then date
    dsf = dsf.sort_values(['permno', 'date']).reset_index(drop=True)

    # Ensure no duplicate permno-date pairs
    n_dupes = dsf.duplicated(subset=['permno', 'date']).sum()
    if n_dupes > 0:
        print(f"  ⚠ Removing {n_dupes:,} duplicate permno-date rows")
        dsf = dsf.drop_duplicates(subset=['permno', 'date'], keep='first')

    # Report basic stats
    print(f"\n  Final panel shape: {dsf.shape}")
    print(f"  Date range: {dsf['date'].min().date()} to {dsf['date'].max().date()}")
    print(f"  Unique permnos: {dsf['permno'].nunique():,}")
    print(f"  S&P 500 rows: {dsf['is_sp500'].sum():,} "
          f"({dsf['is_sp500'].mean():.1%} of total)")

    # Report missing values
    print(f"\n  Missing values:")
    for col in ['ret', 'prc', 'vol', 'shrout', 'mkt_cap']:
        n_miss = dsf[col].isna().sum()
        pct = n_miss / len(dsf)
        print(f"    {col:>10s}: {n_miss:>10,} ({pct:.2%})")

    # Save
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / "daily_panel.parquet"
    dsf.to_parquet(path, index=False)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"\n  Saved to {path} ({size_mb:.1f} MB)")

    return dsf


# =============================================================================
# Main pipeline
# =============================================================================

def build_daily_panel() -> pd.DataFrame:
    """
    Run the full cleaning pipeline:
      1. Load raw CRSP data
      2. Filter to common stocks on major exchanges
      3. Fix negative prices
      4. Compute market cap
      5. Merge delisting returns
      6. Add S&P 500 membership flag
      7. Assemble and save

    Returns the final daily panel DataFrame.
    """
    print("=" * 60)
    print("Building clean daily panel from raw CRSP data")
    print("=" * 60)

    # Step 1
    raw = load_raw_data()

    # Step 2
    dsf = filter_common_stocks(raw['dsf'], raw['names'])

    # Step 3
    dsf = fix_prices(dsf)

    # Step 4
    dsf = compute_market_cap(dsf)

    # Step 5
    dsf = merge_delisting_returns(dsf, raw['delist'])

    # Step 6
    dsf = add_sp500_flag(dsf, raw['sp500'])

    # Step 7
    dsf = assemble_and_save(dsf)

    print("\n" + "=" * 60)
    print("Daily panel build complete.")
    print("=" * 60)

    return dsf


if __name__ == "__main__":
    build_daily_panel()
