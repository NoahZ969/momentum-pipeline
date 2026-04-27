"""
download_crsp.py — Download all required CRSP tables and external benchmark data.

Run once to populate data/raw/ and data/external/ with Parquet files.
All subsequent pipeline stages work from these local files.

Usage:
    python -m src.data.download_crsp

Prerequisites:
    pip install wrds pandas pandas-datareader pyarrow lxml requests
    
    On first run, you'll be prompted for your WRDS username and password.
    Credentials are cached in ~/.pgpass for subsequent runs.
"""

import wrds
import pandas as pd

import sys
from pathlib import Path

# Add project root to path so we can import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_RAW, DATA_EXTERNAL, DATA_START_DATE, DATA_END_DATE


# =============================================================================
# CRSP Downloads
# =============================================================================

def download_daily_stock_file(db: wrds.Connection) -> pd.DataFrame:
    """
    Download CRSP daily stock file: prices, returns, volume, shares outstanding.

    Key columns:
      - permno:  permanent security identifier (stable across ticker changes)
      - date:    trading date
      - ret:     holding-period return (includes dividends, split-adjusted)
      - prc:     closing price (negative = bid/ask midpoint; take abs)
      - vol:     trading volume (shares)
      - shrout:  shares outstanding (thousands)
      - cfacpr:  cumulative price adjustment factor
      - cfacshr: cumulative shares adjustment factor

    For ~500 stocks over ~22 years of trading days, expect ~2-3 million rows.
    """
    print("Downloading CRSP daily stock file (dsf)...")
    query = f"""
        SELECT permno, date, ret, prc, vol, shrout, cfacpr, cfacshr
        FROM crsp.dsf
        WHERE date BETWEEN '{DATA_START_DATE}' AND '{DATA_END_DATE}'
    """
    df = db.raw_sql(query, date_cols=['date'])

    path = DATA_RAW / "crsp_dsf.parquet"
    df.to_parquet(path, index=False)
    print(f"  → {len(df):,} rows, {df['permno'].nunique():,} unique permnos -> {path}")
    return df


def download_name_history(db: wrds.Connection) -> pd.DataFrame:
    """
    Download CRSP stock name history for share type and exchange filtering.

    Key columns:
      - permno:   permanent security identifier
      - namedt:   start date this name record is valid
      - nameendt: end date this name record is valid
      - shrcd:    share code (10, 11 = common domestic stocks)
      - exchcd:   exchange code (1=NYSE, 2=AMEX, 3=NASDAQ)
      - ticker:   ticker symbol
      - comnam:   company name
    """
    print("Downloading CRSP name history (msenames)...")
    query = """
        SELECT permno, namedt, nameendt, shrcd, exchcd, ticker, comnam
        FROM crsp.msenames
    """
    df = db.raw_sql(query, date_cols=['namedt', 'nameendt'])

    path = DATA_RAW / "crsp_msenames.parquet"
    df.to_parquet(path, index=False)
    print(f"  → {len(df):,} rows -> {path}")
    return df


def download_delisting_returns(db: wrds.Connection) -> pd.DataFrame:
    """
    Download CRSP delisting information.

    Key columns:
      - permno: permanent security identifier
      - dlstdt: delisting date
      - dlret:  delisting return (the return on the delisting day)
      - dlstcd: delisting code (500-599 = performance-related)
      - dlprc:  delisting price
    """
    print("Downloading CRSP delisting returns (msedelist)...")
    query = """
        SELECT permno, dlstdt, dlret, dlstcd, dlprc
        FROM crsp.msedelist
    """
    df = db.raw_sql(query, date_cols=['dlstdt'])

    path = DATA_RAW / "crsp_msedelist.parquet"
    df.to_parquet(path, index=False)
    print(f"  → {len(df):,} rows -> {path}")
    return df


def download_sp500_membership(db=None) -> pd.DataFrame:
    """
    Build S&P 500 historical membership by combining:
      1. Point-in-time constituent snapshots from fja05680/sp500 on GitHub
         (maintained dataset with S&P 500 membership since 1996)
      2. CRSP name history to map tickers -> permnos

    Falls back to this approach because crsp.msp500list requires the
    crsp_a_indexes subscription, which many universities don't have.

    The GitHub dataset has rows of (date, comma-separated tickers) — a full
    snapshot of the S&P 500 on each change date. This is far more reliable
    than trying to parse Wikipedia's changes table.

    Output columns (matches the format clean.py expects):
      - permno:  permanent security identifier
      - start:   date the stock entered the S&P 500
      - ending:  date the stock left the S&P 500 (NaT if still a member)
    """
    print("Building S&P 500 historical membership from GitHub + CRSP name history...")

    import requests
    import io

    # ------------------------------------------------------------------
    # Step 1: Download point-in-time constituent snapshots from GitHub
    # ------------------------------------------------------------------
    print("  Downloading S&P 500 historical constituents from GitHub (fja05680/sp500)...")

    # Try the most recent dated file first, fall back to the base file
    urls_to_try = [
        "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes(07-12-2025).csv",
        "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes(08-01-2024).csv",
        "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes.csv",
    ]

    headers = {'User-Agent': 'momentum-pipeline/1.0 (academic research project)'}
    pit_df = None
    for url in urls_to_try:
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                pit_df = pd.read_csv(io.StringIO(resp.text))
                print(f"    Downloaded from: {url.split('/')[-1]}")
                break
        except Exception:
            continue

    if pit_df is None:
        raise RuntimeError(
            "Could not download S&P 500 historical data from GitHub. "
            "Check your internet connection or visit "
            "https://github.com/fja05680/sp500 to download manually."
        )

    # The CSV has columns: date, tickers (comma-separated string)
    # Standardize column names
    pit_df.columns = [c.strip().lower() for c in pit_df.columns]
    date_col = [c for c in pit_df.columns if 'date' in c][0]
    ticker_col = [c for c in pit_df.columns if 'ticker' in c or 'symbol' in c or c == 'tickers'][0]

    pit_df[date_col] = pd.to_datetime(pit_df[date_col])
    pit_df = pit_df.sort_values(date_col).reset_index(drop=True)

    print(f"    {len(pit_df)} snapshots from {pit_df[date_col].min().date()} "
          f"to {pit_df[date_col].max().date()}")

    # ------------------------------------------------------------------
    # Step 2: Convert snapshots to (ticker, start, ending) intervals
    # ------------------------------------------------------------------
    print("  Converting snapshots to membership intervals...")

    # For each snapshot date, get the set of tickers
    snapshots = []
    for _, row in pit_df.iterrows():
        dt = row[date_col]
        tickers_str = str(row[ticker_col])
        tickers = set()
        for t in tickers_str.split(','):
            t = t.strip()
            if t and t.lower() != 'nan':
                tickers.add(t)
        snapshots.append((dt, tickers))

    # Walk through snapshots chronologically to build intervals
    all_intervals = []
    active = {}  # ticker -> start_date

    for i, (dt, tickers) in enumerate(snapshots):
        # New additions: in current snapshot but not in active
        for t in tickers:
            if t not in active:
                active[t] = dt

        # Removals: in active but not in current snapshot
        removed = set(active.keys()) - tickers
        for t in removed:
            all_intervals.append({
                'ticker': t,
                'start': active[t],
                'ending': dt,
            })
            del active[t]

    # Remaining active tickers are still in the S&P 500
    for t, start_dt in active.items():
        all_intervals.append({
            'ticker': t,
            'start': start_dt,
            'ending': pd.NaT,
        })

    intervals_df = pd.DataFrame(all_intervals)

    print(f"    Built {len(intervals_df)} membership intervals "
          f"for {intervals_df['ticker'].nunique()} unique tickers")

    # Quick check: how many tickers are active (no ending) = current members
    n_current = intervals_df['ending'].isna().sum()
    print(f"    Current members (no end date): {n_current}")

    # Diagnostic: print a few sample tickers to verify format
    sample = intervals_df['ticker'].head(10).tolist()
    print(f"    Sample tickers (raw): {sample}")

    # The GitHub CSV disambiguates tickers that appear multiple times
    # by appending a date suffix: e.g., "CCB-199602", "AAL-199702".
    # Strip these suffixes to get the actual ticker symbol.
    # Pattern: ticker ends with -YYYYMM (6 digits after hyphen)
    import re

    def strip_date_suffix(ticker):
        """Remove trailing -YYYYMM suffix if present."""
        return re.sub(r'-\d{6}$', '', ticker)

    intervals_df['ticker'] = intervals_df['ticker'].apply(strip_date_suffix)

    sample_clean = intervals_df['ticker'].head(10).tolist()
    print(f"    Sample tickers (cleaned): {sample_clean}")

    # ------------------------------------------------------------------
    # Step 3: Map tickers to CRSP permnos
    # ------------------------------------------------------------------
    print("  Mapping tickers to CRSP permnos via name history...")

    names = pd.read_parquet(DATA_RAW / "crsp_msenames.parquet")
    ticker_permno = names[['permno', 'ticker', 'namedt', 'nameendt']].copy()

    # Normalize both sides: strip whitespace, uppercase
    ticker_permno['ticker_clean'] = (
        ticker_permno['ticker'].str.strip().str.upper()
    )
    intervals_df['ticker_clean'] = (
        intervals_df['ticker'].str.strip().str.upper()
    )

    # Build a lookup dict: crsp_ticker -> list of (permno, namedt, nameendt)
    from collections import defaultdict
    crsp_lookup = defaultdict(list)
    for _, row in ticker_permno.iterrows():
        crsp_lookup[row['ticker_clean']].append({
            'permno': row['permno'],
            'namedt': row['namedt'],
            'nameendt': row['nameendt'],
        })

    # Also build variants for fuzzy matching:
    # Some tickers differ between Yahoo/Wikipedia and CRSP conventions:
    #   BRK.B (Yahoo) -> BRK-B or BRKB (CRSP)
    #   BF.B  (Yahoo) -> BF-B  or BFB  (CRSP)
    def ticker_variants(ticker):
        """Generate plausible CRSP ticker variants for a given ticker."""
        variants = [ticker]
        # Dot variants: BRK.B -> BRKB, BRK-B, BRK B
        if '.' in ticker:
            variants.append(ticker.replace('.', ''))
            variants.append(ticker.replace('.', '-'))
            variants.append(ticker.replace('.', ' '))
            # Strip share class suffix entirely: BRK.B -> BRK, BF.B -> BF
            # CRSP often stores both classes under the base ticker
            base = ticker.split('.')[0]
            if base not in variants:
                variants.append(base)
        # Hyphen variants: BRK-B -> BRKB, BRK B, BRK
        if '-' in ticker:
            variants.append(ticker.replace('-', ''))
            variants.append(ticker.replace('-', ' '))
            base = ticker.split('-')[0]
            if base not in variants:
                variants.append(base)
        # Space variants: BRK B -> BRKB
        if ' ' in ticker:
            variants.append(ticker.replace(' ', ''))
        # Try without trailing Q (bankruptcy ticker): AAMRQ -> AAMR
        if ticker.endswith('Q') and len(ticker) > 2:
            base = ticker[:-1]
            variants.append(base)
        return variants

    # Match each interval to a permno
    matched_intervals = []
    unmatched_tickers = set()

    for _, irow in intervals_df.iterrows():
        ticker = irow['ticker_clean']
        start = irow['start']
        ending = irow['ending']
        ending_safe = ending if pd.notna(ending) else pd.Timestamp('2099-12-31')

        best_match = None
        best_overlap = -1

        # Try each variant of the ticker
        for variant in ticker_variants(ticker):
            if variant not in crsp_lookup:
                continue
            for crsp_rec in crsp_lookup[variant]:
                # Check temporal overlap
                if crsp_rec['nameendt'] < start or crsp_rec['namedt'] > ending_safe:
                    continue
                # Compute overlap duration
                ov_start = max(start, crsp_rec['namedt'])
                ov_end = min(ending_safe, crsp_rec['nameendt'])
                ov_days = (ov_end - ov_start).days
                if ov_days > best_overlap:
                    best_overlap = ov_days
                    best_match = crsp_rec['permno']

        if best_match is not None:
            matched_intervals.append({
                'permno': int(best_match),
                'start': start,
                'ending': ending,
            })
        else:
            unmatched_tickers.add(irow['ticker'])

    result = pd.DataFrame(matched_intervals)

    n_unmatched = len(intervals_df) - len(result)
    if n_unmatched > 0:
        print(f"    ⚠ {n_unmatched} intervals could not be mapped to a permno "
              f"({len(unmatched_tickers)} unique tickers).")
        print(f"      Unmatched tickers (sample): "
              f"{sorted(unmatched_tickers)[:15]}")
    else:
        print(f"    All intervals mapped successfully.")

    path = DATA_RAW / "crsp_sp500_membership.parquet"
    result.to_parquet(path, index=False)
    print(f"  → {len(result):,} membership intervals, "
          f"{result['permno'].nunique()} unique permnos -> {path}")
    return result


# =============================================================================
# External Benchmark Data (Ken French)
# =============================================================================

def download_french_factors() -> pd.DataFrame:
    """
    Download Fama-French 3 factors + momentum (UMD) from Ken French's data library.
    Free, no WRDS required.

    Daily factors:
      - Mkt-RF: market excess return
      - SMB:    size factor (small minus big)
      - HML:    value factor (high minus low)
      - RF:     risk-free rate
      - UMD:    momentum factor (up minus down)

    Monthly factors: same columns, monthly frequency.

    All values converted from percent to decimal (e.g., 1.5% -> 0.015).
    """
    import pandas_datareader.data as web

    # --- Daily factors ---
    print("Downloading Ken French daily factors...")
    ff3_daily = web.DataReader(
        'F-F_Research_Data_Factors_daily', 'famafrench', start='2003-01-01'
    )[0]
    ff3_daily = ff3_daily / 100  # percent -> decimal

    mom_daily = web.DataReader(
        'F-F_Momentum_Factor_daily', 'famafrench', start='2003-01-01'
    )[0]
    mom_daily = mom_daily / 100
    mom_daily = mom_daily.rename(columns={mom_daily.columns[0]: 'UMD'})

    factors_daily = ff3_daily.join(mom_daily, how='inner')
    factors_daily.index.name = 'date'
    factors_daily = factors_daily.reset_index()
    factors_daily['date'] = pd.to_datetime(factors_daily['date'])

    path_d = DATA_EXTERNAL / "ff_factors_daily.parquet"
    factors_daily.to_parquet(path_d, index=False)
    print(f"  → {len(factors_daily):,} rows -> {path_d}")

    # --- Monthly factors ---
    print("Downloading Ken French monthly factors...")
    ff3_monthly = web.DataReader(
        'F-F_Research_Data_Factors', 'famafrench', start='2003-01-01'
    )[0]
    ff3_monthly = ff3_monthly / 100

    mom_monthly = web.DataReader(
        'F-F_Momentum_Factor', 'famafrench', start='2003-01-01'
    )[0]
    mom_monthly = mom_monthly / 100
    mom_monthly = mom_monthly.rename(columns={mom_monthly.columns[0]: 'UMD'})

    factors_monthly = ff3_monthly.join(mom_monthly, how='inner')
    factors_monthly.index.name = 'date'
    factors_monthly = factors_monthly.reset_index()

    path_m = DATA_EXTERNAL / "ff_factors_monthly.parquet"
    factors_monthly.to_parquet(path_m, index=False)
    print(f"  → {len(factors_monthly):,} rows -> {path_m}")

    return factors_daily


# =============================================================================
# Main entry point
# =============================================================================

def download_all():
    """
    Download all required data. Run once, then work from local Parquet files.

    Creates:
      data/raw/crsp_dsf.parquet
      data/raw/crsp_msenames.parquet
      data/raw/crsp_msedelist.parquet
      data/raw/crsp_sp500_membership.parquet
      data/external/ff_factors_daily.parquet
      data/external/ff_factors_monthly.parquet
    """
    # Ensure output directories exist
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)

    # --- CRSP (requires WRDS connection) ---
    print("=" * 60)
    print("Connecting to WRDS...")
    print("=" * 60)
    db = wrds.Connection()
    try:
        download_daily_stock_file(db)
        download_name_history(db)
        download_delisting_returns(db)
    finally:
        db.close()
        print("\nWRDS connection closed.")

    # --- S&P 500 membership (Wikipedia + local CRSP name history) ---
    print("\n" + "=" * 60)
    print("Building S&P 500 membership...")
    print("=" * 60)
    download_sp500_membership(db=None)

    # --- External data (no WRDS needed) ---
    print("\n" + "=" * 60)
    print("Downloading external benchmark data...")
    print("=" * 60)
    download_french_factors()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("All downloads complete. Files saved:")
    print("=" * 60)
    for directory in [DATA_RAW, DATA_EXTERNAL]:
        for f in sorted(directory.glob("*.parquet")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(PROJECT_ROOT)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    download_all()
