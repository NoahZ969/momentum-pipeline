# Stage 1: Data Infrastructure

> **Goal:** Acquire clean, point-in-time, survivorship-bias-free data from CRSP via WRDS, store it efficiently, and validate it against published benchmarks before any strategy code is written.
>
> **Estimated time:** 1-2 weeks part-time.

---

## Table of Contents

1. [WRDS Setup](#1-wrds-setup)
2. [Repository Structure](#2-repository-structure)
3. [Data Acquisition: CRSP Tables](#3-data-acquisition-crsp-tables)
4. [S&P 500 Historical Membership](#4-sp-500-historical-membership)
5. [Delisting Returns](#5-delisting-returns)
6. [Data Cleaning and Assembly](#6-data-cleaning-and-assembly)
7. [Storage Format](#7-storage-format)
8. [Sanity Checks](#8-sanity-checks)
9. [Ken French Benchmark Data](#9-ken-french-benchmark-data)
10. [Deliverables Checklist](#10-deliverables-checklist)

---

## 1. WRDS Setup

### 1.1 Create your WRDS account

If you don't already have credentials, go to [wrds-www.wharton.upenn.edu](https://wrds-www.wharton.upenn.edu/) and register with your university email. Your institution must have an active subscription. Registration is typically approved within 24 hours.

### 1.2 Install the WRDS Python library

```bash
pip install wrds
```

### 1.3 First-time authentication

Run this once to cache your credentials locally (stored in `~/.pgpass`):

```python
import wrds

db = wrds.Connection(wrds_username='your_username')
# You'll be prompted for your password on first run.
# After this, the connection is cached and automatic.
```

### 1.4 Verify access to CRSP

```python
# List available CRSP libraries
libraries = db.list_libraries()
crsp_libs = [lib for lib in libraries if 'crsp' in lib.lower()]
print(crsp_libs)
# You should see at least: 'crsp', 'crsp_a_stock', 'crsp_a_indexes', etc.

# List tables in the main CRSP stock library
tables = db.list_tables(library='crsp')
print(tables)
# Key tables: 'msf' (monthly stock file), 'dsf' (daily stock file),
# 'msenames' (name history), 'msedelist' (delisting), 'msp500list' (S&P 500 membership)
```

> ⚠️ If you don't see `crsp` in your library list, your university's subscription may not include CRSP. Contact your library's research data services.

---

## 2. Repository Structure

Initialize the project with this layout:

```
momentum-pipeline/
├── README.md                  # Stage 0 pre-registration (already written)
├── stage1_data.md             # This document
├── data/
│   ├── raw/                   # Raw CRSP pulls (parquet), never modified after download
│   ├── processed/             # Cleaned, merged, ready-to-use panels
│   └── external/              # Ken French factors, etc.
├── src/
│   ├── data/
│   │   ├── download_crsp.py   # WRDS queries
│   │   ├── clean.py           # Cleaning and assembly
│   │   ├── universe.py        # Point-in-time universe construction
│   │   └── sanity_checks.py   # All Stage 1 validation
│   ├── signal/                # (Stage 2)
│   ├── portfolio/             # (Stage 3)
│   ├── backtest/              # (Stage 4)
│   └── evaluation/            # (Stage 5)
├── notebooks/
│   └── 01_data_validation.ipynb   # Interactive sanity check plots
├── tests/
│   └── test_data.py           # Automated data quality assertions
└── config.py                  # Paths, date ranges, parameters
```

```bash
# Initialize
mkdir -p momentum-pipeline/{data/{raw,processed,external},src/{data,signal,portfolio,backtest,evaluation},notebooks,tests}
cd momentum-pipeline
git init
```

### `config.py` — central configuration

```python
"""
Central configuration. All parameters defined here;
no magic numbers scattered through the codebase.
"""
from pathlib import Path
from datetime import date

# === Paths ===
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"

# === Sample periods (Stage 0 pre-registration) ===
# We need data starting 252 trading days before the first rebalance date.
# First rebalance: 2005-01-31, so we need data from roughly 2003-12-01.
DATA_START_DATE = date(2003, 1, 1)   # generous buffer
DATA_END_DATE = date(2025, 12, 31)
INSAMPLE_START = date(2005, 1, 1)
INSAMPLE_END = date(2019, 12, 31)
HOLDOUT_START = date(2020, 1, 1)
HOLDOUT_END = date(2025, 12, 31)

# === Signal parameters (Stage 0 pre-registration) ===
LOOKBACK_DAYS = 252          # ~12 months
SKIP_DAYS = 21               # ~1 month (short-term reversal exclusion)
DECILE_LONG = 0.9            # top 10%
DECILE_SHORT = 0.1           # bottom 10%

# === Cost model (Stage 0 pre-registration) ===
HALF_SPREAD_BP = 2.0
COMMISSION_BP = 0.5
IMPACT_BP = 0.0
TOTAL_COST_BP = HALF_SPREAD_BP + COMMISSION_BP + IMPACT_BP

# === CRSP filters ===
# Share codes 10, 11 = common shares of domestic companies
# Exchange codes 1, 2, 3 = NYSE, AMEX, NASDAQ
VALID_SHARE_CODES = [10, 11]
VALID_EXCHANGE_CODES = [1, 2, 3]
```

---

## 3. Data Acquisition: CRSP Tables

You need three core pulls from CRSP. All queries are designed to download once and cache locally as Parquet files in `data/raw/`.

### 3.1 Daily Stock File (`crsp.dsf`)

This is the main price table. For ~500 stocks over ~22 years of trading days, expect roughly 2-3 million rows.

```python
# src/data/download_crsp.py

import wrds
import pandas as pd
from config import DATA_RAW, DATA_START_DATE, DATA_END_DATE

def download_daily_stock_file(db: wrds.Connection) -> pd.DataFrame:
    """
    Download CRSP daily stock file: prices, returns, volume, shares outstanding.
    
    Key columns:
      - permno: permanent security identifier (stable across ticker changes)
      - date: trading date
      - ret: holding-period return (includes dividends, split-adjusted)
      - prc: closing price (negative = bid/ask midpoint, take abs)
      - vol: trading volume (shares)
      - shrout: shares outstanding (thousands)
      - cfacpr: cumulative price adjustment factor
      - cfacshr: cumulative shares adjustment factor
    """
    query = f"""
        SELECT permno, date, ret, prc, vol, shrout, cfacpr, cfacshr
        FROM crsp.dsf
        WHERE date BETWEEN '{DATA_START_DATE}' AND '{DATA_END_DATE}'
    """
    df = db.raw_sql(query, date_cols=['date'])

    path = DATA_RAW / "crsp_dsf.parquet"
    df.to_parquet(path, index=False)
    print(f"Daily stock file: {len(df):,} rows -> {path}")
    return df
```

### 3.2 Stock Name History (`crsp.msenames`)

This gives you the share code, exchange code, ticker, and company name history for each `permno`. You need this to filter to common stocks on major exchanges.

```python
def download_name_history(db: wrds.Connection) -> pd.DataFrame:
    """
    Download CRSP name history for share type and exchange filtering.
    
    Key columns:
      - permno: permanent security identifier
      - namedt / nameendt: date range this name record is valid
      - shrcd: share code (10, 11 = common domestic)
      - exchcd: exchange code (1=NYSE, 2=AMEX, 3=NASDAQ)
      - ticker: ticker symbol
      - comnam: company name
    """
    query = """
        SELECT permno, namedt, nameendt, shrcd, exchcd, ticker, comnam
        FROM crsp.msenames
    """
    df = db.raw_sql(query, date_cols=['namedt', 'nameendt'])

    path = DATA_RAW / "crsp_msenames.parquet"
    df.to_parquet(path, index=False)
    print(f"Name history: {len(df):,} rows -> {path}")
    return df
```

### 3.3 Delisting Returns (`crsp.msedelist`)

Critical — this is where delisting returns live. Without this, delisted stocks vanish from your data and bias your backtest upward.

```python
def download_delisting_returns(db: wrds.Connection) -> pd.DataFrame:
    """
    Download CRSP delisting information.
    
    Key columns:
      - permno: permanent security identifier
      - dlstdt: delisting date
      - dlret: delisting return (the return on the delisting day)
      - dlstcd: delisting code (see CRSP docs for interpretation)
      - dlprc: delisting price
    """
    query = """
        SELECT permno, dlstdt, dlret, dlstcd, dlprc
        FROM crsp.msedelist
    """
    df = db.raw_sql(query, date_cols=['dlstdt'])

    path = DATA_RAW / "crsp_msedelist.parquet"
    df.to_parquet(path, index=False)
    print(f"Delisting returns: {len(df):,} rows -> {path}")
    return df
```

### 3.4 Run all downloads

```python
def download_all():
    """Download all CRSP tables. Run once, then work from local Parquet files."""
    db = wrds.Connection()
    try:
        download_daily_stock_file(db)
        download_name_history(db)
        download_delisting_returns(db)
        download_sp500_membership(db)  # defined in Section 4
    finally:
        db.close()

if __name__ == "__main__":
    download_all()
```

---

## 4. S&P 500 Historical Membership

CRSP provides a table of historical S&P 500 membership with start and end dates for each `permno`.

```python
def download_sp500_membership(db: wrds.Connection) -> pd.DataFrame:
    """
    Download S&P 500 historical membership from CRSP.
    
    Key columns:
      - permno: permanent security identifier
      - start: date the stock entered the S&P 500
      - ending: date the stock left the S&P 500 (NaT if still a member)
    """
    query = """
        SELECT *
        FROM crsp.msp500list
    """
    df = db.raw_sql(query, date_cols=['start', 'ending'])

    path = DATA_RAW / "crsp_sp500_membership.parquet"
    df.to_parquet(path, index=False)
    print(f"S&P 500 membership: {len(df):,} rows -> {path}")
    return df
```

### Building a point-in-time membership function

```python
# src/data/universe.py

import pandas as pd
from config import DATA_RAW

def load_sp500_membership() -> pd.DataFrame:
    """Load and clean S&P 500 membership table."""
    df = pd.read_parquet(DATA_RAW / "crsp_sp500_membership.parquet")
    # Fill missing end dates with a far-future date (still a member)
    df['ending'] = df['ending'].fillna(pd.Timestamp('2099-12-31'))
    return df

def get_sp500_members(date: pd.Timestamp, membership: pd.DataFrame) -> set:
    """
    Return the set of permnos that were in the S&P 500 on a given date.
    
    This is the point-in-time universe: only uses information available
    as of `date`. A stock is a member if start <= date <= ending.
    """
    mask = (membership['start'] <= date) & (membership['ending'] >= date)
    return set(membership.loc[mask, 'permno'])
```

---

## 5. Delisting Returns

Delisting returns must be incorporated into the daily return series. The standard approach (Shumway 1997, Beaver, McNichols & Price 2007):

```python
# src/data/clean.py  (partial — delisting integration)

def merge_delisting_returns(dsf: pd.DataFrame, delist: pd.DataFrame) -> pd.DataFrame:
    """
    Incorporate delisting returns into the daily return series.
    
    Logic:
    1. If a stock has a delisting return (dlret) on its delisting date,
       compound it with the regular return on that date.
    2. If the delisting date has no regular return (stock stopped trading),
       use the delisting return as the return for that date.
    3. If dlret is missing for a performance-related delisting (dlstcd 500-599),
       assume dlret = -0.30 (Shumway 1997 convention).
    """
    # Prepare delisting data
    dl = delist[['permno', 'dlstdt', 'dlret', 'dlstcd']].copy()
    dl = dl.rename(columns={'dlstdt': 'date'})

    # Shumway adjustment: missing dlret for performance delistings
    performance_delist = dl['dlstcd'].between(500, 599)
    dl.loc[performance_delist & dl['dlret'].isna(), 'dlret'] = -0.30

    # Merge onto daily file
    merged = dsf.merge(dl[['permno', 'date', 'dlret']], 
                       on=['permno', 'date'], how='left')

    # Compound delisting return with regular return
    # If ret exists: adjusted_ret = (1 + ret) * (1 + dlret) - 1
    # If ret is NaN but dlret exists: adjusted_ret = dlret
    has_dlret = merged['dlret'].notna()
    has_ret = merged['ret'].notna()
    
    merged.loc[has_dlret & has_ret, 'ret'] = (
        (1 + merged.loc[has_dlret & has_ret, 'ret']) *
        (1 + merged.loc[has_dlret & has_ret, 'dlret']) - 1
    )
    merged.loc[has_dlret & ~has_ret, 'ret'] = merged.loc[has_dlret & ~has_ret, 'dlret']

    merged = merged.drop(columns=['dlret'])
    return merged
```

---

## 6. Data Cleaning and Assembly

```python
# src/data/clean.py  (full assembly function)

import pandas as pd
import numpy as np
from config import DATA_RAW, DATA_PROCESSED, VALID_SHARE_CODES, VALID_EXCHANGE_CODES

def build_clean_daily_panel() -> pd.DataFrame:
    """
    Full data assembly pipeline: load raw CRSP, filter, merge delistings,
    compute adjusted prices, and save a clean daily panel.
    
    Output columns:
      - permno, date, ret (with delisting returns incorporated)
      - prc (absolute value), vol, mkt_cap (price * shares outstanding)
      - is_sp500 (boolean: was this permno in the S&P 500 on this date?)
    """
    # --- Load raw data ---
    dsf = pd.read_parquet(DATA_RAW / "crsp_dsf.parquet")
    names = pd.read_parquet(DATA_RAW / "crsp_msenames.parquet")
    delist = pd.read_parquet(DATA_RAW / "crsp_msedelist.parquet")
    sp500 = pd.read_parquet(DATA_RAW / "crsp_sp500_membership.parquet")

    # --- Filter to common stocks on major exchanges (point-in-time) ---
    # For each permno-date pair, find the applicable name record
    # and check that shrcd and exchcd are valid.
    valid_names = names[
        names['shrcd'].isin(VALID_SHARE_CODES) & 
        names['exchcd'].isin(VALID_EXCHANGE_CODES)
    ][['permno', 'namedt', 'nameendt']].copy()

    # Merge: keep only dsf rows where the permno had valid share/exchange
    # codes on that date
    dsf = dsf.merge(valid_names, on='permno', how='inner')
    dsf = dsf[(dsf['date'] >= dsf['namedt']) & (dsf['date'] <= dsf['nameendt'])]
    dsf = dsf.drop(columns=['namedt', 'nameendt'])

    # --- Absolute price (CRSP uses negative to indicate bid-ask average) ---
    dsf['prc'] = dsf['prc'].abs()

    # --- Market cap (in millions) ---
    # shrout is in thousands, prc is in dollars
    dsf['mkt_cap'] = dsf['prc'] * dsf['shrout'] / 1000.0

    # --- Merge delisting returns ---
    dsf = merge_delisting_returns(dsf, delist)

    # --- Add S&P 500 membership flag ---
    sp500['ending'] = sp500['ending'].fillna(pd.Timestamp('2099-12-31'))
    
    # Build an interval index for efficient lookup
    # (For each permno, we store membership intervals)
    sp500_intervals = []
    for _, row in sp500.iterrows():
        sp500_intervals.append({
            'permno': row['permno'],
            'sp_start': row['start'],
            'sp_end': row['ending']
        })
    sp500_df = pd.DataFrame(sp500_intervals)

    # Merge and flag — an asof merge or conditional join
    dsf = dsf.merge(sp500_df, on='permno', how='left')
    dsf['is_sp500'] = (dsf['date'] >= dsf['sp_start']) & (dsf['date'] <= dsf['sp_end'])
    # A permno can have multiple S&P 500 stints; collapse to one row per permno-date
    dsf = dsf.groupby(['permno', 'date']).agg({
        'ret': 'first',
        'prc': 'first',
        'vol': 'first',
        'shrout': 'first',
        'mkt_cap': 'first',
        'cfacpr': 'first',
        'cfacshr': 'first',
        'is_sp500': 'any'  # True if member in ANY stint covering this date
    }).reset_index()

    # --- Sort and save ---
    dsf = dsf.sort_values(['permno', 'date']).reset_index(drop=True)
    
    path = DATA_PROCESSED / "daily_panel.parquet"
    dsf.to_parquet(path, index=False)
    print(f"Clean daily panel: {len(dsf):,} rows, "
          f"{dsf['permno'].nunique():,} unique permnos -> {path}")
    return dsf
```

> **A subtle but important note on the S&P 500 merge above:** The naive merge shown here works but is slow for large data. A more efficient approach uses `pd.merge_asof` or an interval tree. For S&P 500 (~1,000 membership stints over 20 years), the naive approach runs in seconds to minutes and is fine. If you later extend to Russell 3000, you'll want to optimize.

---

## 7. Storage Format

**Use Parquet throughout.** It's columnar, compressed, preserves dtypes, and Polars/Pandas read it natively.

Storage layout:

```
data/
├── raw/                        # Never modify these after download
│   ├── crsp_dsf.parquet        # ~2-3M rows, ~150 MB
│   ├── crsp_msenames.parquet   # ~100K rows, <5 MB
│   ├── crsp_msedelist.parquet  # ~30K rows, <2 MB
│   └── crsp_sp500_membership.parquet  # ~1.5K rows, <1 MB
├── processed/
│   └── daily_panel.parquet     # Cleaned, merged, ready to use
└── external/
    └── ff_factors.parquet      # Ken French factors (Section 9)
```

**Rule:** `data/raw/` is immutable. If you need to reprocess, always regenerate `data/processed/` from `data/raw/` using the code in `src/data/`. This makes the pipeline reproducible: delete `processed/`, rerun `clean.py`, and get the same result.

Add to `.gitignore`:

```
data/raw/
data/processed/
data/external/
```

Data files are too large for git. The code to regenerate them _is_ version-controlled.

---

## 8. Sanity Checks

These are the five checks from the Stage 0 pre-registration, Section 9. **All must pass before proceeding to Stage 2.** Implement them in `src/data/sanity_checks.py` and visualize interactively in `notebooks/01_data_validation.ipynb`.

### Check 1: Universe count over time

```python
def check_universe_count(panel: pd.DataFrame):
    """
    Plot the number of S&P 500 stocks in the universe at each month-end.
    Expected: ~500, with normal turnover of ~20-25 changes per year.
    
    Red flags:
      - Flat line at exactly 500 → broken point-in-time logic
      - Monotonically increasing → survivorship bias (adding stocks, never removing)
      - Count dropping below 400 or above 600 → data issue
    """
    # Get month-end dates
    sp500 = panel[panel['is_sp500']].copy()
    sp500['month_end'] = sp500['date'] + pd.offsets.MonthEnd(0)
    
    # Count unique permnos per month-end
    counts = (sp500.groupby('month_end')['permno']
              .nunique()
              .rename('n_stocks'))
    
    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 5))
    counts.plot(ax=ax)
    ax.axhline(y=500, color='red', linestyle='--', alpha=0.5, label='Expected ~500')
    ax.set_ylabel('Number of S&P 500 constituents')
    ax.set_title('Sanity Check 1: Universe Count Over Time')
    ax.legend()
    plt.tight_layout()
    plt.savefig('notebooks/check1_universe_count.png', dpi=150)
    plt.show()
    
    # Automated assertion
    assert counts.min() > 400, f"Universe too small: min = {counts.min()}"
    assert counts.max() < 600, f"Universe too large: max = {counts.max()}"
    print("✓ Check 1 passed: universe count within [400, 600]")
```

### Check 2: Cap-weighted index reproduction

```python
def check_index_reproduction(panel: pd.DataFrame):
    """
    Compute the cap-weighted total return of S&P 500 members
    and compare to the published S&P 500 Total Return Index.
    
    Tracking error should be < 50 bp/year annualized.
    
    Benchmark data: Download S&P 500 TR from CRSP index file
    or from Ken French market factor (Rm).
    """
    sp500 = panel[panel['is_sp500'] & panel['ret'].notna()].copy()
    
    # Compute cap-weighted daily return
    # Weight by previous day's market cap (to avoid look-ahead)
    sp500 = sp500.sort_values(['permno', 'date'])
    sp500['lag_mkt_cap'] = sp500.groupby('permno')['mkt_cap'].shift(1)
    
    daily_ret = (sp500.groupby('date')
                 .apply(lambda g: np.average(g['ret'], weights=g['lag_mkt_cap'].fillna(g['mkt_cap'])))
                 .rename('portfolio_ret'))
    
    # Compare to Ken French market return (Rm + Rf)
    # (loaded in Section 9)
    # Compute annualized tracking error
    # tracking_error = (daily_ret - benchmark_ret).std() * np.sqrt(252)
    
    print("Check 2: compute tracking error vs published S&P 500 TR")
    print("Target: < 50 bp/year annualized")
    return daily_ret
```

### Check 3: Equal-weighted index reproduction

```python
def check_equal_weighted_index(panel: pd.DataFrame):
    """
    Compute the equal-weighted total return of S&P 500 members
    and compare to the S&P 500 Equal Weight Index.
    
    Tracking error should be < 100 bp/year.
    """
    sp500 = panel[panel['is_sp500'] & panel['ret'].notna()].copy()
    
    # Equal-weighted daily return: simple average across all members
    ew_daily_ret = sp500.groupby('date')['ret'].mean().rename('ew_ret')
    
    print("Check 3: equal-weighted index")
    print(f"Annualized return: {ew_daily_ret.mean() * 252:.4f}")
    print(f"Annualized vol: {ew_daily_ret.std() * np.sqrt(252):.4f}")
    return ew_daily_ret
```

### Check 4: Delisting accounting

```python
def check_delistings(panel: pd.DataFrame, delist: pd.DataFrame):
    """
    Verify that known delistings are correctly reflected in the data.
    
    Check at least three known cases:
      1. Lehman Brothers (PERMNO 59408) — delisted Sept 2008
      2. General Motors old (PERMNO 12079) — delisted June 2009
      3. Any recent acquisition — verify the last return reflects deal price
    """
    test_cases = {
        'Lehman Brothers': {'permno': 59408, 'approx_date': '2008-09'},
        'General Motors (old)': {'permno': 12079, 'approx_date': '2009-06'},
    }
    
    for name, info in test_cases.items():
        stock = panel[panel['permno'] == info['permno']].sort_values('date')
        last_row = stock.iloc[-1]
        
        # Check that the stock has a last date near the expected delisting
        assert info['approx_date'] in str(last_row['date']), \
            f"{name}: expected delisting around {info['approx_date']}, got {last_row['date']}"
        
        # Check that the last return is not NaN (delisting return should be present)
        assert pd.notna(last_row['ret']), \
            f"{name}: last return is NaN — delisting return not incorporated"
        
        # Check against CRSP delisting table
        dl_row = delist[delist['permno'] == info['permno']]
        if not dl_row.empty:
            print(f"✓ {name}: last date = {last_row['date'].date()}, "
                  f"last ret = {last_row['ret']:.4f}, "
                  f"dlret = {dl_row.iloc[0]['dlret']}, "
                  f"dlstcd = {dl_row.iloc[0]['dlstcd']}")
        else:
            print(f"⚠ {name}: not found in delisting table — investigate")
```

> **Note on PERMNOs:** The PERMNO values above are commonly cited but you should verify them in the CRSP name history table. Search by ticker/company name if needed: `names[names['comnam'].str.contains('LEHMAN', case=False)]`.

### Check 5: No look-ahead in signal

This check is implemented in Stage 2 when the signal function is written. The assertion is:

```python
def compute_signal(panel: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.Series:
    """Compute MOM_12_1 signal using only data available on as_of_date."""
    # LOOK-AHEAD GUARD
    assert panel['date'].max() <= as_of_date, \
        f"Look-ahead detected: data goes to {panel['date'].max()}, but as_of_date is {as_of_date}"
    # ... signal computation ...
```

A unit test will deliberately pass future data and confirm the assertion fires.

---

## 9. Ken French Benchmark Data

Download the Fama-French factors and the UMD (momentum) factor for benchmarking. This data is free and does not require WRDS.

```python
# src/data/download_crsp.py  (append to existing file)

import pandas_datareader.data as web

def download_french_factors():
    """
    Download Fama-French 3 factors + momentum (UMD) from Ken French's data library.
    
    Returns daily and monthly factor returns:
      - Mkt-RF: market excess return
      - SMB: size factor
      - HML: value factor
      - RF: risk-free rate
      - Mom/UMD: momentum factor (from separate dataset)
    """
    # Daily Fama-French 3 factors
    ff3_daily = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench',
                               start='2003-01-01')[0]
    ff3_daily = ff3_daily / 100  # Convert from percent to decimal

    # Daily momentum factor
    mom_daily = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench',
                               start='2003-01-01')[0]
    mom_daily = mom_daily / 100
    mom_daily = mom_daily.rename(columns={mom_daily.columns[0]: 'UMD'})

    # Merge
    factors = ff3_daily.join(mom_daily, how='inner')
    factors.index.name = 'date'
    factors = factors.reset_index()
    factors['date'] = pd.to_datetime(factors['date'])

    path = DATA_EXTERNAL / "ff_factors_daily.parquet"
    factors.to_parquet(path, index=False)
    print(f"French factors (daily): {len(factors):,} rows -> {path}")

    # Monthly — same process, for monthly evaluation later
    ff3_monthly = web.DataReader('F-F_Research_Data_Factors', 'famafrench',
                                 start='2003-01-01')[0]
    ff3_monthly = ff3_monthly / 100
    mom_monthly = web.DataReader('F-F_Momentum_Factor', 'famafrench',
                                 start='2003-01-01')[0]
    mom_monthly = mom_monthly / 100
    mom_monthly = mom_monthly.rename(columns={mom_monthly.columns[0]: 'UMD'})
    factors_m = ff3_monthly.join(mom_monthly, how='inner')
    factors_m.index.name = 'date'
    factors_m = factors_m.reset_index()

    path_m = DATA_EXTERNAL / "ff_factors_monthly.parquet"
    factors_m.to_parquet(path_m, index=False)
    print(f"French factors (monthly): {len(factors_m):,} rows -> {path_m}")
    
    return factors
```

> **Install dependency:** `pip install pandas-datareader`

---

## 10. Deliverables Checklist

Before moving to Stage 2, confirm every item:

- [ ] WRDS connection working; CRSP tables accessible
- [ ] Raw data downloaded and saved in `data/raw/` as Parquet files
- [ ] `crsp_dsf.parquet` — daily prices and returns
- [ ] `crsp_msenames.parquet` — share type and exchange history
- [ ] `crsp_msedelist.parquet` — delisting returns
- [ ] `crsp_sp500_membership.parquet` — S&P 500 historical membership
- [ ] Delisting returns merged into daily return series (Shumway adjustment applied)
- [ ] Clean daily panel saved in `data/processed/daily_panel.parquet`
- [ ] S&P 500 point-in-time membership flag (`is_sp500`) in daily panel
- [ ] Ken French factors downloaded to `data/external/`
- [ ] **Sanity check 1 passed:** universe count is ~500, no flat line
- [ ] **Sanity check 2 passed:** cap-weighted return tracks S&P 500 TR within 50 bp/year
- [ ] **Sanity check 3 passed:** equal-weighted return is reasonable
- [ ] **Sanity check 4 passed:** Lehman, GM, and one other delisting correctly reflected
- [ ] All code committed to git
- [ ] `data/` directories in `.gitignore`
- [ ] `config.py` matches Stage 0 pre-registration parameters exactly

> **Only proceed to Stage 2 when all boxes are checked.**

---

## Appendix A: Common WRDS Gotchas

1. **CRSP returns are already adjusted for splits and dividends.** The `ret` column in `dsf` is a total return including distributions. You do not need to manually adjust. But be aware: `prc` is the raw closing price, not adjusted. Use `ret` for return calculations, not price changes.

2. **Negative prices in CRSP.** When the closing price is missing, CRSP uses the negative of the bid-ask midpoint. Always take `abs(prc)` before using prices.

3. **PERMNO vs PERMCO vs CUSIP vs ticker.** Use `permno` as your primary identifier. It is stable across ticker changes, name changes, and even some structural changes. Ticker is not stable. CUSIP changes at corporate events. PERMCO groups multiple securities of the same company (e.g., class A and class B shares) — you usually want `permno` level.

4. **WRDS query timeouts.** If the daily stock file query times out (unlikely but possible for the full history), add a date filter to download in chunks (e.g., 5 years at a time) and concatenate locally.

5. **`pandas-datareader` version.** If `DataReader('F-F_Research_Data_Factors', 'famafrench')` fails, try `pip install pandas-datareader --upgrade`. The French data interface changes occasionally.
