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
