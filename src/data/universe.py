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
