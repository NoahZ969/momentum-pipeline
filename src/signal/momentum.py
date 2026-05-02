"""
momentum.py — Compute the MOM_12_1 cross-sectional momentum signal.

The signal for stock i on rebalance date T is the cumulative log return
from T-252 to T-21 trading days, cross-sectionally z-scored.

Usage:
    python -m src.signal.momentum
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_PROCESSED, DATA_RAW,
    LOOKBACK_DAYS, SKIP_DAYS,
    INSAMPLE_START, HOLDOUT_END,
)


def get_rebalance_dates(panel: pd.DataFrame,
                        start: pd.Timestamp,
                        end: pd.Timestamp) -> list:
    """
    Get the last trading day of each calendar month within [start, end].
    These are the dates on which we compute the signal and rebalance.
    """
    trading_dates = panel['date'].sort_values().unique()
    trading_dates = pd.DatetimeIndex(trading_dates)
    mask = (trading_dates >= start) & (trading_dates <= end)
    trading_dates = trading_dates[mask]

    # Group by year-month, take the last date in each group
    df = pd.DataFrame({'date': trading_dates})
    df['ym'] = df['date'].dt.to_period('M')
    rebal = df.groupby('ym')['date'].max().sort_values().tolist()
    return rebal


def compute_signal_on_date(panel: pd.DataFrame,
                           rebal_date: pd.Timestamp,
                           sp500_permnos: set) -> pd.DataFrame:
    """
    Compute MOM_12_1 for all eligible stocks on a single rebalance date.

    Parameters
    ----------
    panel : pd.DataFrame
        The full daily panel (permno, date, ret, is_sp500, ...).
        MUST be filtered to dates <= rebal_date before calling.
    rebal_date : pd.Timestamp
        The rebalance date T.
    sp500_permnos : set
        Set of permnos in the S&P 500 on rebal_date.

    Returns
    -------
    pd.DataFrame with columns: permno, date, raw_signal, z_signal
    """
    # ---- LOOK-AHEAD GUARD ----
    assert panel['date'].max() <= rebal_date, (
        f"Look-ahead detected: panel data goes to {panel['date'].max()}, "
        f"but rebal_date is {rebal_date}"
    )

    # Get the trading dates in the panel, sorted
    all_dates = sorted(panel['date'].unique())

    # Find the index of rebal_date (T)
    # T must be in the panel
    if rebal_date not in all_dates:
        return pd.DataFrame(columns=['permno', 'date', 'raw_signal', 'z_signal'])

    t_idx = all_dates.index(rebal_date)

    # We need at least LOOKBACK_DAYS of history before T
    if t_idx < LOOKBACK_DAYS:
        return pd.DataFrame(columns=['permno', 'date', 'raw_signal', 'z_signal'])

    # Define the signal window:
    #   start of lookback: T - LOOKBACK_DAYS
    #   end of lookback:   T - SKIP_DAYS
    date_lookback_start = all_dates[t_idx - LOOKBACK_DAYS]
    date_skip_end = all_dates[t_idx - SKIP_DAYS]

    # Filter to S&P 500 members
    eligible = panel[panel['permno'].isin(sp500_permnos)].copy()

    # For each permno, compute cumulative return from lookback_start to skip_end
    # Using log returns: s_i = log(P(skip_end)) - log(P(lookback_start))
    # Since we have daily returns (ret), we compound them:
    #   cumulative return = product of (1 + ret) over the window
    #   log return = log(cumulative return)

    # Get returns in the signal window [lookback_start, skip_end]
    window = eligible[
        (eligible['date'] >= date_lookback_start) &
        (eligible['date'] <= date_skip_end)
    ].copy()

    # Count trading days per stock in the window
    expected_days = len([d for d in all_dates
                         if d >= date_lookback_start and d <= date_skip_end])

    # Compute cumulative log return per stock
    # Drop stocks with missing returns in the window
    window = window.dropna(subset=['ret'])
    day_counts = window.groupby('permno').size().rename('n_days')

    cum_ret = (
        window
        .groupby('permno')['ret']
        .apply(lambda r: np.log((1 + r).prod()))
        .rename('raw_signal')
    )

    # Merge and filter: require complete data in the window
    signal = pd.DataFrame({'raw_signal': cum_ret, 'n_days': day_counts})
    # Allow a small tolerance: require at least 90% of expected days
    min_days = int(expected_days * 0.9)
    signal = signal[signal['n_days'] >= min_days].copy()

    if len(signal) < 10:
        # Too few stocks to meaningfully z-score
        return pd.DataFrame(columns=['permno', 'date', 'raw_signal', 'z_signal'])

    # Cross-sectional z-score
    mu = signal['raw_signal'].mean()
    sigma = signal['raw_signal'].std()
    signal['z_signal'] = (signal['raw_signal'] - mu) / sigma

    # Add metadata
    signal['date'] = rebal_date
    signal = signal.reset_index()[['permno', 'date', 'raw_signal', 'z_signal']]

    return signal


def compute_all_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MOM_12_1 signal on every monthly rebalance date.

    Returns a panel of (permno, date, raw_signal, z_signal).
    """
    print("Computing MOM_12_1 signals...")

    # Load S&P 500 membership for point-in-time lookups
    sp500 = pd.read_parquet(DATA_RAW / "crsp_sp500_membership.parquet")
    sp500['ending'] = sp500['ending'].fillna(pd.Timestamp('2099-12-31'))

    # Get rebalance dates: we need signals for the full period
    # (in-sample + holdout), but we'll only USE in-sample for development
    start = pd.Timestamp(INSAMPLE_START)
    end = pd.Timestamp(HOLDOUT_END)
    rebal_dates = get_rebalance_dates(panel, start, end)
    print(f"  {len(rebal_dates)} rebalance dates from "
          f"{rebal_dates[0].date()} to {rebal_dates[-1].date()}")

    # Sort panel by date for efficient slicing
    panel = panel.sort_values('date').reset_index(drop=True)

    all_signals = []
    for i, rebal_date in enumerate(rebal_dates):
        # Point-in-time S&P 500 members
        mask = (sp500['start'] <= rebal_date) & (sp500['ending'] >= rebal_date)
        sp500_permnos = set(sp500.loc[mask, 'permno'])

        # Filter panel to dates <= rebal_date (no look-ahead)
        panel_up_to_t = panel[panel['date'] <= rebal_date]

        # Compute signal
        sig = compute_signal_on_date(panel_up_to_t, rebal_date, sp500_permnos)
        all_signals.append(sig)

        if (i + 1) % 12 == 0 or i == len(rebal_dates) - 1:
            n_stocks = len(sig)
            print(f"  [{i+1}/{len(rebal_dates)}] {rebal_date.date()}: "
                  f"{n_stocks} stocks scored")

    result = pd.concat(all_signals, ignore_index=True)

    # Save
    path = DATA_PROCESSED / "signals.parquet"
    result.to_parquet(path, index=False)
    print(f"\n  Signal panel: {len(result):,} rows, "
          f"{result['permno'].nunique()} unique permnos")
    print(f"  Saved to {path}")

    return result


if __name__ == "__main__":
    print("Loading daily panel...")
    panel = pd.read_parquet(DATA_PROCESSED / "daily_panel.parquet")
    compute_all_signals(panel)
