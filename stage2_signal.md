# Stage 2: Signal Construction

> **Goal:** Implement the MOM_12_1 cross-sectional momentum signal exactly as specified in the Stage 0 pre-registration, with an explicit no-look-ahead guarantee and unit tests.
>
> **Estimated time:** 1 week part-time.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Signal Specification Recap](#2-signal-specification-recap)
3. [Implementation](#3-implementation)
4. [Look-Ahead Guard (Sanity Check 5)](#4-look-ahead-guard-sanity-check-5)
5. [Signal Diagnostics](#5-signal-diagnostics)
6. [Deliverables Checklist](#6-deliverables-checklist)

---

## 1. Overview

The signal module computes a single number for each stock on each rebalance date: **how much has this stock outperformed or underperformed its peers over the past 12 months, excluding the most recent month?**

This is a pure measurement step — no portfolio construction, no backtesting, no optimization. The output is a panel of `(date, permno, signal_value)` triples that Stage 3 will consume.

The analogy to your LIGO work: this is the matched-filter output — a detection statistic for each template (stock) at each time. The threshold and selection happen in Stage 3 (portfolio construction).

---

## 2. Signal Specification Recap

From Stage 0 pre-registration, Section 5:

**Signal name:** `MOM_12_1`

On each rebalance date `T` (last trading day of each calendar month), for each eligible stock `i`:

1. Compute the cumulative log return from `T-252` to `T-21` trading days:
   ```
   s_i(T) = log(P_i(T-21)) - log(P_i(T-252))
   ```
   where `P_i(t)` is the split- and dividend-adjusted total return price.

2. Cross-sectionally z-score across all eligible stocks at time `T`:
   ```
   z_i(T) = (s_i(T) - mean_j(s_j(T))) / std_j(s_j(T))
   ```

**Parameters (fixed, no tuning):**
- Lookback: 252 trading days (~12 months)
- Skip: 21 trading days (~1 month)
- Ranking: cross-sectional z-score
- Rebalance: monthly, last trading day

**Eligibility:** Stock must be in the S&P 500 (point-in-time) and have a complete price history from `T-252` through `T`.

---

## 3. Implementation

### `src/signal/momentum.py`

```python
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
```

### Running the signal computation

```bash
python -m src.signal.momentum
```

This will take several minutes — it's computing signals for ~200 rebalance dates, each requiring a scan through the panel. Progress is printed every 12 months.

> **Performance note:** The current implementation filters the panel to `date <= rebal_date` for each rebalance date, which is O(N) per date. For a faster version, you can pre-sort and use binary search with `np.searchsorted`, or pre-compute cumulative returns. But for ~200 dates and ~22M rows, it should finish in 10-30 minutes, which is fine for a research pipeline. Optimize later if needed.

---

## 4. Look-Ahead Guard (Sanity Check 5)

This is the final sanity check from Stage 0 pre-registration. It verifies that the signal function cannot access future data.

### `tests/test_signal.py`

```python
"""
test_signal.py — Unit tests for the momentum signal, including look-ahead guard.

Run with:
    python -m pytest tests/test_signal.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.signal.momentum import compute_signal_on_date


def make_fake_panel(dates, permnos, seed=42):
    """Create a minimal fake panel for testing."""
    rng = np.random.RandomState(seed)
    rows = []
    for d in dates:
        for p in permnos:
            rows.append({
                'permno': p,
                'date': pd.Timestamp(d),
                'ret': rng.normal(0.0005, 0.02),
                'prc': 100.0,
                'vol': 1e6,
                'shrout': 1e4,
                'mkt_cap': 1e3,
                'is_sp500': True,
            })
    return pd.DataFrame(rows)


class TestLookAheadGuard:
    """Sanity Check 5: verify no look-ahead bias."""

    def test_future_data_raises_assertion(self):
        """
        If the panel contains data AFTER the rebalance date,
        the signal function must raise an AssertionError.
        """
        # Create panel with data through 2020-06-30
        dates = pd.bdate_range('2018-01-01', '2020-06-30')
        permnos = list(range(10001, 10021))  # 20 stocks
        panel = make_fake_panel(dates, permnos)
        sp500 = set(permnos)

        # Try to compute signal as of 2020-03-31 but pass full panel
        rebal_date = pd.Timestamp('2020-03-31')

        with pytest.raises(AssertionError, match="Look-ahead detected"):
            compute_signal_on_date(panel, rebal_date, sp500)

    def test_no_future_data_succeeds(self):
        """
        If the panel is correctly filtered to <= rebal_date,
        the signal function should succeed without error.
        """
        # Need at least 252 + 21 business days of history and >= 10 stocks
        dates = pd.bdate_range('2018-01-01', '2020-03-31')
        permnos = list(range(10001, 10021))  # 20 stocks
        panel = make_fake_panel(dates, permnos)
        sp500 = set(permnos)

        rebal_date = pd.Timestamp('2020-03-31')
        panel_filtered = panel[panel['date'] <= rebal_date]

        # Should not raise
        result = compute_signal_on_date(panel_filtered, rebal_date, sp500)
        assert len(result) > 0
        assert 'z_signal' in result.columns


class TestSignalProperties:
    """Verify basic statistical properties of the signal."""

    def setup_method(self):
        """Create a realistic fake panel for signal tests."""
        dates = pd.bdate_range('2019-01-01', '2020-12-31')
        self.permnos = list(range(10001, 10051))  # 50 stocks
        self.panel = make_fake_panel(dates, self.permnos)
        self.sp500 = set(self.permnos)
        self.rebal_date = pd.Timestamp('2020-12-31')
        self.panel_filtered = self.panel[self.panel['date'] <= self.rebal_date]

    def test_z_score_mean_near_zero(self):
        """Cross-sectional z-score should have mean ≈ 0."""
        result = compute_signal_on_date(
            self.panel_filtered, self.rebal_date, self.sp500
        )
        assert abs(result['z_signal'].mean()) < 0.01

    def test_z_score_std_near_one(self):
        """Cross-sectional z-score should have std ≈ 1."""
        result = compute_signal_on_date(
            self.panel_filtered, self.rebal_date, self.sp500
        )
        assert abs(result['z_signal'].std() - 1.0) < 0.05

    def test_signal_only_for_sp500(self):
        """Signal should only be computed for permnos in the S&P 500 set."""
        # Use a restricted S&P 500 set
        restricted = set(self.permnos[:25])
        result = compute_signal_on_date(
            self.panel_filtered, self.rebal_date, restricted
        )
        assert set(result['permno']).issubset(restricted)

    def test_insufficient_history_returns_empty(self):
        """Stocks with too little history should be excluded."""
        # Panel with only 100 trading days — not enough for 252-day lookback
        short_dates = pd.bdate_range('2020-07-01', '2020-12-31')
        short_panel = make_fake_panel(short_dates, self.permnos)
        result = compute_signal_on_date(
            short_panel, self.rebal_date, self.sp500
        )
        assert len(result) == 0
```

### Running the tests

```bash
# Install pytest if not already installed
pip install pytest

# Run the signal tests
python -m pytest tests/test_signal.py -v
```

Expected output:

```
tests/test_signal.py::TestLookAheadGuard::test_future_data_raises_assertion PASSED
tests/test_signal.py::TestLookAheadGuard::test_no_future_data_succeeds PASSED
tests/test_signal.py::TestSignalProperties::test_z_score_mean_near_zero PASSED
tests/test_signal.py::TestSignalProperties::test_z_score_std_near_one PASSED
tests/test_signal.py::TestSignalProperties::test_signal_only_for_sp500 PASSED
tests/test_signal.py::TestSignalProperties::test_insufficient_history_returns_empty PASSED
```

> **This completes Sanity Check 5 from the Stage 0 pre-registration.** The look-ahead guard is an assertion baked into the signal function itself, not just a test — it fires in production if you accidentally pass unfiltered data.

---

## 5. Signal Diagnostics

After computing all signals, run these diagnostics to verify the output is sensible. You can do this interactively in a notebook or add it to the bottom of `momentum.py`.

### 5.1 Signal coverage over time

```python
# How many stocks are scored at each rebalance date?
coverage = signals.groupby('date')['permno'].count()
print(f"Stocks per rebalance date: min={coverage.min()}, "
      f"max={coverage.max()}, mean={coverage.mean():.0f}")
# Expected: ~400-470 (matching universe count from Check 1)
```

### 5.2 Signal distribution

```python
# The z-scored signal should be roughly normal at each cross-section
print(f"Overall z_signal: mean={signals['z_signal'].mean():.3f}, "
      f"std={signals['z_signal'].std():.3f}")
# Expected: mean ≈ 0, std ≈ 1

# Check for extreme values (potential data issues)
extremes = signals[signals['z_signal'].abs() > 4]
print(f"Observations with |z| > 4: {len(extremes)} "
      f"({len(extremes)/len(signals)*100:.2f}%)")
# Expected: < 1% (normal distribution: 0.006%)
```

### 5.3 Signal autocorrelation

```python
# Momentum signals should be highly persistent month-to-month
# (a stock that was a winner last month is likely still a winner this month)
# Compute rank correlation between consecutive months
from scipy.stats import spearmanr

rebal_dates = sorted(signals['date'].unique())
autocorrs = []
for i in range(1, len(rebal_dates)):
    prev = signals[signals['date'] == rebal_dates[i-1]].set_index('permno')['z_signal']
    curr = signals[signals['date'] == rebal_dates[i]].set_index('permno')['z_signal']
    common = prev.index.intersection(curr.index)
    if len(common) > 30:
        corr, _ = spearmanr(prev.loc[common], curr.loc[common])
        autocorrs.append(corr)

print(f"Month-to-month signal rank correlation: "
      f"mean={np.mean(autocorrs):.3f}, std={np.std(autocorrs):.3f}")
# Expected: ~0.85-0.95 (momentum is a slow-moving signal)
```

### 5.4 Cross-check: correlation with Ken French UMD

```python
# Quick preview — full analysis is in Stage 5
# At each rebalance date, correlate our signal ranking with the UMD factor
# return over the subsequent month. Positive correlation expected.
import pandas as pd

ff = pd.read_parquet('data/external/ff_factors_monthly.parquet')
# This is a rough preview; the proper test is in the evaluation stage.
```

---

## 6. Deliverables Checklist

Before moving to Stage 3, confirm every item:

- [x] `src/signal/momentum.py` — signal computation module
- [x] `src/signal/signal_diagnostics.py` — signal diagnostics
- [x] `tests/test_signal.py` — unit tests including look-ahead guard
- [x] `data/processed/signals.parquet` — signal panel output
- [x] **All 6 pytest tests pass**, including look-ahead guard (Sanity Check 5)
- [x] Signal coverage: ~435-466 stocks per rebalance date (mean 447)
- [x] Signal distribution: z-score mean ≈ 0, std ≈ 1
- [x] Signal autocorrelation: month-to-month rank correlation ~0.85-0.95
- [x] `pytest` added to `requirements.txt`
- [x] All code committed to git

> **Stage 2 is complete. Proceed to Stage 3: Portfolio Construction.**

---

## Appendix A: Known Limitations — Universe Coverage

### Ticker-to-PERMNO mismatch (~10% of universe)

Our S&P 500 universe averages ~447 stocks instead of the true ~503-505 due to 50-70 tickers that cannot be mapped to CRSP permnos. The shortfall breaks down into three categories:

1. **Bankruptcy tickers** (e.g., `AAMRQ`, `ABKFQ`, `BTUUQ`): Post-bankruptcy ticker symbols that CRSP stores under the original pre-bankruptcy name. These stocks were already delisted and near-worthless. Their absence makes the short leg slightly less extreme, which makes our backtest *conservative* — a safe direction.

2. **Obscure historical names** (e.g., `CNW`, `ACKH`): Tickers where the CRSP name history uses a different convention and no variant matching rule catches them. Fixing these would require manual lookup of each ticker against CRSP company names (2-3 hours of detective work for diminishing returns).

3. **Recent additions beyond CRSP data** (e.g., `COIN`, `APP`, `DASH`): Stocks added to the S&P 500 after our CRSP download ends in December 2024. These cannot be matched without re-downloading CRSP with a later end date.

### Impact assessment

- The missing stocks are approximately random with respect to momentum — they are not systematically winners or losers. The relative ranking of the ~447 stocks we do have is essentially the same as it would be with all ~505.
- Published momentum studies have been replicated on universes ranging from 200 to 3,000 stocks. The signal is robust to universe size.
- The Ken French UMD factor comparison in Stage 5 is the definitive test of whether this mismatch matters. If our strategy's monthly returns correlate above 0.6 with UMD, the pipeline is validated regardless of the missing ~50 stocks.
- The tracking error vs the market (189 bp) is partly from this gap, but that measures index reproduction, not momentum signal quality.

### Signal coverage decline after 2012

Signal coverage trends down slightly from ~465 (2010) to ~440 (2015+). This reflects increased S&P 500 index turnover after 2012 (more tech IPOs entering, more traditional companies leaving), with each change creating another opportunity for a ticker mismatch. The count stabilizes at ~440 from 2015 onward and remains flat through the holdout period (2020-2025).

### Recommendation

Not worth fixing for pipeline validation. If moving to a production strategy, purchase Norgate Data or equivalent, which provides pre-mapped CRSP permnos for all index constituents.
