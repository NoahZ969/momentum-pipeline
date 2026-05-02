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
