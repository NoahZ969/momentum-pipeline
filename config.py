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
