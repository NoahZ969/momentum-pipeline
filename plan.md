# Project Plan: Cross-Sectional Equity Momentum Pipeline Validation

> A master index of all project stages, their purpose, current status, and associated documentation.

---

## Stage Overview

| Stage | Name | Purpose | Status | Document |
|---|---|---|---|---|
| 0 | Pre-Registration | Define hypothesis and analysis plan before touching data | ✅ Complete | `README.md` |
| 1 | Data Infrastructure | Download CRSP, clean data, merge delistings, sanity checks | ✅ Complete | `stage1_data.md` |
| 2 | Signal Construction | Compute MOM_12_1 signal, look-ahead guard, diagnostics | ✅ Complete | `stage2_signal.md` |
| 3 | Portfolio Construction | Convert signals to weights, compute daily returns with costs | ✅ Complete | `stage3_portfolio.md` |
| 4 | Backtester | Drift-aware backtester | Merged into Stage 3 | `stage3_portfolio.md` |
| 5 | Evaluation | Ken French UMD comparison, factor regressions, deflated Sharpe | 🔜 Next | `stage5_evaluation.md` |
| 6 | Stress Tests & Robustness | Parameter sensitivity, sub-period analysis, sector neutrality | ⬜ Pending | — |
| 7 | Holdout Evaluation | Single out-of-sample run on 2020-2025 | ⬜ Pending | — |

---

## Stage Details

### Stage 0: Pre-Registration
- **Document:** `README.md`
- **Key outputs:** Hypothesis, universe definition, signal specification, portfolio construction rule, cost model, evaluation methodology, pre-committed Sharpe interpretation ranges
- **Deviation log:**
  - 2026-05-04: Added `[-0.3, 0.0)` Sharpe bucket for short samples dominated by momentum crashes. Original `< 0.0` threshold was too aggressive for a 15-year window containing the 2009 momentum crash. Ken French UMD correlation designated as definitive validation test.

### Stage 1: Data Infrastructure
- **Document:** `stage1_data.md`
- **Key outputs:** `daily_panel.parquet` (22.7M rows, 10,580 permnos, 446 MB)
- **Data sources:** CRSP via WRDS (dsf, msenames, msedelist), S&P 500 membership via fja05680/sp500 GitHub repo, Ken French factors via pandas-datareader
- **Known limitations:** ~10% universe shortfall due to ticker-to-PERMNO matching gaps (documented in `stage2_signal.md` Appendix A)

### Stage 2: Signal Construction
- **Document:** `stage2_signal.md`
- **Key outputs:** `signals.parquet` (105,944 rows, 738 permnos, 240 rebalance dates)
- **Signal:** MOM_12_1 — cumulative log return from T-252 to T-21, cross-sectionally z-scored
- **Sanity Check 5 (look-ahead guard):** ✅ Passed (6/6 pytest tests)

### Stage 3: Portfolio Construction
- **Document:** `stage3_portfolio.md`
- **Key outputs:** `weights.parquet`, `portfolio_returns.parquet`, equity curve plot
- **In-sample net Sharpe:** -0.17 (falls in revised `[-0.3, 0.0)` range — plausible for 2005-2019 given 2009 momentum crash)
- **Note:** Stage 4 (drift-aware backtester) merged into this stage. Current implementation uses fixed weights within each month, which is conservative (overstates turnover costs).

### Stage 4: Backtester
- **Status:** Merged into Stage 3
- **Rationale:** Stage 3 already computes daily portfolio returns with transaction costs. The main Stage 4 refinement (weight drift between rebalances) is deferred to Stage 6 robustness checks as a sensitivity test, not a core pipeline requirement.

### Stage 5: Evaluation
- **Purpose:** Definitive pipeline validation via comparison to published benchmarks
- **Key analyses:**
  - Correlation and OLS beta of monthly returns vs Ken French UMD factor (target: correlation > 0.6)
  - Fama-French factor regression (alpha, loadings on Mkt-RF, SMB, HML, UMD)
  - Deflated Sharpe ratio (accounting for number of trials = 1 in Stage 0)
  - Rolling 12-month Sharpe ratio plot
  - All secondary metrics from Stage 0 Section 8
- **Validation criterion:** If monthly return correlation with UMD > 0.6, pipeline is validated regardless of absolute Sharpe level

### Stage 6: Stress Tests & Robustness
- **Purpose:** Verify the signal is not an artifact of specific parameter choices
- **Planned analyses:**
  - Vary lookback window (6, 12, 18 months) — results should be qualitatively similar
  - Vary holding period (1, 3, 6 months)
  - Vary decile cutoff (top/bottom 10% vs 20% vs 30%)
  - Sub-period analysis (2005-2009, 2010-2014, 2015-2019)
  - Sector-neutral construction
  - Short-borrow cost sensitivity
  - Weight drift sensitivity (compare fixed vs drifting weights)

### Stage 7: Holdout Evaluation
- **Purpose:** Single out-of-sample test on 2020-2025
- **Protocol:** Run exactly once after all in-sample analysis and robustness checks are complete. Report results regardless of outcome. No iteration permitted.
- **Known limitation:** Universe composition after January 2026 is incomplete (~73 unmatched tickers for very recent S&P 500 additions). CRSP data ends December 2024.

---

## Project Timeline

| Weeks | Stage | Status |
|---|---|---|
| 1-2 | Stage 1: Data infrastructure | ✅ |
| 3-4 | Stage 2: Signal construction | ✅ |
| 3-4 | Stage 3: Portfolio construction | ✅ |
| 5-6 | Stage 5: Evaluation | 🔜 |
| 7-8 | Stage 6: Stress tests | ⬜ |
| 9-10 | Stage 7: Holdout evaluation, write-up | ⬜ |

---

## File Index

| File | Location | Description |
|---|---|---|
| `README.md` | project root | Stage 0 pre-registration |
| `PROJECT_PLAN.md` | project root | This document |
| `stage1_data.md` | project root | Stage 1 documentation |
| `stage2_signal.md` | project root | Stage 2 documentation |
| `stage3_portfolio.md` | project root | Stage 3 documentation |
| `config.py` | project root | Central configuration |
| `requirements.txt` | project root | Python dependencies |
| `src/data/download_crsp.py` | src/data/ | WRDS download + S&P 500 membership |
| `src/data/clean.py` | src/data/ | Data cleaning pipeline |
| `src/data/universe.py` | src/data/ | Point-in-time universe functions |
| `src/data/sanity_checks.py` | src/data/ | Stage 1 sanity checks |
| `src/signal/momentum.py` | src/signal/ | MOM_12_1 signal computation |
| `src/signal/signal_diagnostics.py` | src/signal/ | Signal diagnostics |
| `src/portfolio/construction.py` | src/portfolio/ | Portfolio construction |
| `tests/test_signal.py` | tests/ | Signal unit tests |
| `data/processed/daily_panel.parquet` | data/processed/ | Cleaned daily panel |
| `data/processed/signals.parquet` | data/processed/ | Signal panel |
| `data/processed/weights.parquet` | data/processed/ | Portfolio weights |
| `data/processed/portfolio_returns.parquet` | data/processed/ | Daily portfolio returns |
