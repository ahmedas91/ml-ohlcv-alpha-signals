"""Scorecard: yearly IC metrics for walk-forward runs.

All functions take the `preds` frame produced by `yearly_walk_forward`
(columns: test_year, date, y, yhat) or the `coefs` Series it returns
(MultiIndex: test_year, feature), so they are usable on any persisted run.
Used by run_normalization_ablation.py; `pooled_pearson_ic_by_year` matches
the corr(y, yhat)-per-test-year convention run_models.py prints.

Decile Sharpe is still not implemented.
"""

from __future__ import annotations

import pandas as pd

# A coefficient with |value| below this is "zero": coordinate descent at
# tol=1e-3 leaves residue well below this, real selections sit well above.
# Matches the 1e-8 default of models.coefficient_comparison so the two
# selection tables can never disagree on whether a feature was kept.
ZERO_TOL = 1e-8


def _safe_corr(g: pd.DataFrame, method: str) -> float:
    """corr(y, yhat), NaN without warning spam when either side is constant
    (a shrunk-to-null model predicts a constant -- a real outcome worth
    reporting as NaN + n_nonzero_coefs=0, not a numerical error)."""
    if g["yhat"].nunique() <= 1 or g["y"].nunique() <= 1:
        return float("nan")
    return g["y"].corr(g["yhat"], method=method)


def pooled_pearson_ic_by_year(preds: pd.DataFrame) -> pd.Series:
    """Per-test-year Pearson corr(y, yhat), pooled over dates -- the
    convention run_models.py prints."""
    return preds.groupby("test_year").apply(lambda g: _safe_corr(g, "pearson"))


def mean_daily_spearman_ic_by_year(preds: pd.DataFrame) -> pd.Series:
    """Mean over dates of the per-date cross-sectional Spearman rank IC --
    the standard signal-quality metric for a cross-sectional signal."""
    daily = preds.groupby(["test_year", "date"]).apply(lambda g: _safe_corr(g, "spearman"))
    return daily.groupby("test_year").mean()


def n_nonzero_by_year(coefs: pd.Series, *, zero_tol: float = ZERO_TOL) -> pd.Series:
    """Count of nonzero coefficients per test-year fold. 0 = the validation
    search picked the null (all-shrunk) model for that fold."""
    return (coefs.abs() > zero_tol).groupby(level=0).sum()
