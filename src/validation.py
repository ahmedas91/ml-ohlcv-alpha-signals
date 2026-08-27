"""Yearly walk-forward splitter.

Expanding window: for test year Y, train uses every
year strictly before val, val is the `val_years` years ending at Y-1, and
test is Y itself. `df` is expected to already have the correlation-pruning
block of initial years filtered out. `min_test_year` is caller-supplied 
(not derived from `df`'s first year) so the caller can control how much train 
history a fold needs before testing starts.

Each block is purged of its trailing `purge_days` trading dates before
fitting, since a forward-return target near a block boundary can be
computed from a price that falls in the next block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    test_year: int
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray


def _purge_tail(date_arr: np.ndarray, mask: np.ndarray, purge_days: int) -> np.ndarray:
    if purge_days <= 0 or not mask.any():
        return mask
    block_dates = np.unique(date_arr[mask])
    purge_dates = block_dates[-purge_days:]
    return mask & ~np.isin(date_arr, purge_dates)


def yearly_walk_forward(
    df: pd.DataFrame,
    fit_fn,
    x_cols,
    y_col,
    *,
    min_test_year: int,
    max_test_year: int | None = None,
    val_years: int = 1,
    purge_days: int = 5,
    date_col: str = "date",
    fillna_zero: bool = True,
    **kwargs,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Fit/predict per year via `fit_fn`, returning preds, coefs, and the split table.

    `fit_fn(X_tr, y_tr, X_val, y_val, X_tv, y_tv, X_te, **kwargs) ->
    (test_pred, coef)`.

    `fillna_zero=True` (default) fills NaN features with 0
    before fitting -- neutral when the features are rank-normalized to
    (-1, 1]. Pass False to hand NaNs through to `fit_fn` untouched, for
    fit_fns that scale per fold and must impute *after* scaling.
    """
    y = df[y_col].to_numpy(dtype=np.float64)
    X = df[x_cols].to_numpy(dtype=np.float64)
    if fillna_zero:
        # NaN -> 0 only (exactly fillna(0)); leave any inf untouched.
        # to_numpy() may return a read-only view when no dtype cast happens,
        # so only fill in place when we own a writable array.
        X = np.nan_to_num(X, copy=not X.flags.writeable, posinf=np.inf, neginf=-np.inf)
    date = df[date_col].to_numpy()
    year = df[date_col].dt.year.to_numpy()

    if max_test_year is None:
        max_test_year = int(year.max())

    pred_blocks = []
    coefs = {}
    folds = []

    for test_year in range(min_test_year, max_test_year + 1):
        val_end = test_year - 1
        val_start = val_end - val_years + 1

        train_mask = year < val_start
        val_mask = (year >= val_start) & (year <= val_end)
        test_mask = year == test_year

        train_mask = _purge_tail(date, train_mask, purge_days)
        val_mask = _purge_tail(date, val_mask, purge_days)

        assert not np.any(train_mask & val_mask)
        assert not np.any(val_mask & test_mask)
        assert not np.any(train_mask & test_mask)

        tv_mask = train_mask | val_mask

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        X_tv, y_tv = X[tv_mask], y[tv_mask]

        test_pred, coef = fit_fn(X_tr, y_tr, X_val, y_val, X_tv, y_tv, X_te, **kwargs)

        pred_blocks.append(pd.DataFrame({
            "test_year": test_year,
            "date": date[test_mask],
            "y": y_te,
            "yhat": test_pred,
        }))
        coefs[test_year] = pd.Series(coef, index=x_cols)
        folds.append(Fold(test_year, train_mask, val_mask, test_mask))

    preds = pd.concat(pred_blocks).reset_index(drop=True)
    coefs = pd.concat(coefs)
    split_table = build_split_table(df, folds, date_col=date_col)
    return preds, coefs, split_table


def build_split_table(df: pd.DataFrame, folds: list[Fold], *, date_col: str = "date") -> pd.DataFrame:
    """Long table of (test_year, date, role) for every date assigned a role in a fold."""
    date = df[date_col].to_numpy()

    rows = []
    for fold in folds:
        for role, mask in (("train", fold.train_mask), ("val", fold.val_mask), ("test", fold.test_mask)):
            role_dates = np.unique(date[mask])
            rows.append(pd.DataFrame({
                "test_year": fold.test_year,
                "date": role_dates,
                "role": role,
            }))

    return pd.concat(rows, ignore_index=True).sort_values(["test_year", "date"]).reset_index(drop=True)


def _toy_fit_fn(X_tr, y_tr, X_val, y_val, X_tv, y_tv, X_te, l1_ratio=0.5):
    """Elastic net w/ alpha picked on val, refit on train+val. Sanity-check only."""
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)

    val_error = {}
    for alpha in np.logspace(-3, -0.5, 10):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(X_tr_s, y_tr)
        val_error[alpha] = mean_squared_error(y_val, model.predict(X_val_s))
    alpha = min(val_error, key=val_error.get)

    scaler = StandardScaler().fit(X_tv)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(scaler.transform(X_tv), y_tv)
    return model.predict(scaler.transform(X_te)), model.coef_


if __name__ == "__main__":
    # Fake 8-year panel, one predictor correlated with the target.
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", "2022-12-31")
    x1 = rng.normal(size=len(dates))
    fake_df = pd.DataFrame({
        "date": dates,
        "x1": x1,
        "y": 2.0 * x1 + rng.normal(scale=0.5, size=len(dates)),
    })

    preds, coefs, split_table = yearly_walk_forward(
        fake_df, _toy_fit_fn, x_cols=["x1"], y_col="y",
        min_test_year=2019, purge_days=5,
    )

    print("=== per-fold test-year IC (corr(y, yhat)) ===")
    print(preds.groupby("test_year").apply(lambda g: g["y"].corr(g["yhat"])))

    print("\n=== coefs (x1) by test_year ===")
    print(coefs)

    print("\n=== split table: date range + row count per (test_year, role) ===")
    print(split_table.groupby(["test_year", "role"])["date"].agg(["min", "max", "count"]))
