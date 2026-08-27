"""Kakushadze time-series operators on a `(permno, segment_id, date)` Series.

Windows never look ahead or across a panel gap (`segment_id` blocks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _norm_window(d: float) -> int:
    d = int(np.floor(d))
    if d < 1:
        raise ValueError(f"window must be >= 1, got {d}")
    return d


def signedpower(x: pd.Series, a: float) -> pd.Series:
    """sign(x) * |x|^a -- not literal x**a, so fractional a stays real-valued for negative x."""
    return np.sign(x) * np.abs(x) ** a


def where(cond: pd.Series, if_true, if_false) -> pd.Series:
    """Paper ternary `cond ? if_true : if_false`. NA in `cond` stays NA."""
    true_s = if_true if isinstance(if_true, pd.Series) else pd.Series(if_true, index=cond.index)
    false_s = if_false if isinstance(if_false, pd.Series) else pd.Series(if_false, index=cond.index)
    result = true_s.where(cond.fillna(False).astype(bool), false_s)
    return result.mask(cond.isna())


def delay(x: pd.Series, d: int) -> pd.Series:
    """Value of x, d days ago (within the same trading segment)."""
    d = _norm_window(d)
    values = np.asarray(x.to_numpy(dtype=float))
    n = len(values)
    out = np.full(n, np.nan)
    starts = _group_starts(x)
    ends = np.append(starts[1:], n)
    for s, e in zip(starts, ends, strict=True):
        if e - s <= d:
            continue
        out[s + d : e] = values[s : e - d]
    return pd.Series(out, index=x.index)


def delta(x: pd.Series, d: int) -> pd.Series:
    """Today's value of x minus the value d days ago."""
    d = _norm_window(d)
    return x - delay(x, d)


def ts_sum(x: pd.Series, d: int) -> pd.Series:
    return _window_reduce(x, d, lambda w: w.sum(axis=1))


def ts_mean(x: pd.Series, d: int) -> pd.Series:
    return _window_reduce(x, d, lambda w: w.mean(axis=1))


def ts_min(x: pd.Series, d: int) -> pd.Series:
    return _window_reduce(x, d, lambda w: w.min(axis=1))


def ts_max(x: pd.Series, d: int) -> pd.Series:
    return _window_reduce(x, d, lambda w: w.max(axis=1))


def ts_stddev(x: pd.Series, d: int) -> pd.Series:
    return _window_reduce(x, d, lambda w: w.std(axis=1, ddof=1))


def _group_starts(x: pd.Series) -> np.ndarray:
    """Start index of each (permno, segment_id) block. Requires rows grouped that way."""
    permno = x.index.get_level_values("permno").to_numpy()
    seg = x.index.get_level_values("segment_id").to_numpy()
    new_group = np.empty(len(x), dtype=bool)
    new_group[0] = True
    new_group[1:] = (permno[1:] != permno[:-1]) | (seg[1:] != seg[:-1])
    return np.flatnonzero(new_group)


def _window_reduce(x: pd.Series, d: int, reduce_fn) -> pd.Series:
    """Apply `reduce_fn` to each trailing-d window. Window row is oldest → today.

    `reduce_fn(windows)` gets an array of shape (n_windows, d). Result is NaN
    if the window is short or contains NaN. Does not look ahead or across segments.
    """
    d = _norm_window(d)
    values = np.asarray(x.to_numpy(dtype=float))
    n = len(values)
    out = np.full(n, np.nan)
    starts = _group_starts(x)
    ends = np.append(starts[1:], n)
    for s, e in zip(starts, ends, strict=True):
        if e - s < d:
            continue
        windows = sliding_window_view(values[s:e], d)
        result = reduce_fn(windows)
        result = np.where(np.isnan(windows).any(axis=1), np.nan, result)
        out[s + d - 1 : e] = result
    return pd.Series(out, index=x.index)


def ts_product(x: pd.Series, d: int) -> pd.Series:
    return _window_reduce(x, d, lambda w: w.prod(axis=1))


def ts_rank(x: pd.Series, d: int) -> pd.Series:
    """Percentile rank of today's value within the trailing d-day window, in (0, 1]."""
    return _window_reduce(x, d, lambda w: (w <= w[:, -1:]).sum(axis=1) / w.shape[1])


def ts_argmax(x: pd.Series, d: int) -> pd.Series:
    """Days ago that ts_max(x, d) occurred: 0 = today, d - 1 = oldest day in window."""
    return _window_reduce(x, d, lambda w: w.shape[1] - 1 - np.argmax(w, axis=1))


def ts_argmin(x: pd.Series, d: int) -> pd.Series:
    """Days ago that ts_min(x, d) occurred: 0 = today, d - 1 = oldest day in window."""
    return _window_reduce(x, d, lambda w: w.shape[1] - 1 - np.argmin(w, axis=1))


def decay_linear(x: pd.Series, d: int) -> pd.Series:
    """Weighted moving average over the past d days; today's weight = d, oldest = 1."""
    d = _norm_window(d)
    weights = np.arange(1, d + 1, dtype=float)
    weights /= weights.sum()
    return _window_reduce(x, d, lambda w: w @ weights)


def ts_cov(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """Time-serial sample covariance (ddof=1) of x and y over the past d days."""
    d = _norm_window(d)
    mean_x = ts_mean(x, d)
    mean_y = ts_mean(y, d)
    mean_xy = ts_mean(x * y, d)
    return (mean_xy - mean_x * mean_y) * d / (d - 1)


def ts_corr(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """Trailing-d correlation. NaN (not inf) if either series is constant."""
    d = _norm_window(d)
    std_x = ts_stddev(x, d)
    std_y = ts_stddev(y, d)
    corr = ts_cov(x, y, d) / (std_x * std_y)
    return corr.where((std_x != 0) & (std_y != 0))
