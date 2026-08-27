"""Cross-sectional operators. Each groups by date only.

`indneutralize` uses `sic2`, mapped to `IndClass.industry`.
"""

from __future__ import annotations

import pandas as pd

_DATE_LEVEL = "date"


def rank(x: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank within each date, in (0, 1], ties averaged."""
    return x.groupby(level=_DATE_LEVEL).rank(pct=True)


def scale(x: pd.Series, a: float = 1.0) -> pd.Series:
    """Rescale x within each date so sum(abs(x)) == a."""
    abs_sum = x.abs().groupby(level=_DATE_LEVEL).transform("sum")
    return x / abs_sum * a


def zscore(x: pd.Series) -> pd.Series:
    """Cross-sectional z-score within each date: (x - mean) / std."""
    grouped = x.groupby(level=_DATE_LEVEL)
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    return (x - mean) / std


def winsorize(x: pd.Series, limits: tuple[float, float] = (0.01, 0.99)) -> pd.Series:
    """Clip x within each date to its [lo, hi] percentile range."""
    lo, hi = limits
    grouped = x.groupby(level=_DATE_LEVEL)
    dates = x.index.get_level_values(_DATE_LEVEL)
    lower = grouped.quantile(lo).reindex(dates).to_numpy()
    upper = grouped.quantile(hi).reindex(dates).to_numpy()
    return x.clip(lower=lower, upper=upper)


def indneutralize(x: pd.Series, g: pd.Series) -> pd.Series:
    """Demean x within each (date, g) group. Null `g` → NaN."""
    date = x.index.get_level_values(_DATE_LEVEL)
    group_mean = x.groupby([date, g]).transform("mean")
    return (x - group_mean).where(g.notna())
