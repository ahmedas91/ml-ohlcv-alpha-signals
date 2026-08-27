"""Signal library. Call `prepare_panel` before any operator.

Paper `min`/`max`/`sum` map to `ts_min`/`ts_max`/`ts_sum`.
"""

from src.signals.alphas import ALPHAS, build_alpha_features
from src.signals.baseline import build_baseline_features, mom_12_1, rev_1d, rev_5d
from src.signals.classic_features import CLASSIC_FEATURES, build_classic_features
from src.signals.cross_sectional import indneutralize, rank, scale, winsorize, zscore
from src.signals.inputs import adv, prepare_panel
from src.signals.operators import (
    decay_linear,
    delay,
    delta,
    signedpower,
    ts_argmax,
    ts_argmin,
    ts_corr,
    ts_cov,
    ts_max,
    ts_mean,
    ts_min,
    ts_product,
    ts_rank,
    ts_stddev,
    ts_sum,
    where,
)

__all__ = [
    "ALPHAS",
    "CLASSIC_FEATURES",
    "adv",
    "build_alpha_features",
    "build_baseline_features",
    "build_classic_features",
    "decay_linear",
    "delay",
    "delta",
    "indneutralize",
    "mom_12_1",
    "prepare_panel",
    "rank",
    "rev_1d",
    "rev_5d",
    "scale",
    "signedpower",
    "ts_argmax",
    "ts_argmin",
    "ts_corr",
    "ts_cov",
    "ts_max",
    "ts_mean",
    "ts_min",
    "ts_product",
    "ts_rank",
    "ts_stddev",
    "ts_sum",
    "where",
    "winsorize",
    "zscore",
]
