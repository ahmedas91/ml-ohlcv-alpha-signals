"""Kakushadze alphas transcribed via operators.py. Gaps in numbering are skips.

11 of 101 formulas are outside the implemented library. `returns` = `ret`,
`cap` = `mcap`, `vwap` = daily-bar proxy, and `IndClass.industry` = `sic2`.
Docstrings quote Appendix A.1.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

from src.signals.cross_sectional import indneutralize, rank, scale
from src.signals.inputs import adv
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
    ts_min,
    ts_product,
    ts_rank,
    ts_stddev,
    ts_sum,
    where,
)


def alpha_001(panel: pd.DataFrame) -> pd.Series:
    """rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5"""
    returns = panel["ret"]
    close = panel["close"]
    x = where(returns < 0, ts_stddev(returns, 20), close)
    return rank(ts_argmax(signedpower(x, 2), 5)) - 0.5


def alpha_002(panel: pd.DataFrame) -> pd.Series:
    """(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
    open_ = panel["open"]
    close = panel["close"]
    volume = panel["volume"]
    return -1 * ts_corr(rank(delta(np.log(volume), 2)), rank((close - open_) / open_), 6)


def alpha_003(panel: pd.DataFrame) -> pd.Series:
    """(-1 * correlation(rank(open), rank(volume), 10))"""
    open_ = panel["open"]
    volume = panel["volume"]
    return -1 * ts_corr(rank(open_), rank(volume), 10)


def alpha_004(panel: pd.DataFrame) -> pd.Series:
    """(-1 * Ts_Rank(rank(low), 9))"""
    low = panel["low"]
    return -1 * ts_rank(rank(low), 9)


def alpha_005(panel: pd.DataFrame) -> pd.Series:
    """(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"""
    open_ = panel["open"]
    close = panel["close"]
    vwap = panel["vwap"]
    return rank(open_ - ts_sum(vwap, 10) / 10) * (-1 * rank(close - vwap).abs())


def alpha_006(panel: pd.DataFrame) -> pd.Series:
    """(-1 * correlation(open, volume, 10))"""
    open_ = panel["open"]
    volume = panel["volume"]
    return -1 * ts_corr(open_, volume, 10)


def alpha_007(panel: pd.DataFrame) -> pd.Series:
    """((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))"""
    close = panel["close"]
    volume = panel["volume"]
    if_true = (-1 * ts_rank(delta(close, 7).abs(), 60)) * np.sign(delta(close, 7))
    return where(adv(panel, 20) < volume, if_true, -1)


def alpha_008(panel: pd.DataFrame) -> pd.Series:
    """(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))"""
    open_ = panel["open"]
    returns = panel["ret"]
    x = ts_sum(open_, 5) * ts_sum(returns, 5)
    return -1 * rank(x - delay(x, 10))


def alpha_009(panel: pd.DataFrame) -> pd.Series:
    """((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))"""
    close = panel["close"]
    d1 = delta(close, 1)
    inner = where(ts_max(d1, 5) < 0, d1, -1 * d1)
    return where(0 < ts_min(d1, 5), d1, inner)


def alpha_010(panel: pd.DataFrame) -> pd.Series:
    """rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))"""
    close = panel["close"]
    d1 = delta(close, 1)
    inner = where(ts_max(d1, 4) < 0, d1, -1 * d1)
    return rank(where(0 < ts_min(d1, 4), d1, inner))


def alpha_011(panel: pd.DataFrame) -> pd.Series:
    """((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))"""
    close = panel["close"]
    volume = panel["volume"]
    vwap = panel["vwap"]
    spread = vwap - close
    return (rank(ts_max(spread, 3)) + rank(ts_min(spread, 3))) * rank(delta(volume, 3))


def alpha_012(panel: pd.DataFrame) -> pd.Series:
    """(sign(delta(volume, 1)) * (-1 * delta(close, 1)))"""
    close = panel["close"]
    volume = panel["volume"]
    return np.sign(delta(volume, 1)) * (-1 * delta(close, 1))


def alpha_013(panel: pd.DataFrame) -> pd.Series:
    """(-1 * rank(covariance(rank(close), rank(volume), 5)))"""
    close = panel["close"]
    volume = panel["volume"]
    return -1 * rank(ts_cov(rank(close), rank(volume), 5))


def alpha_014(panel: pd.DataFrame) -> pd.Series:
    """((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"""
    open_ = panel["open"]
    volume = panel["volume"]
    returns = panel["ret"]
    return (-1 * rank(delta(returns, 3))) * ts_corr(open_, volume, 10)


def alpha_015(panel: pd.DataFrame) -> pd.Series:
    """(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))"""
    high = panel["high"]
    volume = panel["volume"]
    return -1 * ts_sum(rank(ts_corr(rank(high), rank(volume), 3)), 3)


def alpha_016(panel: pd.DataFrame) -> pd.Series:
    """(-1 * rank(covariance(rank(high), rank(volume), 5)))"""
    high = panel["high"]
    volume = panel["volume"]
    return -1 * rank(ts_cov(rank(high), rank(volume), 5))


def alpha_017(panel: pd.DataFrame) -> pd.Series:
    """(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))"""
    close = panel["close"]
    volume = panel["volume"]
    return (
        (-1 * rank(ts_rank(close, 10)))
        * rank(delta(delta(close, 1), 1))
        * rank(ts_rank(volume / adv(panel, 20), 5))
    )


def alpha_018(panel: pd.DataFrame) -> pd.Series:
    """(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))"""
    open_ = panel["open"]
    close = panel["close"]
    return -1 * rank(ts_stddev((close - open_).abs(), 5) + (close - open_) + ts_corr(close, open_, 10))


def alpha_019(panel: pd.DataFrame) -> pd.Series:
    """((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))"""
    close = panel["close"]
    returns = panel["ret"]
    return (-1 * np.sign((close - delay(close, 7)) + delta(close, 7))) * (1 + rank(1 + ts_sum(returns, 250)))


def alpha_020(panel: pd.DataFrame) -> pd.Series:
    """(((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))"""
    open_ = panel["open"]
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    return (
        (-1 * rank(open_ - delay(high, 1)))
        * rank(open_ - delay(close, 1))
        * rank(open_ - delay(low, 1))
    )


def alpha_021(panel: pd.DataFrame) -> pd.Series:
    """((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))"""
    close = panel["close"]
    volume = panel["volume"]
    vol_ratio = volume / adv(panel, 20)
    cond_a = ts_sum(close, 8) / 8 + ts_stddev(close, 8) < ts_sum(close, 2) / 2
    cond_b = ts_sum(close, 2) / 2 < ts_sum(close, 8) / 8 - ts_stddev(close, 8)
    cond_c = (1 < vol_ratio) | (vol_ratio == 1)
    inner = where(cond_c, 1, -1)
    middle = where(cond_b, 1, inner)
    return where(cond_a, -1, middle)


def alpha_022(panel: pd.DataFrame) -> pd.Series:
    """(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))"""
    high = panel["high"]
    close = panel["close"]
    volume = panel["volume"]
    return -1 * delta(ts_corr(high, volume, 5), 5) * rank(ts_stddev(close, 20))


def alpha_023(panel: pd.DataFrame) -> pd.Series:
    """(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)"""
    high = panel["high"]
    return where(ts_sum(high, 20) / 20 < high, -1 * delta(high, 2), 0)


def alpha_024(panel: pd.DataFrame) -> pd.Series:
    """((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))"""
    close = panel["close"]
    x = delta(ts_sum(close, 100) / 100, 100) / delay(close, 100)
    cond = (x < 0.05) | (x == 0.05)
    return where(cond, -1 * (close - ts_min(close, 100)), -1 * delta(close, 3))


def alpha_025(panel: pd.DataFrame) -> pd.Series:
    """rank(((((-1 * returns) * adv20) * vwap) * (high - close)))"""
    high = panel["high"]
    close = panel["close"]
    returns = panel["ret"]
    vwap = panel["vwap"]
    return rank((-1 * returns) * adv(panel, 20) * vwap * (high - close))


def alpha_026(panel: pd.DataFrame) -> pd.Series:
    """(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))"""
    high = panel["high"]
    volume = panel["volume"]
    return -1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)


def alpha_027(panel: pd.DataFrame) -> pd.Series:
    """((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)"""
    volume = panel["volume"]
    vwap = panel["vwap"]
    x = rank(ts_sum(ts_corr(rank(volume), rank(vwap), 6), 2) / 2.0)
    return where(0.5 < x, -1, 1)


def alpha_028(panel: pd.DataFrame) -> pd.Series:
    """scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"""
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    return scale(ts_corr(adv(panel, 20), low, 5) + (high + low) / 2 - close)


def alpha_029(panel: pd.DataFrame) -> pd.Series:
    """(min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))"""
    close = panel["close"]
    returns = panel["ret"]
    a1 = rank(delta(close - 1, 5))
    a2 = rank(-1 * a1)
    a3 = rank(a2)
    a4 = ts_min(a3, 2)
    a5 = ts_sum(a4, 1)
    a6 = scale(np.log(a5))
    a7 = rank(a6)
    a8 = rank(a7)
    a9 = ts_product(a8, 1)
    return ts_min(a9, 5) + ts_rank(delay(-1 * returns, 6), 5)


def alpha_030(panel: pd.DataFrame) -> pd.Series:
    """(((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))"""
    close = panel["close"]
    volume = panel["volume"]
    sign_sum = (
        np.sign(close - delay(close, 1))
        + np.sign(delay(close, 1) - delay(close, 2))
        + np.sign(delay(close, 2) - delay(close, 3))
    )
    return (1.0 - rank(sign_sum)) * ts_sum(volume, 5) / ts_sum(volume, 20)


def alpha_031(panel: pd.DataFrame) -> pd.Series:
    """((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))"""
    close = panel["close"]
    low = panel["low"]
    inner = decay_linear(-1 * rank(rank(delta(close, 10))), 10)
    return (
        rank(rank(rank(inner)))
        + rank(-1 * delta(close, 3))
        + np.sign(scale(ts_corr(adv(panel, 20), low, 12)))
    )


def alpha_032(panel: pd.DataFrame) -> pd.Series:
    """(scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))"""
    close = panel["close"]
    vwap = panel["vwap"]
    return scale(ts_sum(close, 7) / 7 - close) + 20 * scale(ts_corr(vwap, delay(close, 5), 230))


def alpha_033(panel: pd.DataFrame) -> pd.Series:
    """rank((-1 * ((1 - (open / close))^1)))"""
    open_ = panel["open"]
    close = panel["close"]
    return rank(-1 * (1 - open_ / close) ** 1)


def alpha_034(panel: pd.DataFrame) -> pd.Series:
    """rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))"""
    close = panel["close"]
    returns = panel["ret"]
    return rank((1 - rank(ts_stddev(returns, 2) / ts_stddev(returns, 5))) + (1 - rank(delta(close, 1))))


def alpha_035(panel: pd.DataFrame) -> pd.Series:
    """((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))"""
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    volume = panel["volume"]
    returns = panel["ret"]
    return ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16)) * (1 - ts_rank(returns, 32))


def alpha_036(panel: pd.DataFrame) -> pd.Series:
    """(((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))"""
    open_ = panel["open"]
    close = panel["close"]
    volume = panel["volume"]
    vwap = panel["vwap"]
    returns = panel["ret"]
    return (
        2.21 * rank(ts_corr(close - open_, delay(volume, 1), 15))
        + 0.7 * rank(open_ - close)
        + 0.73 * rank(ts_rank(delay(-1 * returns, 6), 5))
        + rank(ts_corr(vwap, adv(panel, 20), 6).abs())
        + 0.6 * rank((ts_sum(close, 200) / 200 - open_) * (close - open_))
    )


def alpha_037(panel: pd.DataFrame) -> pd.Series:
    """(rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))"""
    open_ = panel["open"]
    close = panel["close"]
    return rank(ts_corr(delay(open_ - close, 1), close, 200)) + rank(open_ - close)


def alpha_038(panel: pd.DataFrame) -> pd.Series:
    """((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))"""
    open_ = panel["open"]
    close = panel["close"]
    return (-1 * rank(ts_rank(close, 10))) * rank(close / open_)


def alpha_039(panel: pd.DataFrame) -> pd.Series:
    """((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))"""
    close = panel["close"]
    volume = panel["volume"]
    returns = panel["ret"]
    inner = delta(close, 7) * (1 - rank(decay_linear(volume / adv(panel, 20), 9)))
    return (-1 * rank(inner)) * (1 + rank(ts_sum(returns, 250)))


def alpha_040(panel: pd.DataFrame) -> pd.Series:
    """((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))"""
    high = panel["high"]
    volume = panel["volume"]
    return (-1 * rank(ts_stddev(high, 10))) * ts_corr(high, volume, 10)


def alpha_041(panel: pd.DataFrame) -> pd.Series:
    """(((high * low)^0.5) - vwap)"""
    high = panel["high"]
    low = panel["low"]
    vwap = panel["vwap"]
    return (high * low) ** 0.5 - vwap


def alpha_042(panel: pd.DataFrame) -> pd.Series:
    """(rank((vwap - close)) / rank((vwap + close)))"""
    close = panel["close"]
    vwap = panel["vwap"]
    return rank(vwap - close) / rank(vwap + close)


def alpha_043(panel: pd.DataFrame) -> pd.Series:
    """(ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))"""
    close = panel["close"]
    volume = panel["volume"]
    return ts_rank(volume / adv(panel, 20), 20) * ts_rank(-1 * delta(close, 7), 8)


def alpha_044(panel: pd.DataFrame) -> pd.Series:
    """(-1 * correlation(high, rank(volume), 5))"""
    high = panel["high"]
    volume = panel["volume"]
    return -1 * ts_corr(high, rank(volume), 5)


def alpha_045(panel: pd.DataFrame) -> pd.Series:
    """(-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))"""
    close = panel["close"]
    volume = panel["volume"]
    return (
        -1
        * rank(ts_sum(delay(close, 5), 20) / 20)
        * ts_corr(close, volume, 2)
        * rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2))
    )


def alpha_047(panel: pd.DataFrame) -> pd.Series:
    """((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))"""
    high = panel["high"]
    close = panel["close"]
    volume = panel["volume"]
    vwap = panel["vwap"]
    return (
        (rank(1 / close) * volume / adv(panel, 20))
        * (high * rank(high - close) / (ts_sum(high, 5) / 5))
        - rank(vwap - delay(vwap, 5))
    )


def _close_slope(panel: pd.DataFrame) -> pd.Series:
    """Shared driver for alphas 46/49/51 (identical piecewise shape, different thresholds):
    (delay(close,20)-delay(close,10))/10 - (delay(close,10)-close)/10."""
    close = panel["close"]
    return (delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10


def alpha_046(panel: pd.DataFrame) -> pd.Series:
    """((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))"""
    close = panel["close"]
    slope = _close_slope(panel)
    inner = where(slope < 0, 1, -1 * (close - delay(close, 1)))
    return where(0.25 < slope, -1, inner)


def alpha_049(panel: pd.DataFrame) -> pd.Series:
    """(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))"""
    close = panel["close"]
    slope = _close_slope(panel)
    return where(slope < -0.1, 1, -1 * (close - delay(close, 1)))


def alpha_050(panel: pd.DataFrame) -> pd.Series:
    """(-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))"""
    volume = panel["volume"]
    vwap = panel["vwap"]
    return -1 * ts_max(rank(ts_corr(rank(volume), rank(vwap), 5)), 5)


def alpha_051(panel: pd.DataFrame) -> pd.Series:
    """(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))"""
    close = panel["close"]
    slope = _close_slope(panel)
    return where(slope < -0.05, 1, -1 * (close - delay(close, 1)))


def alpha_052(panel: pd.DataFrame) -> pd.Series:
    """((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))"""
    low = panel["low"]
    volume = panel["volume"]
    returns = panel["ret"]
    ts_min_low5 = ts_min(low, 5)
    return (
        (-1 * ts_min_low5 + delay(ts_min_low5, 5))
        * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220)
        * ts_rank(volume, 5)
    )


def alpha_053(panel: pd.DataFrame) -> pd.Series:
    """(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))"""
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    x = ((close - low) - (high - close)) / (close - low + 0.01)
    return -1 * delta(x, 9)


def alpha_054(panel: pd.DataFrame) -> pd.Series:
    """((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))"""
    open_ = panel["open"]
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    return (-1 * (low - close) * open_**5) / ((low - high + 0.01) * close**5)


def alpha_055(panel: pd.DataFrame) -> pd.Series:
    """(-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))"""
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    volume = panel["volume"]
    x = (close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12) + 0.01)
    return -1 * ts_corr(rank(x), rank(volume), 6)


def alpha_056(panel: pd.DataFrame) -> pd.Series:
    """(0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))"""
    returns = panel["ret"]
    cap = panel["mcap"]
    return -1 * rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) * rank(returns * cap)


def alpha_057(panel: pd.DataFrame) -> pd.Series:
    """(0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))"""
    close = panel["close"]
    vwap = panel["vwap"]
    return -1 * (close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)


def alpha_060(panel: pd.DataFrame) -> pd.Series:
    """(0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))"""
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    volume = panel["volume"]
    x = ((close - low) - (high - close)) / (high - low + 0.01) * volume
    return -1 * (2 * scale(rank(x)) - scale(rank(ts_argmax(close, 10))))


def alpha_061(panel: pd.DataFrame) -> pd.Series:
    """(rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))"""
    vwap = panel["vwap"]
    lhs = rank(vwap - ts_min(vwap, 16.1219))
    rhs = rank(ts_corr(vwap, adv(panel, 180), 17.9282))
    return (lhs < rhs).astype(float)


def alpha_062(panel: pd.DataFrame) -> pd.Series:
    """((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)"""
    open_ = panel["open"]
    high = panel["high"]
    low = panel["low"]
    vwap = panel["vwap"]
    inner_cond = ((rank(open_) + rank(open_)) < (rank((high + low) / 2) + rank(high))).astype(float)
    lhs = rank(ts_corr(vwap, ts_sum(adv(panel, 20), 22.4101), 9.91009))
    rhs = rank(inner_cond)
    return (lhs < rhs).astype(float) * -1


def alpha_063(panel: pd.DataFrame) -> pd.Series:
    """(rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1"""
    open_ = panel["open"]
    close = panel["close"]
    vwap = panel["vwap"]
    sic2 = panel["sic2"]
    lhs = rank(decay_linear(delta(indneutralize(close, sic2), 2.25164), 8.22237))
    rhs = rank(
        decay_linear(
            ts_corr(vwap * 0.318108 + open_ * (1 - 0.318108), ts_sum(adv(panel, 180), 37.2467), 13.557),
            12.2883,
        )
    )
    return (lhs - rhs) * -1


def alpha_064(panel: pd.DataFrame) -> pd.Series:
    """((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)"""
    open_ = panel["open"]
    high = panel["high"]
    low = panel["low"]
    vwap = panel["vwap"]
    lhs = rank(
        ts_corr(
            ts_sum(open_ * 0.178404 + low * (1 - 0.178404), 12.7054),
            ts_sum(adv(panel, 120), 12.7054),
            16.6208,
        )
    )
    rhs = rank(delta((high + low) / 2 * 0.178404 + vwap * (1 - 0.178404), 3.69741))
    return (lhs < rhs).astype(float) * -1


def alpha_065(panel: pd.DataFrame) -> pd.Series:
    """((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)"""
    open_ = panel["open"]
    vwap = panel["vwap"]
    lhs = rank(
        ts_corr(open_ * 0.00817205 + vwap * (1 - 0.00817205), ts_sum(adv(panel, 60), 8.6911), 6.40374)
    )
    rhs = rank(open_ - ts_min(open_, 13.635))
    return (lhs < rhs).astype(float) * -1


def alpha_068(panel: pd.DataFrame) -> pd.Series:
    """((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)"""
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    lhs = ts_rank(ts_corr(rank(high), rank(adv(panel, 15)), 8.91644), 13.9333)
    rhs = rank(delta(close * 0.518371 + low * (1 - 0.518371), 1.06157))
    return (lhs < rhs).astype(float) * -1


def alpha_069(panel: pd.DataFrame) -> pd.Series:
    """((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)"""
    close = panel["close"]
    vwap = panel["vwap"]
    sic2 = panel["sic2"]
    base = rank(ts_max(delta(indneutralize(vwap, sic2), 2.72412), 4.79344))
    exponent = ts_rank(ts_corr(close * 0.490655 + vwap * (1 - 0.490655), adv(panel, 20), 4.92416), 9.0615)
    return base**exponent * -1


def alpha_070(panel: pd.DataFrame) -> pd.Series:
    """((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)"""
    close = panel["close"]
    vwap = panel["vwap"]
    sic2 = panel["sic2"]
    base = rank(delta(vwap, 1.29456))
    exponent = ts_rank(ts_corr(indneutralize(close, sic2), adv(panel, 50), 17.8256), 17.9171)
    return base**exponent * -1


def alpha_071(panel: pd.DataFrame) -> pd.Series:
    """max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))"""
    open_ = panel["open"]
    low = panel["low"]
    close = panel["close"]
    vwap = panel["vwap"]
    branch_a = ts_rank(
        decay_linear(ts_corr(ts_rank(close, 3.43976), ts_rank(adv(panel, 180), 12.0647), 18.0175), 4.20501),
        15.6948,
    )
    branch_b = ts_rank(decay_linear(rank((low + open_) - 2 * vwap) ** 2, 16.4662), 4.4388)
    return np.maximum(branch_a, branch_b)


def alpha_072(panel: pd.DataFrame) -> pd.Series:
    """(rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))"""
    high = panel["high"]
    low = panel["low"]
    volume = panel["volume"]
    vwap = panel["vwap"]
    numerator = rank(decay_linear(ts_corr((high + low) / 2, adv(panel, 40), 8.93345), 10.1519))
    denominator = rank(
        decay_linear(ts_corr(ts_rank(vwap, 3.72469), ts_rank(volume, 18.5188), 6.86671), 2.95011)
    )
    return numerator / denominator


def alpha_073(panel: pd.DataFrame) -> pd.Series:
    """(max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)"""
    open_ = panel["open"]
    low = panel["low"]
    vwap = panel["vwap"]
    mix = open_ * 0.147155 + low * (1 - 0.147155)
    branch_a = rank(decay_linear(delta(vwap, 4.72775), 2.91864))
    branch_b = ts_rank(decay_linear(-1 * delta(mix, 2.03608) / mix, 3.33829), 16.7411)
    return np.maximum(branch_a, branch_b) * -1


def alpha_074(panel: pd.DataFrame) -> pd.Series:
    """((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)"""
    high = panel["high"]
    close = panel["close"]
    volume = panel["volume"]
    vwap = panel["vwap"]
    lhs = rank(ts_corr(close, ts_sum(adv(panel, 30), 37.4843), 15.1365))
    rhs = rank(ts_corr(rank(high * 0.0261661 + vwap * (1 - 0.0261661)), rank(volume), 11.4791))
    return (lhs < rhs).astype(float) * -1


def alpha_075(panel: pd.DataFrame) -> pd.Series:
    """(rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))"""
    low = panel["low"]
    volume = panel["volume"]
    vwap = panel["vwap"]
    lhs = rank(ts_corr(vwap, volume, 4.24304))
    rhs = rank(ts_corr(rank(low), rank(adv(panel, 50)), 12.4413))
    return (lhs < rhs).astype(float)


def alpha_077(panel: pd.DataFrame) -> pd.Series:
    """min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))"""
    high = panel["high"]
    low = panel["low"]
    vwap = panel["vwap"]
    branch_a = rank(decay_linear((high + low) / 2 - vwap, 20.0451))
    branch_b = rank(decay_linear(ts_corr((high + low) / 2, adv(panel, 40), 3.1614), 5.64125))
    return np.minimum(branch_a, branch_b)


def alpha_078(panel: pd.DataFrame) -> pd.Series:
    """(rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))"""
    low = panel["low"]
    volume = panel["volume"]
    vwap = panel["vwap"]
    base = rank(
        ts_corr(
            ts_sum(low * 0.352233 + vwap * (1 - 0.352233), 19.7428),
            ts_sum(adv(panel, 40), 19.7428),
            6.83313,
        )
    )
    exponent = rank(ts_corr(rank(vwap), rank(volume), 5.77492))
    return base**exponent


def alpha_080(panel: pd.DataFrame) -> pd.Series:
    """((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)"""
    open_ = panel["open"]
    high = panel["high"]
    sic2 = panel["sic2"]
    mix = open_ * 0.868128 + high * (1 - 0.868128)
    base = rank(np.sign(delta(indneutralize(mix, sic2), 4.04545)))
    exponent = ts_rank(ts_corr(high, adv(panel, 10), 5.11456), 5.53756)
    return base**exponent * -1


def alpha_081(panel: pd.DataFrame) -> pd.Series:
    """((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)"""
    volume = panel["volume"]
    vwap = panel["vwap"]
    inner = rank(rank(ts_corr(vwap, ts_sum(adv(panel, 10), 49.6054), 8.47743)) ** 4)
    lhs = rank(np.log(ts_product(inner, 14.9655)))
    rhs = rank(ts_corr(rank(vwap), rank(volume), 5.07914))
    return (lhs < rhs).astype(float) * -1


def alpha_083(panel: pd.DataFrame) -> pd.Series:
    """((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))"""
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    volume = panel["volume"]
    vwap = panel["vwap"]
    range_ratio = (high - low + 0.01) / (ts_sum(close, 5) / 5)
    return (rank(delay(range_ratio, 2)) * rank(rank(volume))) / (range_ratio / (vwap - close + 0.01))


def alpha_084(panel: pd.DataFrame) -> pd.Series:
    """SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796)). Inf → NaN."""
    close = panel["close"]
    vwap = panel["vwap"]
    result = signedpower(ts_rank(vwap - ts_max(vwap, 15.3217), 20.7127), delta(close, 4.96796))
    return result.replace([np.inf, -np.inf], np.nan)


def alpha_085(panel: pd.DataFrame) -> pd.Series:
    """(rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))"""
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    volume = panel["volume"]
    base = rank(ts_corr(high * 0.876703 + close * (1 - 0.876703), adv(panel, 30), 9.61331))
    exponent = rank(ts_corr(ts_rank((high + low) / 2, 3.70596), ts_rank(volume, 10.1595), 7.11408))
    return base**exponent


def alpha_086(panel: pd.DataFrame) -> pd.Series:
    """((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)"""
    close = panel["close"]
    vwap = panel["vwap"]
    lhs = ts_rank(ts_corr(close, ts_sum(adv(panel, 20), 14.7444), 6.00049), 20.4195)
    rhs = rank(close - vwap)
    return (lhs < rhs).astype(float) * -1


def alpha_087(panel: pd.DataFrame) -> pd.Series:
    """(max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)"""
    close = panel["close"]
    vwap = panel["vwap"]
    sic2 = panel["sic2"]
    branch_a = rank(decay_linear(delta(close * 0.369701 + vwap * (1 - 0.369701), 1.91233), 2.65461))
    branch_b = ts_rank(
        decay_linear(ts_corr(indneutralize(adv(panel, 81), sic2), close, 13.4132).abs(), 4.89768),
        14.4535,
    )
    return np.maximum(branch_a, branch_b) * -1


def alpha_088(panel: pd.DataFrame) -> pd.Series:
    """min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))"""
    open_ = panel["open"]
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    branch_a = rank(decay_linear((rank(open_) + rank(low)) - (rank(high) + rank(close)), 8.06882))
    branch_b = ts_rank(
        decay_linear(ts_corr(ts_rank(close, 8.44728), ts_rank(adv(panel, 60), 20.6966), 8.01266), 6.65053),
        2.61957,
    )
    return np.minimum(branch_a, branch_b)


def alpha_092(panel: pd.DataFrame) -> pd.Series:
    """min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))"""
    open_ = panel["open"]
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    cond = ((high + low) / 2 + close < low + open_).astype(float)
    branch_a = ts_rank(decay_linear(cond, 14.7221), 18.8683)
    branch_b = ts_rank(decay_linear(ts_corr(rank(low), rank(adv(panel, 30)), 7.58555), 6.94024), 6.80584)
    return np.minimum(branch_a, branch_b)


def alpha_091(panel: pd.DataFrame) -> pd.Series:
    """((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)"""
    close = panel["close"]
    volume = panel["volume"]
    vwap = panel["vwap"]
    sic2 = panel["sic2"]
    lhs = ts_rank(
        decay_linear(decay_linear(ts_corr(indneutralize(close, sic2), volume, 9.74928), 16.398), 3.83219),
        4.8667,
    )
    rhs = rank(decay_linear(ts_corr(vwap, adv(panel, 30), 4.01303), 2.6809))
    return (lhs - rhs) * -1


def alpha_093(panel: pd.DataFrame) -> pd.Series:
    """(Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))"""
    close = panel["close"]
    vwap = panel["vwap"]
    sic2 = panel["sic2"]
    numerator = ts_rank(
        decay_linear(ts_corr(indneutralize(vwap, sic2), adv(panel, 81), 17.4193), 19.848), 7.54455
    )
    denominator = rank(decay_linear(delta(close * 0.524434 + vwap * (1 - 0.524434), 2.77377), 16.2664))
    return numerator / denominator


def alpha_094(panel: pd.DataFrame) -> pd.Series:
    """((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)"""
    vwap = panel["vwap"]
    base = rank(vwap - ts_min(vwap, 11.5783))
    exponent = ts_rank(ts_corr(ts_rank(vwap, 19.6462), ts_rank(adv(panel, 60), 4.02992), 18.0926), 2.70756)
    return base**exponent * -1


def alpha_095(panel: pd.DataFrame) -> pd.Series:
    """(rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))"""
    open_ = panel["open"]
    high = panel["high"]
    low = panel["low"]
    lhs = rank(open_ - ts_min(open_, 12.4105))
    inner = rank(ts_corr(ts_sum((high + low) / 2, 19.1351), ts_sum(adv(panel, 40), 19.1351), 12.8742)) ** 5
    rhs = ts_rank(inner, 11.7584)
    return (lhs < rhs).astype(float)


def alpha_096(panel: pd.DataFrame) -> pd.Series:
    """(max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)"""
    close = panel["close"]
    volume = panel["volume"]
    vwap = panel["vwap"]
    branch_a = ts_rank(decay_linear(ts_corr(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151)
    branch_b = ts_rank(
        decay_linear(
            ts_argmax(ts_corr(ts_rank(close, 7.45404), ts_rank(adv(panel, 60), 4.13242), 3.65459), 12.6556),
            14.0365,
        ),
        13.4143,
    )
    return np.maximum(branch_a, branch_b) * -1


def alpha_097(panel: pd.DataFrame) -> pd.Series:
    """((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)"""
    low = panel["low"]
    vwap = panel["vwap"]
    sic2 = panel["sic2"]
    mix = low * 0.721001 + vwap * (1 - 0.721001)
    lhs = rank(decay_linear(delta(indneutralize(mix, sic2), 3.3705), 20.4523))
    rhs = ts_rank(
        decay_linear(
            ts_rank(ts_corr(ts_rank(low, 7.87871), ts_rank(adv(panel, 60), 17.255), 4.97547), 18.5925),
            15.7152,
        ),
        6.71659,
    )
    return (lhs - rhs) * -1


def alpha_098(panel: pd.DataFrame) -> pd.Series:
    """(rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))"""
    open_ = panel["open"]
    vwap = panel["vwap"]
    lhs = rank(decay_linear(ts_corr(vwap, ts_sum(adv(panel, 5), 26.4719), 4.58418), 7.18088))
    rhs = rank(
        decay_linear(
            ts_rank(ts_argmin(ts_corr(rank(open_), rank(adv(panel, 15)), 20.8187), 8.62571), 6.95668),
            8.07206,
        )
    )
    return lhs - rhs


def alpha_099(panel: pd.DataFrame) -> pd.Series:
    """((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)"""
    high = panel["high"]
    low = panel["low"]
    volume = panel["volume"]
    lhs = rank(ts_corr(ts_sum((high + low) / 2, 19.8975), ts_sum(adv(panel, 60), 19.8975), 8.8136))
    rhs = rank(ts_corr(low, volume, 6.28259))
    return (lhs < rhs).astype(float) * -1


def alpha_101(panel: pd.DataFrame) -> pd.Series:
    """((close - open) / ((high - low) + .001))"""
    open_ = panel["open"]
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    return (close - open_) / (high - low + 0.001)


ALPHAS = {
    "alpha_001": alpha_001,
    "alpha_002": alpha_002,
    "alpha_003": alpha_003,
    "alpha_004": alpha_004,
    "alpha_005": alpha_005,
    "alpha_006": alpha_006,
    "alpha_007": alpha_007,
    "alpha_008": alpha_008,
    "alpha_009": alpha_009,
    "alpha_010": alpha_010,
    "alpha_011": alpha_011,
    "alpha_012": alpha_012,
    "alpha_013": alpha_013,
    "alpha_014": alpha_014,
    "alpha_015": alpha_015,
    "alpha_016": alpha_016,
    "alpha_017": alpha_017,
    "alpha_018": alpha_018,
    "alpha_019": alpha_019,
    "alpha_020": alpha_020,
    "alpha_021": alpha_021,
    "alpha_022": alpha_022,
    "alpha_023": alpha_023,
    "alpha_024": alpha_024,
    "alpha_025": alpha_025,
    "alpha_026": alpha_026,
    "alpha_027": alpha_027,
    "alpha_028": alpha_028,
    "alpha_029": alpha_029,
    "alpha_030": alpha_030,
    "alpha_031": alpha_031,
    "alpha_032": alpha_032,
    "alpha_033": alpha_033,
    "alpha_034": alpha_034,
    "alpha_035": alpha_035,
    "alpha_036": alpha_036,
    "alpha_037": alpha_037,
    "alpha_038": alpha_038,
    "alpha_039": alpha_039,
    "alpha_040": alpha_040,
    "alpha_041": alpha_041,
    "alpha_042": alpha_042,
    "alpha_043": alpha_043,
    "alpha_044": alpha_044,
    "alpha_045": alpha_045,
    "alpha_046": alpha_046,
    "alpha_047": alpha_047,
    "alpha_049": alpha_049,
    "alpha_050": alpha_050,
    "alpha_051": alpha_051,
    "alpha_052": alpha_052,
    "alpha_053": alpha_053,
    "alpha_054": alpha_054,
    "alpha_055": alpha_055,
    "alpha_056": alpha_056,
    "alpha_057": alpha_057,
    "alpha_060": alpha_060,
    "alpha_061": alpha_061,
    "alpha_062": alpha_062,
    "alpha_063": alpha_063,
    "alpha_064": alpha_064,
    "alpha_065": alpha_065,
    "alpha_068": alpha_068,
    "alpha_069": alpha_069,
    "alpha_070": alpha_070,
    "alpha_071": alpha_071,
    "alpha_072": alpha_072,
    "alpha_073": alpha_073,
    "alpha_074": alpha_074,
    "alpha_075": alpha_075,
    "alpha_077": alpha_077,
    "alpha_078": alpha_078,
    "alpha_080": alpha_080,
    "alpha_081": alpha_081,
    "alpha_083": alpha_083,
    "alpha_084": alpha_084,
    "alpha_085": alpha_085,
    "alpha_086": alpha_086,
    "alpha_087": alpha_087,
    "alpha_088": alpha_088,
    "alpha_091": alpha_091,
    "alpha_092": alpha_092,
    "alpha_093": alpha_093,
    "alpha_094": alpha_094,
    "alpha_095": alpha_095,
    "alpha_096": alpha_096,
    "alpha_097": alpha_097,
    "alpha_098": alpha_098,
    "alpha_099": alpha_099,
    "alpha_101": alpha_101,
}


def build_alpha_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Run every alpha in ALPHAS. `panel` must already be prepare_panel()'d."""
    return pd.DataFrame({name: fn(panel) for name, fn in ALPHAS.items()}, index=panel.index)


if __name__ == "__main__":
    from src.data import load_panel
    from src.signals.inputs import prepare_panel

    raw = load_panel(top_n=500, lookback_years=3)
    panel = prepare_panel(raw)

    features = build_alpha_features(panel)
    print(features.describe())
    print("\nall-NaN columns:", features.columns[features.isna().all()].tolist())
    print(
        "\ninf columns:",
        [c for c in features.columns if np.isinf(features[c].to_numpy(dtype=float)).any()],
    )
