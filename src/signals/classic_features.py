"""15 classic OHLCV features spanning price shape, momentum, volatility, and liquidity."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

from src.signals.cross_sectional import rank
from src.signals.inputs import adv
from src.signals.operators import delay, ts_max, ts_mean, ts_stddev

# --- Price shape ---


def oc_ret(panel: pd.DataFrame) -> pd.Series:
    """close/open - 1."""
    return panel["close"] / panel["open"] - 1.0


def hl_range(panel: pd.DataFrame) -> pd.Series:
    """(high-low)/close."""
    return (panel["high"] - panel["low"]) / panel["close"]


def overnight_gap(panel: pd.DataFrame) -> pd.Series:
    """open/delay(close,1) - 1."""
    return panel["open"] / delay(panel["close"], 1) - 1.0


def close_loc(panel: pd.DataFrame) -> pd.Series:
    """(close-low)/(high-low+0.01)."""
    return (panel["close"] - panel["low"]) / (panel["high"] - panel["low"] + 0.01)


# --- Reversal/momentum ---


def ret_21d(panel: pd.DataFrame) -> pd.Series:
    """close/delay(close,21) - 1."""
    return panel["close"] / delay(panel["close"], 21) - 1.0


def mom_63(panel: pd.DataFrame) -> pd.Series:
    """close/delay(close,63) - 1."""
    return panel["close"] / delay(panel["close"], 63) - 1.0


def mom_6_1(panel: pd.DataFrame) -> pd.Series:
    """delay(close,21)/delay(close,126) - 1."""
    close = panel["close"]
    return delay(close, 21) / delay(close, 126) - 1.0


# --- Volatility ---


def vol_21d(panel: pd.DataFrame) -> pd.Series:
    """ts_stddev(ret, 21)."""
    return ts_stddev(panel["ret"], 21)


def parkinson_vol_21d(panel: pd.DataFrame) -> pd.Series:
    """sqrt(ts_mean((ln(high/low))^2, 21) / (4*ln(2)))."""
    log_hl_sq = np.log(panel["high"] / panel["low"]) ** 2
    return np.sqrt(ts_mean(log_hl_sq, 21) / (4.0 * np.log(2.0)))


def vol_of_vol_63d(panel: pd.DataFrame) -> pd.Series:
    """ts_stddev(vol_21d, 63)."""
    return ts_stddev(vol_21d(panel), 63)


# --- Volume/liquidity ---


def vol_vs_mean21(panel: pd.DataFrame) -> pd.Series:
    """volume/ts_mean(volume,21) - 1."""
    volume = panel["volume"]
    return volume / ts_mean(volume, 21) - 1.0


def dollar_vol_rank(panel: pd.DataFrame) -> pd.Series:
    """rank(adv(panel, 21))."""
    return rank(adv(panel, 21))


def amihud_21d(panel: pd.DataFrame) -> pd.Series:
    """ts_mean(|ret|/(close*volume), 21). Zero volume → NaN."""
    dollar_volume = (panel["close"] * panel["volume"]).replace(0, np.nan)
    return ts_mean(panel["ret"].abs() / dollar_volume, 21)


# --- Trend ---


def dist_52wk_high(panel: pd.DataFrame) -> pd.Series:
    """close/ts_max(close,252) - 1."""
    close = panel["close"]
    return close / ts_max(close, 252) - 1.0


def max_ret_21d(panel: pd.DataFrame) -> pd.Series:
    """ts_max(ret, 21)."""
    return ts_max(panel["ret"], 21)


CLASSIC_FEATURES = {
    "oc_ret": oc_ret,
    "hl_range": hl_range,
    "overnight_gap": overnight_gap,
    "close_loc": close_loc,
    "ret_21d": ret_21d,
    "mom_63": mom_63,
    "mom_6_1": mom_6_1,
    "vol_21d": vol_21d,
    "parkinson_vol_21d": parkinson_vol_21d,
    "vol_of_vol_63d": vol_of_vol_63d,
    "vol_vs_mean21": vol_vs_mean21,
    "dollar_vol_rank": dollar_vol_rank,
    "amihud_21d": amihud_21d,
    "dist_52wk_high": dist_52wk_high,
    "max_ret_21d": max_ret_21d,
}


def build_classic_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Run every feature in CLASSIC_FEATURES. Needs prepare_panel()."""
    return pd.DataFrame({name: fn(panel) for name, fn in CLASSIC_FEATURES.items()}, index=panel.index)


if __name__ == "__main__":
    from src.data import load_panel
    from src.signals.inputs import prepare_panel

    raw = load_panel(top_n=None, lookback_years=2)
    panel = prepare_panel(raw)

    features = build_classic_features(panel)
    print(features.describe())
    print("\nall-NaN columns:", features.columns[features.isna().all()].tolist())
    print(
        "\ninf columns:",
        [c for c in features.columns if np.isinf(features[c].to_numpy(dtype=float)).any()],
    )
