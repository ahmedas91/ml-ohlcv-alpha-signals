"""Panel prep: `prepare_panel` before any operator.

Indexes by `(permno, segment_id, date)` and adds `vwap = (high+low+close)/3`
(no intraday data, so this is a daily-bar proxy, not a real VWAP).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from src.data import DEFAULT_MAX_GAP_DAYS, compute_segment_id
from src.signals.operators import ts_mean

_INDEX_LEVELS = ["permno", "segment_id", "date"]


def prepare_panel(panel: pd.DataFrame, *, max_gap_days: int = DEFAULT_MAX_GAP_DAYS) -> pd.DataFrame:
    """Sort, attach segment_id + vwap, index by (permno, segment_id, date)."""
    out = panel.sort_values(["permno", "date"]).reset_index(drop=True)
    out["segment_id"] = compute_segment_id(out, max_gap_days=max_gap_days)
    out["vwap"] = (out["high"] + out["low"] + out["close"]) / 3
    return out.set_index(_INDEX_LEVELS)


def adv(panel: pd.DataFrame, d: float) -> pd.Series:
    """Average daily dollar volume over the past d days: ts_mean(volume * close, d)."""
    key = math.floor(d)
    cache = panel.attrs.setdefault("_adv_cache", {})
    if key not in cache:
        cache[key] = ts_mean(panel["volume"] * panel["close"], key)
    return cache[key]


if __name__ == "__main__":
    from src.data import load_panel
    from src.signals.cross_sectional import rank as xs_rank
    from src.signals.cross_sectional import zscore
    from src.signals.operators import decay_linear, delta, ts_rank

    raw = load_panel(top_n=500, lookback_years=5)
    panel = prepare_panel(raw)

    close = panel["close"]
    print("delta(close, 1):\n", delta(close, 1).dropna().head())
    print("\nts_rank(close, 10):\n", ts_rank(close, 10).dropna().head())
    print("\ndecay_linear(close, 5):\n", decay_linear(close, 5).dropna().head())
    print("\ncross-sectional rank(close):\n", xs_rank(close).head())
    print("\ncross-sectional zscore(close):\n", zscore(close).head())
    print("\nadv(20):\n", adv(panel, 20).dropna().head())
    print("\nvwap:\n", panel["vwap"].dropna().head())
