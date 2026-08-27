"""Fixed 3-feature baseline + pairwise interactions. Do not change.

Built from `close` (split-adjusted), not `ret` — `ret` after a re-entry
gap is a stale multi-month return. `rev_*` are negated past returns
(reversal); `mom_12_1` keeps its raw sign (momentum). Interactions are
products of per-date ranks in (-1, 1]; the three base columns stay in
raw return units.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from src.signals.cross_sectional import rank as xs_rank
from src.signals.operators import delay


def rev_1d(close: pd.Series) -> pd.Series:
    """Negative 1-day return: -(close / delay(close, 1) - 1)."""
    return -(close / delay(close, 1) - 1.0)


def rev_5d(close: pd.Series) -> pd.Series:
    """Negative 5-day return: -(close / delay(close, 5) - 1)."""
    return -(close / delay(close, 5) - 1.0)


def mom_12_1(close: pd.Series) -> pd.Series:
    """Past ~12-month return excluding the most recent ~21 trading days."""
    return delay(close, 21) / delay(close, 252) - 1.0


def _normalize_for_interaction(x: pd.Series) -> pd.Series:
    """Per-date rank rescaled from (0, 1] to (-1, 1]."""
    return 2.0 * xs_rank(x) - 1.0


def build_baseline_features(panel: pd.DataFrame) -> pd.DataFrame:
    """rev_1d, rev_5d, mom_12_1 plus 3 pairwise interactions. Needs prepare_panel()."""
    close = panel["close"]

    base = {
        "rev_1d": rev_1d(close),
        "rev_5d": rev_5d(close),
        "mom_12_1": mom_12_1(close),
    }

    normalized = {name: _normalize_for_interaction(x) for name, x in base.items()}

    interactions = {
        "rev_1d_x_rev_5d": normalized["rev_1d"] * normalized["rev_5d"],
        "rev_1d_x_mom_12_1": normalized["rev_1d"] * normalized["mom_12_1"],
        "rev_5d_x_mom_12_1": normalized["rev_5d"] * normalized["mom_12_1"],
    }

    return pd.DataFrame({"close": close, **base, **interactions}, index=panel.index)


if __name__ == "__main__":
    from src.data import load_panel
    from src.signals.inputs import prepare_panel

    raw = load_panel(top_n=500, lookback_years=3)
    panel = prepare_panel(raw)

    features = build_baseline_features(panel)
    print(features.describe())
    print("\nany all-NaN columns:", features.columns[features.isna().all()].tolist())
    print("any inf values:", pd.Series({c: bool((features[c].abs() == float("inf")).any()) for c in features}))
    print("\nsample rows:\n", features.dropna().head(10))
    features.reset_index().to_clipboard()
