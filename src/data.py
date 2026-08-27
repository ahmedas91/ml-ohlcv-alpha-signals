"""OHLCV loader and forward-return targets.

Join/group on `permno`, never `ticker`. `ret` is backward-looking — the
only legal forward return is `build_forward_return_targets`. Date gaps
are bimodal (weekend/holiday ≤7d, real exit/re-entry ≥29d), so a 10-day
cutoff (`DEFAULT_MAX_GAP_DAYS`) splits those cases.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "olhcv_merged.parquet"

DEFAULT_MAX_GAP_DAYS = 10  # weekend/holiday vs real exit/re-entry; see module docstring

_REQUIRED_COLUMNS = {
    "permno", "ticker", "date", "ret", "mcap", "turnover",
    "sic2", "volume", "open", "high", "low", "close", "shrout",
}


def load_panel(
    path: str | Path = DEFAULT_DATA_PATH,
    *,
    top_n: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    lookback_years: int | None = None,
    require_price_data: bool = True,
) -> pd.DataFrame:
    """Load olhcv_merged.parquet, optionally restricted.

    `top_n` optionally keeps the top names per date by `mcap` (scratch
    runs). Default None is the full universe — a daily top-N cut punches
    lookback gaps and drops small caps. `start_year` wins over
    `lookback_years` if both are set. Missing `open` is left as-is;
    missing `mcap`/`close` are dropped when `require_price_data` is True.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find OHLCV parquet at {path}. Pass an explicit "
            "`path=` or drop the file at that location."
        )

    df = pd.read_parquet(path)

    missing_cols = _REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Loaded panel is missing expected columns: {sorted(missing_cols)}")

    df["date"] = pd.to_datetime(df["date"])

    if require_price_data:
        before = len(df)
        df = df.dropna(subset=["mcap", "close"])
        dropped = before - len(df)
        if dropped:
            print(f"load_panel: dropped {dropped:,} rows missing mcap/close "
                  f"({dropped / before:.3%} of panel)")

    if start_year is None and lookback_years is not None:
        start_year = int(df["date"].dt.year.max()) - lookback_years + 1

    if start_year is not None:
        df = df[df["date"].dt.year >= start_year]
    if end_year is not None:
        df = df[df["date"].dt.year <= end_year]

    if top_n is not None:
        rank = (
            df.groupby("date")["mcap"]
            .rank(method="first", ascending=False)
        )
        df = df[rank <= top_n]

    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    return df


def compute_segment_id(
    df: pd.DataFrame,
    *,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> pd.Series:
    """Per-permno id that increments on a gap > `max_gap_days`.

    `df` must already be sorted by (permno, date).
    """
    grouped_date = df.groupby("permno", sort=False)["date"]
    step_gap_days = grouped_date.diff().dt.days
    new_segment = (step_gap_days > max_gap_days).fillna(False)
    return new_segment.groupby(df["permno"]).cumsum()


def build_forward_return_targets(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5),
    *,
    price_col: str = "close",
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> pd.DataFrame:
    """Add `fwd_ret_{h}d` = close[t+h]/close[t] - 1, NaN across a segment gap."""
    out = df.sort_values(["permno", "date"]).reset_index(drop=True)
    grouped = out.groupby("permno", sort=False)

    out["_segment_id"] = compute_segment_id(out, max_gap_days=max_gap_days)

    price = out[price_col]
    for h in horizons:
        fwd_price = grouped[price_col].shift(-h)
        fwd_segment = grouped["_segment_id"].shift(-h)
        same_segment = fwd_segment == out["_segment_id"]
        target = fwd_price / price - 1.0
        out[f"fwd_ret_{h}d"] = target.where(same_segment)

    out = out.drop(columns="_segment_id")
    return out


if __name__ == "__main__":
    panel = load_panel(top_n=500, lookback_years=10)
    print(panel.shape, panel["date"].min(), panel["date"].max())
    panel = build_forward_return_targets(panel, horizons=(1, 5))
    print(panel[["permno", "date", "close", "fwd_ret_1d", "fwd_ret_5d"]].head(10))
