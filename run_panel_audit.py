"""Panel hygiene audit. Read-only: no model refit, no interpolation.

Run from the repo root:
    uv run python run_panel_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

from src.data import (
    DEFAULT_MAX_GAP_DAYS,
    build_forward_return_targets,
    compute_segment_id,
    load_panel,
)
from src.signals.alphas import alpha_007
from src.signals.inputs import prepare_panel

OUT_PATH = Path("outputs") / "panel_audit.json"


def _share(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else float("nan")


def _abs_diff_stats(a: pd.Series, b: pd.Series) -> dict[str, float]:
    d = (a - b).abs()
    d = d[d.notna()]
    return {
        "n": int(d.shape[0]),
        "median_abs": float(d.median()) if len(d) else float("nan"),
        "p99_abs": float(d.quantile(0.99)) if len(d) else float("nan"),
        "share_lt_1e8": _share(d < 1e-8),
        "share_lt_1e4": _share(d < 1e-4),
    }


def main() -> None:
    raw = load_panel(top_n=None, require_price_data=True)
    raw = raw.sort_values(["permno", "date"]).reset_index(drop=True)

    n_rows = len(raw)
    n_names = int(raw["permno"].nunique())
    n_dates = int(raw["date"].nunique())
    names_per_day = raw.groupby("date")["permno"].nunique()

    missing = {c: float(raw[c].isna().mean()) for c in raw.columns}

    dup_keys = int(raw.duplicated(subset=["permno", "date"]).sum())

    complete = raw[["open", "high", "low", "close"]].notna().all(axis=1)
    high_lt_body = complete & (raw["high"] < raw[["open", "close"]].max(axis=1))
    low_gt_body = complete & (raw["low"] > raw[["open", "close"]].min(axis=1))
    nonpos_close = raw["close"] <= 0
    neg_volume = raw["volume"] < 0

    raw["segment_id"] = compute_segment_id(raw, max_gap_days=DEFAULT_MAX_GAP_DAYS)
    gap_days = raw.groupby("permno", sort=False)["date"].diff().dt.days
    n_segments = int((raw.groupby("permno")["segment_id"].max() + 1).sum())
    long_gaps = int((gap_days > DEFAULT_MAX_GAP_DAYS).sum())
    gap_hist = (
        gap_days.dropna().clip(upper=60).astype(int).value_counts().sort_index().head(15).to_dict()
    )
    gap_hist = {str(k): int(v) for k, v in gap_hist.items()}

    # Backward identity: CRSP ret vs close-to-close. Dividends make these differ.
    prev_close = raw.groupby("permno", sort=False)["close"].shift(1)
    prev_seg = raw.groupby("permno", sort=False)["segment_id"].shift(1)
    same_seg_back = prev_seg == raw["segment_id"]
    px_ret_back = (raw["close"] / prev_close - 1.0).where(same_seg_back)
    ret_vs_px = _abs_diff_stats(raw["ret"], px_ret_back)

    with_tgt = build_forward_return_targets(raw.drop(columns="segment_id"), horizons=(1, 5))
    with_tgt["segment_id"] = compute_segment_id(with_tgt, max_gap_days=DEFAULT_MAX_GAP_DAYS)
    grouped = with_tgt.groupby("permno", sort=False)
    next_ret = grouped["ret"].shift(-1)
    next_seg = grouped["segment_id"].shift(-1)
    same_seg_fwd = next_seg == with_tgt["segment_id"]
    fwd1_vs_next_ret = _abs_diff_stats(
        with_tgt["fwd_ret_1d"].where(same_seg_fwd),
        next_ret.where(same_seg_fwd),
    )
    fwd1_vs_px = _abs_diff_stats(
        with_tgt["fwd_ret_1d"],
        (grouped["close"].shift(-1) / with_tgt["close"] - 1.0).where(same_seg_fwd),
    )
    same_seg_5 = grouped["segment_id"].shift(-5) == with_tgt["segment_id"]
    fwd5_vs_px = _abs_diff_stats(
        with_tgt["fwd_ret_5d"],
        (grouped["close"].shift(-5) / with_tgt["close"] - 1.0).where(same_seg_5),
    )

    target_nan = {
        "fwd_ret_1d": float(with_tgt["fwd_ret_1d"].isna().mean()),
        "fwd_ret_5d": float(with_tgt["fwd_ret_5d"].isna().mean()),
    }

    panel = prepare_panel(raw.drop(columns="segment_id"))
    a7 = alpha_007(panel)
    a7_year = a7.rename("value").reset_index()
    a7_year["year"] = pd.to_datetime(a7_year["date"]).dt.year
    share_minus_one = float((a7 == -1).mean())
    nunique_by_year = a7_year.groupby("year")["value"].nunique()
    share_days_constant = float((a7.groupby(level="date").nunique() == 1).mean())

    out = {
        "n_rows": n_rows,
        "n_permno": n_names,
        "n_dates": n_dates,
        "date_min": str(raw["date"].min().date()),
        "date_max": str(raw["date"].max().date()),
        "names_per_day_min": int(names_per_day.min()),
        "names_per_day_median": float(names_per_day.median()),
        "names_per_day_max": int(names_per_day.max()),
        "dup_permno_date": dup_keys,
        "missing_share": missing,
        "nonpos_close_share": _share(nonpos_close),
        "neg_volume_share": _share(neg_volume),
        "ohlc_high_lt_body_share": _share(high_lt_body),
        "ohlc_low_gt_body_share": _share(low_gt_body),
        "n_segments": n_segments,
        "n_long_gaps": long_gaps,
        "gap_hist_head": gap_hist,
        "ret_vs_price_return": ret_vs_px,
        "fwd_ret_1d_vs_next_ret": fwd1_vs_next_ret,
        "fwd_ret_1d_vs_close_ratio": fwd1_vs_px,
        "fwd_ret_5d_vs_close_ratio": fwd5_vs_px,
        "target_nan_share": target_nan,
        "alpha_007": {
            "share_eq_minus_1": share_minus_one,
            "nunique_total": int(a7.nunique(dropna=True)),
            "share_dates_cross_section_constant": float(share_days_constant),
            "nunique_by_year_min": int(nunique_by_year.min()),
            "nunique_by_year_median": float(nunique_by_year.median()),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(obj=out, indent=2))

    print("=== panel ===")
    print(
        f"rows {n_rows:,}  permno {n_names:,}  dates {n_dates:,}  "
        f"{out['date_min']} → {out['date_max']}"
    )
    print(
        f"names/day min/med/max {out['names_per_day_min']}/"
        f"{out['names_per_day_median']:.0f}/{out['names_per_day_max']}"
    )
    print(f"dup (permno,date) {dup_keys}")
    print("missing (nonzero):")
    for c, s in missing.items():
        if s > 0:
            print(f"  {c}: {s:.4%}")
    print(
        f"OHLC high<body {out['ohlc_high_lt_body_share']:.4%}  "
        f"low>body {out['ohlc_low_gt_body_share']:.4%}  "
        f"close<=0 {out['nonpos_close_share']:.4%}"
    )
    print(f"segments {n_segments:,}  gaps>{DEFAULT_MAX_GAP_DAYS}d {long_gaps:,}")
    print("=== identities ===")
    print(f"ret vs close/close[-1]-1     {ret_vs_px}")
    print(f"fwd_1d vs next ret           {fwd1_vs_next_ret}")
    print(f"fwd_1d vs close[t+1]/close-1 {fwd1_vs_px}")
    print(f"fwd_5d vs close[t+5]/close-1 {fwd5_vs_px}")
    print(f"target NaN share {target_nan}")
    print("=== alpha_007 ===")
    print(out["alpha_007"])
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
