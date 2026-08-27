
"""
Run from the repo root:
    uv run python run_models.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from src.models import build_model_frame_chunked, coefficient_comparison, run_suite

START_YEAR = 2011
END_YEAR = 2025
MIN_TEST_YEAR = 2013  # first test year 
VAL_YEARS = 1
PURGE_DAYS = 5
HORIZONS = (1, 5)

OUTPUT_DIR = Path("outputs")


def main() -> None:
    print(f"=== Building full-universe features, {START_YEAR}-{END_YEAR} ===")
    t0 = time.time()
    frame = build_model_frame_chunked(
        top_n=None,
        start_year=START_YEAR,
        end_year=END_YEAR,
        warmup_years=2,
        horizons=HORIZONS,
    )
    print(f"Feature build done in {time.time() - t0:.1f}s, shape {frame.shape}\n")

    print(f"=== Running elastic-net suite: 4 configs x {len(HORIZONS)} horizons ===")
    t0 = time.time()
    results = run_suite(
        frame,
        min_test_year=MIN_TEST_YEAR,
        max_test_year=END_YEAR,
        val_years=VAL_YEARS,
        purge_days=PURGE_DAYS,
        horizons=HORIZONS,
    )
    print(f"\nFull suite done in {time.time() - t0:.1f}s\n")

    print("=== Per-config, per-year IC (corr(y, yhat)) ===")
    for name, out in results.items():
        preds = out["preds"]
        ic = preds.groupby("test_year").apply(lambda g: g["y"].corr(g["yhat"]))
        print(f"\n{name}:")
        print(ic.round(4).to_string())

    print("\n=== Coefficient comparison: correlation filter (c) vs elastic net's own selection (d) ===")
    for h in HORIZONS:
        table = coefficient_comparison(results, horizon=h)
        agreement = (table["kept_by_correlation_filter"] == table["kept_by_elastic_net_any_fold"]).mean()
        print(f"\n--- horizon {h}d --- (agreement rate: {agreement:.1%})")
        print(table.to_string())
        out_path = OUTPUT_DIR / f"coefficient_comparison_{h}d.csv"
        table.to_csv(out_path)
        print(f"saved -> {out_path}")

    print("\n=== Done. Predictions/coefs/params are in outputs/model_runs/, ===")
    print("=== per-year feature caches are in outputs/features_by_year/. ===")


if __name__ == "__main__":
    main()
