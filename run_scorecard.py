"""One-shot runner: discovers every saved prediction file across elastic net,
normalization ablation, and boosted tree runs, scores each with
src/scorecard.py, and produces the final consolidated table + figures for
the report.

Run from the repo root, AFTER the relevant run scripts (elastic net, boosted
trees, and normalization ablation if you want its runs included) have
already produced their preds parquets:
    uv run python run_scorecard.py

Output lands under outputs/scorecard/:
    scorecard_table.csv          -- the one tidy table for the report
    stress_windows.csv           -- per-run stress-window breakdown
    cumulative_long_short_{h}d.png
    ic_by_year_{h}d.png
    decile_monotonicity_{h}d.png
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd

from src.scorecard import build_scorecard_table, plot_cumulative_long_short, plot_decile_monotonicity, plot_ic_by_year, score_one_run

OUTPUT_DIR = Path("outputs/scorecard")

# (label prefix, directory, glob-relative horizon parse). Each entry is
# scanned for *__{h}d__preds.parquet files; the config name is whatever
# precedes "__{h}d__preds.parquet" in the filename.
SOURCES = [
    ("elastic_net", Path("outputs/model_runs")),
    ("xgboost", Path("outputs/model_runs_trees")),
    ("elastic_net_rank_ablation", Path("outputs/ablation_normalization/rank")),
    ("elastic_net_standardize_ablation", Path("outputs/ablation_normalization/standardize")),
]

PREDS_PATTERN = re.compile(r"^(?P<config>.+)__(?P<horizon>\d+)d__preds\.parquet$")


def discover_runs() -> dict[str, tuple[Path, int]]:
    """{run_label: (preds_path, horizon)} for every preds file found."""
    found = {}
    for prefix, directory in SOURCES:
        if not directory.exists():
            print(f"  (skipping {directory} -- not found)")
            continue
        for path in sorted(directory.glob("*__preds.parquet")):
            m = PREDS_PATTERN.match(path.name)
            if not m:
                print(f"  warning: {path.name} doesn't match the expected naming pattern, skipping")
                continue
            config, horizon = m.group("config"), int(m.group("horizon"))
            label = f"{prefix}__{config}__{horizon}d"
            found[label] = (path, horizon)
    return found


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Discovering saved runs ===")
    run_files = discover_runs()
    for label, (path, horizon) in run_files.items():
        print(f"  {label}  <-  {path}")
    if not run_files:
        print("No preds files found under any of:", [str(d) for _, d in SOURCES])
        print("Run run_models.py and/or the tree-model run script first.")
        return

    print(f"\n=== Scoring {len(run_files)} runs ===")
    scored_runs = {}
    for label, (path, horizon) in run_files.items():
        preds = pd.read_parquet(path)
        scored_runs[label] = score_one_run(preds, horizon=horizon)
        stats = scored_runs[label]["ic_stats"]
        print(f"  {label}: mean_ic={stats['mean_ic']:.4f}  ir={stats['ir']:.3f}  n_years={stats['n_years']}")

    print("\n=== Building final tidy table ===")
    table = build_scorecard_table(scored_runs)
    print(table.round(4).to_string())
    table_path = OUTPUT_DIR / "scorecard_table.csv"
    table.to_csv(table_path)
    print(f"saved -> {table_path}")

    stress_rows = []
    for label, scored in scored_runs.items():
        sw = scored["stress_windows"].copy()
        sw.insert(0, "run", label)
        stress_rows.append(sw.reset_index())
    stress_table = pd.concat(stress_rows, ignore_index=True)
    stress_path = OUTPUT_DIR / "stress_windows.csv"
    stress_table.to_csv(stress_path, index=False)
    print(f"saved -> {stress_path}")

    print("\n=== Building figures (split by horizon for readability) ===")
    horizons = sorted({h for _, h in run_files.values()})
    for h in horizons:
        runs_h = {label: scored for label, scored in scored_runs.items() if label.endswith(f"__{h}d")}
        if not runs_h:
            continue

        fig1 = plot_cumulative_long_short(runs_h, title=f"Cumulative long-short growth of $1 ({h}d horizon)")
        fig1_path = OUTPUT_DIR / f"cumulative_long_short_{h}d.png"
        fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")

        fig2 = plot_ic_by_year(runs_h, title=f"Mean daily Spearman IC by year ({h}d horizon)")
        fig2_path = OUTPUT_DIR / f"ic_by_year_{h}d.png"
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")

        fig3 = plot_decile_monotonicity(runs_h, title=f"Mean return by decile ({h}d horizon)")
        fig3_path = OUTPUT_DIR / f"decile_monotonicity_{h}d.png"
        fig3.savefig(fig3_path, dpi=150, bbox_inches="tight")

        print(f"  {h}d: saved 3 figures ({len(runs_h)} runs on each)")

    print(f"\n=== Done. Everything is under {OUTPUT_DIR}/ ===")


if __name__ == "__main__":
    main()
