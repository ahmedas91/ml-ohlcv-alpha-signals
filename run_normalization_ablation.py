"""Normalization ablation — per-date rank vs train-only StandardScaler.

Reruns run_models.py's elastic-net pipeline for configs (a) baseline, (c)
pruned, (d) full pool at the 1-day horizon under BOTH normalization schemes,
holding everything else fixed (same frame, same 2013-2025 walk-forward
splits and parameters as run_models.py, same hyperparameter search), and
writes one comparison table answering "did the normalization choice
matter?".

Config (b) baseline+interactions is excluded by design: its interaction
columns are products of rank-normalized features, so re-standardizing
them is ill-defined (`run_one` refuses that combination).

The rank side is also rerun here (not read from outputs/model_runs/)
because run_models.py's prediction parquets are not in git — only
coefs/params are. The rerun is deterministic, and
`check_rank_reproduction` asserts its coefficients match the committed
run's ones exactly, which is what guarantees the two arms differ only in
normalization.

The summary's headline delta is PAIRED: mean over only the test years
where both schemes produced a non-null model. Unpaired per-scheme means
(each over its own non-null folds) are also reported, but comparing them
directly would compare different year sets.

Run from the repo root:
    uv run python run_normalization_ablation.py                # full run, ~16 min
                                                               # (+ ~9 min feature build if uncached)
    uv run python run_normalization_ablation.py --smoke        # small wiring check, isolated
                                                               # under outputs/ablation_normalization/smoke/
    uv run python run_normalization_ablation.py --tables-only  # rebuild tables from persisted preds
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.evaluation import (
    mean_daily_spearman_ic_by_year,
    n_nonzero_by_year,
    pooled_pearson_ic_by_year,
)
from src.models import (
    BASE_COLS,
    INTERACTION_COLS,
    OUTPUT_DIR as MODEL_RUNS_DIR,
    build_model_frame,
    build_model_frame_chunked,
    get_feature_configs,
    load_pruned_keep,
    run_suite,
)

# Mirror run_models.py's full-run parameters so the ablation uses the
# exact splits the official run used. Keep these in sync BY HAND with
# run_models.py -- if its values change, the ablation must be rerun with
# the new ones or it stops being a controlled comparison.
START_YEAR = 2011
END_YEAR = 2025
MIN_TEST_YEAR = 2013
VAL_YEARS = 1
PURGE_DAYS = 5

# The frame is built with run_models.py's FULL horizon set even though the
# ablation only fits 1d: build_model_frame_chunked's per-year cache under
# outputs/features_by_year/ is shared with run_models.py and keyed by year
# only, so a (1,)-only build here would overwrite files run_models.py's 5d
# suite depends on (this happened once; don't reintroduce it).
FRAME_HORIZONS = (1, 5)

HORIZON = 1
SCHEMES = ("rank", "standardize")
CONFIG_NAMES = ("a_baseline", "c_pruned", "d_full_pool")
ABLATION_DIR = Path(__file__).resolve().parent / "outputs" / "ablation_normalization"
SMOKE_DIR = ABLATION_DIR / "smoke"


def _pool_cols(model_df: pd.DataFrame) -> list[str]:
    """Config (d)'s candidate pool. Duplicates the filter inside `run_suite`
    (src/models.py) on purpose -- kept local to avoid touching that file.
    If run_suite's exclusion set ever changes, change this to match."""
    return [
        c for c in model_df.columns
        if c not in {"permno", "segment_id", "date"}
        and not c.startswith("fwd_ret_")
        and c not in BASE_COLS
        and c not in INTERACTION_COLS
    ]


def build_ablation_configs(model_df: pd.DataFrame, *, strict: bool) -> dict[str, list[str]]:
    """Configs (a), (c), (d), minus (b) baseline_interactions.

    strict=True (full run): delegates to `get_feature_configs`, so any
    pruned-keep column missing from the built pool is an error -- same
    contract as the full run. strict=False (smoke run): a small universe/short
    window can drop more low-coverage columns than the full run did, so
    (c) is intersected with what's actually available instead of erroring.
    """
    pruned_keep = load_pruned_keep()
    pool_cols = _pool_cols(model_df)
    if strict:
        configs = get_feature_configs(pool_cols, pruned_keep)
        del configs["b_baseline_interactions"]
    else:
        pool_set = set(pool_cols)
        pruned_avail = [c for c in pruned_keep if c in pool_set]
        missing = sorted(set(pruned_keep) - pool_set)
        if missing:
            print(f"[smoke] {len(missing)} pruned-keep columns absent from the small pool, "
                  f"using the {len(pruned_avail)} available: missing={missing}")
        configs = {
            "a_baseline": list(BASE_COLS),
            "c_pruned": pruned_avail,
            "d_full_pool": list(pool_cols),
        }
    assert tuple(configs) == CONFIG_NAMES, f"unexpected config set: {tuple(configs)}"
    return configs


def build_detail(runs: dict[tuple[str, str], dict]) -> pd.DataFrame:
    """Per (config, scheme, test_year): pooled Pearson IC, mean daily
    Spearman IC, and the fold's nonzero-coefficient count (0 = validation
    chose the null, all-shrunk model; its ICs are NaN)."""
    rows = []
    for (scheme, config), run in runs.items():
        year_ic = pooled_pearson_ic_by_year(run["preds"])
        daily_ic = mean_daily_spearman_ic_by_year(run["preds"])
        nonzero = n_nonzero_by_year(run["coefs"])
        for year in year_ic.index:
            rows.append({
                "config": config,
                "scheme": scheme,
                "test_year": int(year),
                "pearson_ic": year_ic.loc[year],
                "daily_spearman_ic": daily_ic.loc[year],
                "n_nonzero_coefs": int(nonzero.loc[year]),
            })
    return pd.DataFrame(rows).sort_values(["config", "test_year", "scheme"]).reset_index(drop=True)


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Per config: unpaired per-scheme means (each over that scheme's
    non-null-model folds), null/active fold counts, and for each metric a
    PAIRED block -- means and std-minus-rank delta over only the years
    where both schemes produced a non-null model, plus that year count.
    The paired delta is the headline number; the unpaired means are
    descriptive and not directly comparable to each other."""
    year_sets = detail.groupby(["config", "scheme"])["test_year"].agg(frozenset)
    if year_sets.nunique() != 1:
        raise ValueError(
            "runs disagree on the test-year set -- mixed smoke/full artifacts or a partial "
            f"rerun?\n{year_sets.apply(sorted).to_string()}"
        )

    summary = detail.groupby(["config", "scheme"]).agg(
        pearson_ic=("pearson_ic", "mean"),          # mean over that scheme's non-null-model folds
        daily_spearman_ic=("daily_spearman_ic", "mean"),
        n_null_folds=("n_nonzero_coefs", lambda s: int((s == 0).sum())),
        n_active_folds=("pearson_ic", lambda s: int(s.notna().sum())),
        n_folds=("test_year", "count"),
    ).unstack("scheme")  # columns: (stat, scheme)
    summary.columns = [f"{stat}__{scheme}" for stat, scheme in summary.columns]

    for metric in ("pearson_ic", "daily_spearman_ic"):
        wide = detail.pivot(index="test_year", columns=["config", "scheme"], values=metric)
        # per config: the years where BOTH schemes were non-null
        paired = {config: wide[config].dropna() for config in summary.index}
        summary[f"{metric}__paired_rank"] = pd.Series({c: p["rank"].mean() for c, p in paired.items()})
        summary[f"{metric}__paired_standardize"] = pd.Series({c: p["standardize"].mean() for c, p in paired.items()})
        summary[f"{metric}__paired_delta_std_minus_rank"] = pd.Series(
            {c: (p["standardize"] - p["rank"]).mean() for c, p in paired.items()})
        summary[f"{metric}__n_paired_years"] = pd.Series({c: len(p) for c, p in paired.items()})
    return summary


def load_persisted_runs() -> dict[tuple[str, str], dict]:
    """Reload the full run's persisted preds/coefs, strictly.

    Requires every (scheme, config) pair to be present, with preds and
    coefs agreeing on their test years, and errors on any gap -- a partial
    or mixed set would otherwise be silently differenced as if it were a
    controlled A/B. (build_summary additionally asserts all runs share one
    test-year set.)
    """
    expected = {f"{c}__{HORIZON}d__preds.parquet" for c in CONFIG_NAMES}
    out = {}
    for scheme in SCHEMES:
        scheme_dir = ABLATION_DIR / scheme
        for config in CONFIG_NAMES:
            preds_path = scheme_dir / f"{config}__{HORIZON}d__preds.parquet"
            coefs_path = scheme_dir / f"{config}__{HORIZON}d__coefs.parquet"
            for path in (preds_path, coefs_path):
                if not path.exists():
                    raise FileNotFoundError(
                        f"{path} is missing -- all {len(SCHEMES) * len(CONFIG_NAMES)} "
                        "(scheme, config) runs must be persisted. Rerun without --tables-only "
                        "(these parquets are gitignored, so a fresh clone always needs the rerun)."
                    )
            preds = pd.read_parquet(preds_path)
            coefs = pd.read_parquet(coefs_path)["coef"]
            pred_years = sorted(int(y) for y in preds["test_year"].unique())
            coef_years = sorted(int(y) for y in coefs.index.get_level_values(0).unique())
            if pred_years != coef_years:
                raise ValueError(
                    f"{scheme}/{config}: preds cover test years {pred_years} but coefs cover "
                    f"{coef_years} -- stale or mixed artifacts on disk; rerun the ablation."
                )
            out[(scheme, config)] = {"preds": preds, "coefs": coefs}
        extra = {p.name for p in scheme_dir.glob("*__preds.parquet")} - expected
        if extra:
            print(f"warning: ignoring unexpected run files in {scheme_dir}: {sorted(extra)}")
    return out


def check_rank_reproduction(results: dict) -> None:
    """Assert the rank arm reproduces run_models.py's committed coefficients exactly.

    This is the evidence that the two ablation arms differ ONLY in
    normalization — enforced here rather than claimed in prose.
    """
    for config in CONFIG_NAMES:
        committed_path = MODEL_RUNS_DIR / f"{config}__{HORIZON}d__coefs.parquet"
        if not committed_path.exists():
            print(f"[repro] {committed_path.name} not found (fresh clone?) -- skipping diff")
            continue
        committed = pd.read_parquet(committed_path)["coef"]
        ours = results[f"{config}__{HORIZON}d"]["coefs"]
        if not committed.index.equals(ours.index):
            sample = sorted(set(map(str, committed.index)) ^ set(map(str, ours.index)))[:5]
            raise AssertionError(
                f"[repro] {config}: coef index differs from the committed run "
                f"({len(committed)} committed vs {len(ours)} rerun entries; e.g. {sample})"
            )
        max_diff = float((committed - ours).abs().max())
        print(f"[repro] {config}: max |coef diff| vs committed run = {max_diff:.3g}")
        if max_diff > 0:
            raise AssertionError(
                f"[repro] {config}: rank rerun no longer matches the committed coefs "
                f"(max |diff| = {max_diff:.3g}) -- the ablation is not a controlled comparison "
                "until this is explained (feature-code change? cache rebuilt differently?)."
            )


def _print_pivot(detail: pd.DataFrame, value: str, title: str, digits: int | None = None) -> None:
    print(f"\n=== {title} ===")
    table = detail.pivot(index="test_year", columns=["config", "scheme"], values=value)
    print((table.round(digits) if digits is not None else table).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="small wiring check: top_n=100, last 6 years, last 2 test years; "
                             "writes under outputs/ablation_normalization/smoke/")
    parser.add_argument("--tables-only", action="store_true",
                        help="skip all model runs; rebuild tables from the persisted full run")
    args = parser.parse_args()
    if args.smoke and args.tables_only:
        parser.error("--smoke and --tables-only are mutually exclusive")

    out_root = SMOKE_DIR if args.smoke else ABLATION_DIR

    if args.tables_only:
        runs = load_persisted_runs()
    else:
        t0 = time.time()
        if args.smoke:
            print("=== [smoke] building frame: top_n=100, last 6 years ===")
            frame = build_model_frame(top_n=100, lookback_years=6, horizons=(HORIZON,))
            # last year with any usable target; the final calendar year can
            # be all-NaN targets if the panel ends within the horizon
            target_years = frame.loc[frame[f"fwd_ret_{HORIZON}d"].notna(), "date"].dt.year
            min_test_year = int(target_years.max()) - 1
            max_test_year = None
        else:
            print(f"=== building full-universe frame {START_YEAR}-{END_YEAR} (cached per-year under outputs/features_by_year/) ===")
            frame = build_model_frame_chunked(
                top_n=None, start_year=START_YEAR, end_year=END_YEAR,
                warmup_years=2, horizons=FRAME_HORIZONS,  # full set -- see FRAME_HORIZONS comment
            )
            min_test_year = MIN_TEST_YEAR
            max_test_year = END_YEAR
        print(f"frame ready in {time.time() - t0:.1f}s, shape {frame.shape}\n")

        configs = build_ablation_configs(frame, strict=not args.smoke)
        print("configs:", {k: len(v) for k, v in configs.items()})

        runs = {}
        for scheme in SCHEMES:
            print(f"\n=== scheme: {scheme} ===")
            results = run_suite(
                frame,
                min_test_year=min_test_year, max_test_year=max_test_year,
                val_years=VAL_YEARS, purge_days=PURGE_DAYS,
                horizons=(HORIZON,),
                configs_to_run=configs,
                normalize=scheme,
                output_dir=out_root / scheme,
            )
            if scheme == "rank" and not args.smoke:
                check_rank_reproduction(results)
            for config in configs:
                runs[(scheme, config)] = results[f"{config}__{HORIZON}d"]

    detail = build_detail(runs)
    summary = build_summary(detail)
    out_root.mkdir(parents=True, exist_ok=True)
    detail_path = out_root / "ic_by_year.csv"
    summary_path = out_root / "summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path)

    pd.set_option("display.width", 200)
    _print_pivot(detail, "pearson_ic",
                 "per-year IC by (config, scheme) -- NaN = fold chose the null model (see n_nonzero_coefs)",
                 digits=4)
    _print_pivot(detail, "n_nonzero_coefs", "nonzero coefficients per fold")
    print("\n=== summary: unpaired per-scheme means + paired std-minus-rank deltas ===")
    print(summary.round(4).to_string())
    print(f"\nsaved -> {detail_path}\nsaved -> {summary_path}")


if __name__ == "__main__":
    main()
