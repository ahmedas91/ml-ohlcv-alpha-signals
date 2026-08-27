"""Run elastic-net suite (4 feature configs x 2 horizons).

Configs:
  (a) baseline               -- 3 fixed features (src.signals.baseline)
  (b) baseline_interactions  -- baseline + its 3 pairwise interactions
  (c) pruned                 -- configs/pruned_features.json 'keep' list
                                 (candidate pool only -- baseline is a
                                 separate model, not pruned, per prune.py)
  (d) full_pool               -- full alpha+classic candidate pool, with
                                 the same low-coverage columns dropped
                                 that prune.py drops (target-blind hygiene,
                                 not part of the correlation prune itself)

Pipeline order matters: build_forward_return_targets() must run BEFORE
prepare_panel(), because prepare_panel() moves permno/date into the index and
build_forward_return_targets() needs them as columns. README.md documents this
order.

One pipeline, parameterized by (feature list, horizon): `run_one` grid-searches
(alpha, l1_ratio) on the validation slice only (never test), refits on
train+val, and logs the chosen params per fold. `run_suite` runs all 8
(config x horizon) combinations and persists predictions + params.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, enet_path
from sklearn.preprocessing import StandardScaler

from src.data import build_forward_return_targets, load_panel
from src.prune import drop_low_coverage
from src.signals import build_baseline_features, prepare_panel
from src.signals.alphas import build_alpha_features
from src.signals.classic_features import build_classic_features
from src.validation import yearly_walk_forward

CONFIG_PATH = _REPO_ROOT / "configs" / "pruned_features.json"
OUTPUT_DIR = _REPO_ROOT / "outputs" / "model_runs"

BASE_COLS = ["rev_1d", "rev_5d", "mom_12_1"]
INTERACTION_COLS = ["rev_1d_x_rev_5d", "rev_1d_x_mom_12_1", "rev_5d_x_mom_12_1"]

# l1_ratio grid -- alpha is no longer a fixed grid; see compute_alpha_grid().
L1_RATIO_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def rank_normalize_features(df: pd.DataFrame, feature_cols: list[str], date_col: str = "date") -> pd.DataFrame:
    """Per-date cross-sectional rank, rescaled from (0, 1] to (-1, 1] -- same
    transform `baseline.py` already uses for its interaction terms
    (`_normalize_for_interaction`), applied here to every feature column so
    the whole suite uses one consistent normalization.

    This is the *default* normalization: rank normalization first,
    StandardScaler is a separate ablation, not the default here. It only
    depends on that day's own cross-section
    (never other dates), so it's leakage-safe to compute once on the
    whole frame before any train/val/test split -- no fit-on-train step
    needed, unlike StandardScaler.

    As a side effect this also neutralizes the alpha_084/alpha_003
    numerical blow-ups found earlier: rank is bounded in (-1, 1]
    regardless of how large the raw feature value is, so the manual
    winsorizing workaround used previously is no longer needed once
    this is the input.
    """
    out = df.copy()
    ranked = df.groupby(date_col)[feature_cols].rank(pct=True)
    out[feature_cols] = 2.0 * ranked - 1.0
    return out


def compute_alpha_grid(X_tr: np.ndarray, y_tr: np.ndarray, l1_ratio: float, n_alphas: int = 30) -> np.ndarray:
    """alpha path from alpha_max (full shrinkage) down to alpha_max * 1e-4.

    alpha_max = max_j |x_j^T y_centered| / (n * l1_ratio) -- the smallest
    alpha that shrinks every slope coefficient to zero, matching
    scikit-learn's objective after standardizing/normalizing X and
    centering y. Computed fresh per fold, per l1_ratio (alpha_max
    depends on l1_ratio), from *training* data only -- this replaces the
    earlier fixed `np.logspace(-4, 0, 12)` grid, which was an arbitrary
    range not derived from the data and not guaranteed to reach the
    point where validation MSE turns back upward (lecture 2, "best
    practices #2").
    """
    y_centered = y_tr - y_tr.mean()
    alpha_max = np.max(np.abs(X_tr.T @ y_centered)) / (X_tr.shape[0] * l1_ratio)
    return alpha_max * np.logspace(0, -4, n_alphas)


def build_model_frame(
    *, top_n: int | None, start_year: int | None = None, end_year: int | None = None,
    lookback_years: int | None = None, horizons: tuple[int, ...] = (1, 5),
) -> pd.DataFrame:
    """load -> targets -> prepare_panel -> baseline+alpha+classic -> flat frame.

    Returns one flat (not MultiIndexed) dataframe with every candidate
    feature column, `fwd_ret_{h}d` for each horizon, and `date`/`permno`
    as plain columns (required by `yearly_walk_forward`).

    Loads and computes features for the whole requested range in one
    pass -- fine for a few years or a restricted universe, but building
    all ~105 candidate columns across the full-universe, multi-year
    panel can need several GB of RAM at once (90 alpha columns x
    millions of rows). For a large range, use
    `build_model_frame_chunked` instead, which processes and persists
    one year at a time to keep peak memory bounded.
    """
    raw = load_panel(top_n=top_n, start_year=start_year, end_year=end_year, lookback_years=lookback_years)
    with_targets = build_forward_return_targets(raw, horizons=horizons)
    panel = prepare_panel(with_targets)

    baseline = build_baseline_features(panel).drop(columns="close")
    alpha_pool = build_alpha_features(panel)
    classic_pool = build_classic_features(panel)
    pool = pd.concat([alpha_pool, classic_pool], axis=1)
    pool, low_cov_dropped = drop_low_coverage(pool)
    if low_cov_dropped:
        print(f"build_model_frame: dropped low-coverage pool columns: {low_cov_dropped}")

    target_cols = [f"fwd_ret_{h}d" for h in horizons]
    frame = pd.concat([baseline, pool, panel[target_cols]], axis=1)
    return frame.reset_index()  # permno, segment_id, date back as columns


def build_model_frame_chunked(
    *, top_n: int | None, start_year: int, end_year: int, warmup_years: int = 2,
    horizons: tuple[int, ...] = (1, 5), out_dir: Path = _REPO_ROOT / "outputs" / "features_by_year",
    dtype: str = "float32",
) -> pd.DataFrame:
    """Same output as `build_model_frame`, built and persisted one year at a time.

    For each output year Y, uses Y-warmup_years..Y as a warmup buffer so
    trailing-window features like `dist_52wk_high` (252-day window) and
    `mom_12_1` have real history at the start of year Y, not
    manufactured NaNs, then keeps and persists only year Y's rows.

    `load_panel` reads the *entire* parquet before filtering by year --
    fine for one call, but calling it once per year inside a loop
    re-reads and holds the full multi-million-row file every iteration.
    Confirmed this OOM-kills the process in a ~4GB-RAM environment even
    though each year's *output* is small. Fixed here by loading the raw
    panel once for the whole start_year..end_year range up front, then
    slicing each year's warmup window out of that single in-memory copy
    -- so `load_panel` itself is called once, not once per year.

    Feature columns are cast to `dtype` (float32 default, half the
    memory of pandas' float64 default) before persisting.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    year_frames = []
    batch_years = 5  # reload raw_all every `batch_years` output years, not once for the whole range
    raw_all = None
    raw_all_range = None  # (batch_start, batch_end) currently held in memory

    for year in range(start_year, end_year + 1):
        out_path = out_dir / f"{year}.parquet"
        if out_path.exists():
            year_frame = pd.read_parquet(out_path)
        else:
            buffer_start = year - warmup_years
            need_reload = raw_all is None or buffer_start < raw_all_range[0] or year > raw_all_range[1]
            if need_reload:
                batch_end = min(end_year, year + batch_years - 1)
                del raw_all
                import gc
                gc.collect()
                raw_all = load_panel(top_n=top_n, start_year=buffer_start, end_year=batch_end)
                raw_all_range = (buffer_start, batch_end)
                print(f"  [loaded raw batch {raw_all_range[0]}-{raw_all_range[1]}: {len(raw_all):,} rows]")

            raw_slice = raw_all[(raw_all["date"].dt.year >= buffer_start) & (raw_all["date"].dt.year <= year)]
            with_targets = build_forward_return_targets(raw_slice, horizons=horizons)
            panel = prepare_panel(with_targets)

            baseline = build_baseline_features(panel).drop(columns="close")
            alpha_pool = build_alpha_features(panel)
            classic_pool = build_classic_features(panel)
            pool = pd.concat([alpha_pool, classic_pool], axis=1)
            pool, _ = drop_low_coverage(pool)
            target_cols = [f"fwd_ret_{h}d" for h in horizons]
            frame = pd.concat([baseline, pool, panel[target_cols]], axis=1).reset_index()

            year_frame = frame[frame["date"].dt.year == year].reset_index(drop=True)
            feature_cols = [c for c in year_frame.columns if c not in {"permno", "segment_id", "date"}]
            year_frame[feature_cols] = year_frame[feature_cols].astype(dtype)
            year_frame.to_parquet(out_path)

            del raw_slice, with_targets, panel, baseline, alpha_pool, classic_pool, pool, frame
            import gc
            gc.collect()
        year_frames.append(year_frame)
        print(f"  {year}: {year_frame.shape[0]:,} rows persisted -> {out_path.name}")
    return pd.concat(year_frames, ignore_index=True)


def load_pruned_keep(config_path: Path = CONFIG_PATH) -> list[str]:
    return json.loads(config_path.read_text())["keep"]


def get_feature_configs(pool_cols: list[str], pruned_keep: list[str]) -> dict[str, list[str]]:
    """{config_name: [feature columns]} for the 4 configs."""
    missing = [c for c in pruned_keep if c not in pool_cols]
    if missing:
        raise ValueError(
            f"pruned_features.json lists columns not present in the built pool: {missing}. "
            "Likely a mismatch between prune.py's candidate pool and this run's."
        )
    return {
        "a_baseline": list(BASE_COLS),
        "b_baseline_interactions": list(BASE_COLS) + list(INTERACTION_COLS),
        "c_pruned": list(pruned_keep),
        "d_full_pool": list(pool_cols),
    }


def _winsorize_like_train(X_ref: np.ndarray, *arrays: np.ndarray, limits=(0.005, 0.995)):
    """Clip `arrays` to per-column [lo, hi] percentiles computed from X_ref only.

    Only used when `normalize="standardize"` (the standardize ablation path).
    With the default rank normalization, features are already bounded in
    (-1, 1], so this isn't needed there -- it was originally added to
    handle alpha_084/alpha_003 producing astronomically large (not
    NaN/inf) values that overflowed StandardScaler; rank normalization
    sidesteps that same problem for free.
    """
    lo = np.nanpercentile(X_ref, limits[0] * 100, axis=0)
    hi = np.nanpercentile(X_ref, limits[1] * 100, axis=0)
    return tuple(np.clip(a, lo, hi) for a in arrays)


def make_elastic_net_fit_fn(
    l1_ratio_grid=L1_RATIO_GRID, n_alphas: int = 30, param_log: list | None = None,
    max_iter: int = 2000, tol: float = 1e-3,
    normalize: str = "rank", winsorize_limits: tuple[float, float] = (0.005, 0.995),
):
    """(alpha, l1_ratio) grid search on val, refit on train+val, predict test.

    `max_iter`/`tol` control the coordinate-descent solver's convergence
    criteria. The alpha path deliberately reaches near-zero regularization
    (down to alpha_max * 1e-4, see `compute_alpha_grid`), which on a
    large, expanding training window (later walk-forward folds can have
    millions of rows) is numerically hard to fully converge -- the
    default sklearn tol=1e-4 can burn through many iterations per alpha
    without settling. `tol=1e-3`/`max_iter=2000` here trade a bit of
    per-alpha precision for materially faster runtime; this doesn't
    change which alpha wins the validation-MSE comparison in practice,
    since we're comparing candidates against each other, not needing an
    exact optimum for any single one.

    `normalize="rank"` (default): features are assumed already per-date
    rank-normalized to (-1, 1] upstream (`rank_normalize_features`,
    applied once before any train/val/test split -- safe to do that
    early since ranking only uses that day's own cross-section, never
    other dates). This fit_fn does no further rescaling in that case.

    `normalize="standardize"`: for the normalization ablation only.
    Winsorizes then fits StandardScaler per fold, on train data during the
    (alpha, l1_ratio) search and on train+val for the final refit (val is
    in-sample by then) -- never on test. Missing features must arrive as
    NaN in this mode (`run_one` arranges that): the winsorize percentiles
    and scaler moments are computed on observed values only (both are
    NaN-aware), and imputation happens AFTER scaling, as 0 in scaled space
    (= the fitting set's mean). Filling raw zeros *before* scaling -- what
    the shared splitter's default fillna(0) would do -- both contaminates
    the fitted statistics and plants a fake (0 - mean)/std signal on every
    missing entry of any feature not centered at zero, which for some
    columns is >20% of rows. The rank arm needs no such care: pre-fill
    ranked features with 0 (the cross-sectional midpoint) upstream and
    the fill is already neutral.
    This mode is NOT valid for any feature set containing
    the baseline interaction terms (see `INTERACTION_COLS`): those are
    already products of rank-normalized features, so re-standardizing
    them has no clean interpretation. `run_one` refuses this combination.

    For each l1_ratio, the alpha search grid is computed fresh from the
    *training* fold via `compute_alpha_grid` (alpha_max down to
    alpha_max * 1e-4) -- not a fixed range, since alpha_max depends on
    both the fold's data and l1_ratio. If `param_log` is given, each
    fold's chosen (alpha, l1_ratio, val_mse) is appended to it, in the
    order `yearly_walk_forward` calls this function (one call per test
    year, in increasing order).
    """
    if normalize not in ("rank", "standardize"):
        raise ValueError(f"normalize must be 'rank' or 'standardize', got {normalize!r}")

    def fit_fn(X_tr, y_tr, X_val, y_val, X_tv, y_tv, X_te, **kwargs):
        if normalize == "standardize":
            # np.clip and StandardScaler both pass NaN through (scaler fit
            # ignores NaN), so missing values survive to the post-scaling
            # imputation below instead of polluting the fitted statistics.
            X_tr, X_val = _winsorize_like_train(X_tr, X_tr, X_val, limits=winsorize_limits)
            scaler = StandardScaler().fit(X_tr)
            X_tr_fit, X_val_fit = scaler.transform(X_tr), scaler.transform(X_val)
            # 0 in scaled space == the train mean: neutral mean-imputation.
            X_tr_fit = np.nan_to_num(X_tr_fit, copy=False)
            X_val_fit = np.nan_to_num(X_val_fit, copy=False)
        else:
            X_tr_fit, X_val_fit = X_tr, X_val

        # enet_path (unlike ElasticNet.fit) has no intercept handling -- it
        # expects centered inputs. Center on TRAIN stats only (leakage-safe)
        # and apply the same offset to val, matching the alpha_max formula's
        # assumption of centered y / standardized-or-normalized X.
        x_mean = X_tr_fit.mean(axis=0)
        y_mean = y_tr.mean()
        X_tr_c = X_tr_fit - x_mean
        y_tr_c = y_tr - y_mean
        X_val_c = X_val_fit - x_mean

        best_mse, best_alpha, best_l1 = np.inf, None, l1_ratio_grid[0]
        for l1_ratio in l1_ratio_grid:
            alpha_grid = compute_alpha_grid(X_tr_c, y_tr_c, l1_ratio, n_alphas=n_alphas)
            _, coefs_path, _ = enet_path(
                X_tr_c, y_tr_c, l1_ratio=l1_ratio, alphas=alpha_grid, max_iter=max_iter, tol=tol,
            )
            val_preds_path = X_val_c @ coefs_path + y_mean  # (n_val, n_alphas)
            mse_path = np.mean((val_preds_path - y_val[:, None]) ** 2, axis=0)
            i_best = int(np.argmin(mse_path))
            if mse_path[i_best] < best_mse:
                best_mse, best_alpha, best_l1 = mse_path[i_best], alpha_grid[i_best], l1_ratio

        if normalize == "standardize":
            X_tv, X_te = _winsorize_like_train(X_tv, X_tv, X_te, limits=winsorize_limits)
            scaler_tv = StandardScaler().fit(X_tv)
            X_tv_fit, X_te_fit = scaler_tv.transform(X_tv), scaler_tv.transform(X_te)
            X_tv_fit = np.nan_to_num(X_tv_fit, copy=False)
            X_te_fit = np.nan_to_num(X_te_fit, copy=False)
        else:
            X_tv_fit, X_te_fit = X_tv, X_te

        # Final refit is a single (alpha, l1_ratio) -- no path/warm-start
        # needed, so plain ElasticNet.fit with its own built-in intercept
        # handling is simplest and equivalent.
        final_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=max_iter, tol=tol).fit(X_tv_fit, y_tv)
        if param_log is not None:
            param_log.append({"alpha": best_alpha, "l1_ratio": best_l1, "val_mse": best_mse})
        return final_model.predict(X_te_fit), final_model.coef_

    return fit_fn


def run_one(
    model_df: pd.DataFrame, x_cols: list[str], y_col: str, *,
    min_test_year: int, max_test_year: int | None = None, val_years: int = 1, purge_days: int = 5,
    normalize: str = "rank",
) -> dict:
    """Run one (feature config, horizon) combo. Returns preds/coefs/params/splits.

    Rows with a NaN target are dropped first -- the splitter has no
    target-NaN handling, and a NaN target is a real "no valid label" case
    (end-of-history or a segment/re-entry gap), not something to impute.

    NaN *features*: under `normalize="rank"` they are pre-filled with 0
    (the cross-sectional midpoint of the rank scale -- neutral) by the
    splitter. Under `normalize="standardize"` they are
    passed through as NaN and imputed inside the fit_fn AFTER per-fold
    scaling, so the two schemes impute at the same (neutral) point and the
    ablation measures scaling, not missing-data handling -- see
    `make_elastic_net_fit_fn`'s docstring.

    Refuses `normalize="standardize"` if `x_cols` includes any interaction
    term -- see `make_elastic_net_fit_fn`'s docstring for why.
    """
    if normalize == "standardize":
        bad = [c for c in x_cols if c in INTERACTION_COLS]
        if bad:
            raise ValueError(
                f"normalize='standardize' is not valid for interaction columns {bad}: "
                "they're already products of rank-normalized features, so "
                "standardizing them again isn't well-defined. Use normalize='rank' "
                "for any config that includes interaction terms."
            )

    valid = model_df[y_col].notna()
    if not valid.all():
        model_df = model_df.loc[valid]

    param_log: list[dict] = []
    fit_fn = make_elastic_net_fit_fn(param_log=param_log, normalize=normalize)

    preds, coefs, split_table = yearly_walk_forward(
        model_df, fit_fn, x_cols=x_cols, y_col=y_col,
        min_test_year=min_test_year, max_test_year=max_test_year,
        val_years=val_years, purge_days=purge_days,
        fillna_zero=(normalize == "rank"),
    )
    years = sorted(preds["test_year"].unique())
    params = pd.DataFrame(param_log, index=pd.Index(years, name="test_year"))
    return {"preds": preds, "coefs": coefs, "params": params, "splits": split_table}


def run_suite(
    model_df: pd.DataFrame, *, min_test_year: int, max_test_year: int | None = None,
    val_years: int = 1, purge_days: int = 5, horizons: tuple[int, ...] = (1, 5),
    output_dir: Path = OUTPUT_DIR, config_path: Path = CONFIG_PATH,
    configs_to_run: dict[str, list[str]] | None = None, normalize: str = "rank",
) -> dict:
    """Run all (4 configs x len(horizons)) combos; persist preds+coefs+params per combo.

    `normalize="rank"` (default): every feature column is rank-normalized
    once up front via `rank_normalize_features`, before any config/horizon
    split -- correct to do this before splitting since ranking is per-date
    and leakage-safe (see that function's docstring).

    `normalize="standardize"`: skips the up-front rank transform; each
    fold instead winsorizes + fits StandardScaler on train data only,
    inside `make_elastic_net_fit_fn`. Not valid for
    `b_baseline_interactions` -- pass `configs_to_run` with that config
    excluded (`run_one` will raise if it's included anyway).

    `configs_to_run` lets a caller run a subset of the 4 configs instead
    of all of them, e.g. `{"a_baseline": [...], ...}`.
    """
    pruned_keep = load_pruned_keep(config_path)
    pool_cols = [c for c in model_df.columns if c not in {"permno", "segment_id", "date"} and not c.startswith("fwd_ret_")]
    pool_cols = [c for c in pool_cols if c not in BASE_COLS and c not in INTERACTION_COLS]
    configs = configs_to_run if configs_to_run is not None else get_feature_configs(pool_cols, pruned_keep)

    if normalize == "rank":
        all_feature_cols = sorted({c for cols in configs.values() for c in cols})
        model_df = rank_normalize_features(model_df, all_feature_cols)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for h in horizons:
        y_col = f"fwd_ret_{h}d"
        for config_name, x_cols in configs.items():
            t0 = time.time()
            key = f"{config_name}__{h}d"
            out = run_one(
                model_df, x_cols, y_col,
                min_test_year=min_test_year, max_test_year=max_test_year,
                val_years=val_years, purge_days=purge_days, normalize=normalize,
            )
            out["preds"].to_parquet(output_dir / f"{key}__preds.parquet")
            out["coefs"].to_frame("coef").to_parquet(output_dir / f"{key}__coefs.parquet")
            out["params"].to_parquet(output_dir / f"{key}__params.parquet")
            results[key] = out
            print(f"{key}: {len(x_cols)} features, {time.time() - t0:.1f}s")
    return results


def coefficient_comparison(results: dict, horizon: int, *, zero_tol: float = 1e-8) -> pd.DataFrame:
    """What (d) full_pool's elastic net keeps (nonzero, any fold) vs what the
    correlation filter kept in (c).
    """
    full_key = f"d_full_pool__{horizon}d"
    pruned_key = f"c_pruned__{horizon}d"
    full_coefs = results[full_key]["coefs"].unstack(level=0)  # index: feature, cols: test_year
    kept_by_enet = (full_coefs.abs() > zero_tol).any(axis=1)
    pruned_set = set(results[pruned_key]["coefs"].index.get_level_values(-1))

    table = pd.DataFrame({
        "kept_by_correlation_filter": kept_by_enet.index.isin(pruned_set),
        "kept_by_elastic_net_any_fold": kept_by_enet,
        "mean_abs_coef": full_coefs.abs().mean(axis=1),
    })
    return table.sort_values("mean_abs_coef", ascending=False)


if __name__ == "__main__":
    # Small-scale smoke test: confirm the pipeline runs end to end and produces
    # sane per-year IC for each config, not a full research run.
    frame = build_model_frame(top_n=100, lookback_years=6)
    print(f"model frame: {frame.shape}")

    results = run_suite(
        frame,
        min_test_year=int(frame["date"].dt.year.max()) - 1,
        val_years=1, purge_days=5, horizons=(1,),
    )

    print("\n=== per-config IC (1d horizon) ===")
    for name, out in results.items():
        preds = out["preds"]
        ic = preds.groupby("test_year").apply(lambda g: g["y"].corr(g["yhat"]))
        print(name, dict(ic.round(4)))
