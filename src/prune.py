"""Target-blind correlation prune on freeze years that are never val/test.

Candidate pool = 90 alphas + 15 classic. Baseline is a separate model, not pruned.
Correlation is mean daily cross-sectional Spearman (how two features rank stocks
that day), not pooled Pearson. Elastic net still selects later; this only drops
collinear copies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

FREEZE_START_YEAR = 2001
FREEZE_END_YEAR = 2010
TOP_N = None  # full universe; do not daily-cut to 500 (gaps + small-cap loss)
THRESHOLD = 0.7
SENSITIVITY = (0.6, 0.7, 0.8)
MIN_NAMES_PER_DATE = 30
MIN_COVERAGE = 0.20
CONFIG_PATH = _REPO_ROOT / "configs" / "pruned_features.json"
HEATMAP_PATH = _REPO_ROOT / "outputs" / "report" / "prune_corr.png"


def build_candidate_pool(panel: pd.DataFrame) -> pd.DataFrame:
    """Alphas + classic OHLCV. Does not include the 3-feature baseline."""
    from src.signals.alphas import build_alpha_features
    from src.signals.classic_features import build_classic_features

    return pd.concat(
        [build_alpha_features(panel), build_classic_features(panel)],
        axis=1,
    )


def drop_low_coverage(X: pd.DataFrame, *, min_coverage: float = MIN_COVERAGE) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns that are finite on too few rows. Target-blind hygiene."""
    X = X.replace([np.inf, -np.inf], np.nan)
    coverage = X.notna().mean()
    dropped = coverage[coverage < min_coverage].index.tolist()
    return X.drop(columns=dropped), dropped


def mean_daily_spearman(X: pd.DataFrame, *, min_names: int = MIN_NAMES_PER_DATE) -> pd.DataFrame:
    """Average of per-date Spearman correlations (Pearson of within-date ranks)."""
    cols = list(X.columns)
    n_f = len(cols)
    acc = np.zeros((n_f, n_f))
    count = np.zeros((n_f, n_f))
    dates = X.index.get_level_values("date")
    for _, day in X.groupby(dates, sort=False):
        if len(day) < min_names:
            continue
        ranked = day.rank(axis=0, method="average")
        c = ranked.corr(method="pearson").to_numpy()
        valid = np.isfinite(c)
        acc += np.where(valid, c, 0.0)
        count += valid
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = acc / count
    np.fill_diagonal(mean, 1.0)
    return pd.DataFrame(mean, index=cols, columns=cols)


def greedy_drop(corr: pd.DataFrame, threshold: float) -> tuple[list[str], list[dict]]:
    """Drop one side of the max |ρ| pair until every remaining pair is below threshold.

    Of the two features in that pair, drop the one with higher mean |ρ| to the
    rest of the remaining set. Ties: drop the lexicographically later name.
    """
    remaining = list(corr.columns)
    dropped: list[dict] = []
    abs_c = corr.abs()
    while len(remaining) > 1:
        sub = abs_c.loc[remaining, remaining].to_numpy(dtype=float, copy=True)
        np.fill_diagonal(sub, 0.0)
        if not np.isfinite(sub).any():
            break
        i, j = np.unravel_index(np.nanargmax(sub), sub.shape)
        max_rho = float(sub[i, j])
        if not np.isfinite(max_rho) or max_rho < threshold:
            break
        a, b = remaining[i], remaining[j]
        mean_a = float(np.nanmean(np.delete(sub[i], i)))
        mean_b = float(np.nanmean(np.delete(sub[j], j)))
        drop = a if (mean_a > mean_b or (mean_a == mean_b and a > b)) else b
        keep_other = b if drop == a else a
        dropped.append({"feature": drop, "rho": round(max_rho, 4), "with": keep_other})
        remaining.remove(drop)
    return remaining, dropped


def _cluster_order(corr: pd.DataFrame) -> list[str]:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    abs_c = corr.abs().fillna(0.0).clip(upper=1.0)
    dist = (1.0 - abs_c).to_numpy(dtype=float, copy=True)
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    condensed = squareform(dist, checks=False)
    order = leaves_list(linkage(condensed, method="average"))
    return [corr.index[i] for i in order]


def save_heatmap(corr: pd.DataFrame, keep: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    full_order = _cluster_order(corr)
    keep_corr = corr.loc[keep, keep]
    keep_order = _cluster_order(keep_corr) if len(keep) > 1 else keep

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, order, title, labeled in (
        (axes[0], corr, full_order, f"all {corr.shape[0]}", False),
        (axes[1], keep_corr, keep_order, f"kept {len(keep)}", True),
    ):
        m = mat.loc[order, order].to_numpy()
        im = ax.imshow(m, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(title)
        if labeled:
            ax.set_xticks(range(len(order)))
            ax.set_yticks(range(len(order)))
            ax.set_xticklabels(order, rotation=90, fontsize=6)
            ax.set_yticklabels(order, fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_prune(
    *,
    start_year: int = FREEZE_START_YEAR,
    end_year: int = FREEZE_END_YEAR,
    top_n: int | None = TOP_N,
    threshold: float = THRESHOLD,
    config_path: Path = CONFIG_PATH,
    heatmap_path: Path = HEATMAP_PATH,
) -> dict:
    from src.data import load_panel
    from src.signals.inputs import prepare_panel

    raw = load_panel(top_n=top_n, start_year=start_year, end_year=end_year)
    panel = prepare_panel(raw)
    print(f"loaded {len(raw):,} rows {start_year}-{end_year} top_n={top_n}; building candidate pool")
    X = build_candidate_pool(panel)
    print(f"feature matrix {X.shape}")
    X, low_cov = drop_low_coverage(X)
    corr = mean_daily_spearman(X)

    sensitivity = {}
    for t in SENSITIVITY:
        keep_t, dropped_t = greedy_drop(corr, t)
        sensitivity[str(t)] = {"n_keep": len(keep_t), "keep": sorted(keep_t), "n_dropped": len(dropped_t)}

    keep, dropped = greedy_drop(corr, threshold)
    save_heatmap(corr, keep, heatmap_path)

    result = {
        "start_year": start_year,
        "end_year": end_year,
        "top_n": top_n,
        "n_rows": len(X),
        "method": "mean_daily_spearman",
        "threshold": threshold,
        "min_coverage": MIN_COVERAGE,
        "low_coverage_dropped": sorted(low_cov),
        "keep": sorted(keep),
        "n_keep": len(keep),
        "dropped": dropped,
        "sensitivity": sensitivity,
        "never_val_test_years": list(range(start_year, end_year + 1)),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    out = run_prune()
    print(f"rows {out['n_rows']:,}  keep {out['n_keep']}  low-coverage {out['low_coverage_dropped']}")
    print("sensitivity:", {k: v["n_keep"] for k, v in out["sensitivity"].items()})
    print("keep:", out["keep"])
    print("wrote", CONFIG_PATH)
    print("wrote", HEATMAP_PATH)
