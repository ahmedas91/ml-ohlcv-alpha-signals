"""Scorecard -- turns saved predictions (from outputs/model_runs*/ folders)
into the final report metrics and figures.

Everything here is by-year, never a single pooled full-sample number:
by-year always, never a single full-sample max IC (winner's curse). A
pooled statistic invites picking whichever
model/config/horizon happens to look best across the whole sample; per-year
breakdowns force an honest look at the full spread, including bad years.

Expects `preds` dataframes shaped exactly like `yearly_walk_forward`'s
output: columns `test_year`, `date`, `y` (actual forward return), `yhat`
(prediction). No stock identifier column is needed or present -- every
metric here is a per-date cross-sectional computation (rank IC, decile
sorts), which only requires grouping by `date`, not knowing which stock is
which.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Stress windows named in the course brief / proposal. 2008 is included for
# completeness but will produce an empty slice for any run whose test years
# start at 2013 (see `stress_window_table`'s explicit note about this,
# rather than silently returning a blank row).
STRESS_WINDOWS = {
    "2008_gfc": (2008, 2008),
    "2020_covid": (2020, 2020),
    "2022_rate_hikes": (2022, 2022),
    "2025_tariffs": (2025, 2025),
}


# --- Daily cross-sectional rank IC ---


def daily_spearman_ic(preds: pd.DataFrame) -> pd.Series:
    """Spearman rank correlation of (y, yhat) computed separately per date.

    Returns a Series indexed by date. A date with fewer than 3 names, or
    where yhat is constant that day (e.g. a null-model fold predicting the
    same value for everyone), yields NaN rather than a spurious 0 or 1.
    """
    def _corr(g: pd.DataFrame) -> float:
        if len(g) < 3 or g["yhat"].nunique() < 2:
            return np.nan
        return g["y"].corr(g["yhat"], method="spearman")

    return preds.groupby("date").apply(_corr, include_groups=False)


def ic_by_year(preds: pd.DataFrame) -> pd.Series:
    """Mean daily Spearman IC within each test_year. NaN dates are dropped
    before averaging (a handful of NaN dates inside an otherwise-active
    year shouldn't blank out the whole year's IC)."""
    ic = daily_spearman_ic(preds)
    date_to_year = preds.drop_duplicates("date").set_index("date")["test_year"]
    return ic.groupby(date_to_year.reindex(ic.index)).mean()


def ic_summary_stats(ic_series: pd.Series) -> dict:
    """mean IC, IC std, IR (mean/std), and a t-stat (IR * sqrt(n_years))
    over the *active* years only (a null-model year's IC is NaN and
    dropped here, not treated as a zero -- a zero would understate IC std
    and overstate significance)."""
    active = ic_series.dropna()
    n = len(active)
    mean_ic = active.mean()
    std_ic = active.std()
    ir = mean_ic / std_ic if std_ic and not np.isnan(std_ic) else np.nan
    t_stat = ir * np.sqrt(n) if n > 1 and not np.isnan(ir) else np.nan
    return {"mean_ic": mean_ic, "ic_std": std_ic, "ir": ir, "t_stat": t_stat, "n_years": n}


# --- Decile long-short portfolios ---


def _decile_returns_for_date(g: pd.DataFrame, n_deciles: int) -> pd.Series | None:
    """Equal-weighted mean realized return per decile for one date's
    cross-section, decile 1 = lowest predicted, n_deciles = highest.

    Ties in `yhat` are broken by a stable arbitrary order (`rank(method="first")`)
    before cutting into deciles, rather than cutting on the raw values
    directly. This matters a lot for tree-model predictions specifically:
    XGBoost's output is only as fine-grained as its leaf structure, so
    hundreds of stocks can share the exact same predicted value on a given
    date (observed directly on real data: 392/975 stocks tied at one
    value on one date). A raw `qcut` on values that clustered can't form
    n_deciles equal-sized groups at all and would silently drop most
    dates; breaking ties by rank first means every date with enough names
    still gets a full decile split, at the cost of that split being
    arbitrary among stocks that were tied.

    Returns None if there aren't enough names that day to form
    `n_deciles` non-empty groups at all.
    """
    if len(g) < n_deciles or g["yhat"].nunique() < 2:
        return None
    ranked = g["yhat"].rank(method="first")
    try:
        decile = pd.qcut(ranked, n_deciles, labels=False) + 1
    except ValueError:
        return None
    return g.groupby(decile)["y"].mean()


def build_decile_portfolio(
    preds: pd.DataFrame, *, horizon: int, n_deciles: int = 10,
) -> pd.DataFrame:
    """Per-formation-date decile returns and the long-short spread.

    5-day-horizon implementation choice: NON-OVERLAPPING. A new
    portfolio is formed only every `horizon` trading days within each
    test_year (days 0, 5, 10, ... relative to that year's first date);
    its return is the actual realized h-day forward return already stored
    in `y` (no averaging/approximation needed). Chosen over the
    overlapping-averaged alternative because it avoids two things: (1)
    the serial correlation overlapping daily-equivalent returns would
    introduce into a Sharpe calculation, which biases the ratio if not
    corrected for; (2) the return/horizon "daily-equivalent" step itself
    is an approximation, not the number that was actually realized.
    Trade-off: this only uses 1-in-5 of the daily predictions for the
    return series (all 5 are still used for the daily IC calculation
    above, which does not have this issue -- IC is computed per date
    regardless of holding-period overlap).

    For horizon=1, every date is used -- no overlap to resolve.
    """
    dates = sorted(preds["date"].unique())
    step = 1 if horizon == 1 else horizon
    if horizon != 1:
        # restart the every-`horizon`-days cadence at the start of each
        # test_year, not globally, so a year boundary can't shift which
        # calendar dates get selected within it
        keep_dates = []
        for year, group in preds.groupby("test_year"):
            year_dates = sorted(group["date"].unique())
            keep_dates.extend(year_dates[::step])
        dates = sorted(keep_dates)

    rows = []
    for d in dates:
        day = preds.loc[preds["date"] == d]
        dec = _decile_returns_for_date(day, n_deciles)
        if dec is None:
            continue
        row = {"date": d, "test_year": int(day["test_year"].iloc[0])}
        for k in range(1, n_deciles + 1):
            row[f"decile_{k}"] = dec.get(k, np.nan)
        row["long_short"] = dec.get(n_deciles, np.nan) - dec.get(1, np.nan)
        rows.append(row)

    if not rows:
        cols = ["date", "test_year"] + [f"decile_{k}" for k in range(1, n_deciles + 1)] + ["long_short"]
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def decile_monotonicity(decile_portfolio: pd.DataFrame, n_deciles: int = 10) -> pd.Series:
    """Mean realized return per decile, averaged across all formation
    dates -- a monotonic decile_1 -> decile_n_deciles increase is the
    visual/quantitative check that the signal ranks stocks sensibly, not
    just that the two tails are far apart."""
    cols = [f"decile_{k}" for k in range(1, n_deciles + 1)]
    return decile_portfolio[cols].mean()


def gross_sharpe_by_year(decile_portfolio: pd.DataFrame, *, horizon: int) -> pd.Series:
    """Annualized gross Sharpe of the long-short return series, per
    test_year. Annualization uses TRADING_DAYS_PER_YEAR / horizon periods
    per year, matching the non-overlapping period length used to build
    the portfolio (see `build_decile_portfolio`)."""
    periods_per_year = TRADING_DAYS_PER_YEAR / horizon

    if decile_portfolio.empty:
        # groupby().apply() on a fully empty frame returns an ambiguous
        # empty DataFrame rather than a Series, which corrupts downstream
        # .mean() calls (build_scorecard_table would store a garbled
        # stringified Series in a single cell). Return an explicit empty
        # float Series instead.
        return pd.Series(dtype=float, name="long_short")

    def _sharpe(g: pd.DataFrame) -> float:
        r = g["long_short"].dropna()
        if len(r) < 2 or r.std() == 0:
            return np.nan
        return (r.mean() / r.std()) * np.sqrt(periods_per_year)

    return decile_portfolio.groupby("test_year").apply(_sharpe, include_groups=False)


def cumulative_long_short(decile_portfolio: pd.DataFrame) -> pd.Series:
    """Cumulative growth of $1 in the long-short portfolio over calendar
    time (simple compounding), indexed by formation date."""
    r = decile_portfolio.set_index("date")["long_short"].dropna()
    return (1.0 + r).cumprod()


# --- Stress windows ---


def stress_window_table(ic_series: pd.Series, sharpe_series: pd.Series) -> pd.DataFrame:
    """Mean IC and Sharpe within each named stress window (STRESS_WINDOWS).
    A window with no years present in the data (e.g. 2008, if the test
    range starts at 2013) gets an explicit NaN row with n_years=0 rather
    than being silently omitted, so a reader can't assume it was covered.
    """
    rows = []
    for name, (start, end) in STRESS_WINDOWS.items():
        years = [y for y in range(start, end + 1) if y in ic_series.index]
        rows.append({
            "window": name,
            "years_covered": years,
            "n_years": len(years),
            "mean_ic": ic_series.loc[years].mean() if years else np.nan,
            "mean_gross_sharpe": sharpe_series.loc[years].mean() if years and any(y in sharpe_series.index for y in years) else np.nan,
        })
    return pd.DataFrame(rows).set_index("window")


# --- Assembling one run's full scorecard, and the final tidy table ---


def score_one_run(preds: pd.DataFrame, *, horizon: int, n_deciles: int = 10) -> dict:
    """Everything for one (model, config, horizon) run: IC series, IC
    summary stats, decile portfolio, per-year gross Sharpe, stress-window
    table, cumulative L/S curve, and decile monotonicity."""
    ic = ic_by_year(preds)
    ic_stats = ic_summary_stats(ic)
    deciles = build_decile_portfolio(preds, horizon=horizon, n_deciles=n_deciles)
    sharpe = gross_sharpe_by_year(deciles, horizon=horizon)
    return {
        "ic_by_year": ic,
        "ic_stats": ic_stats,
        "decile_portfolio": deciles,
        "sharpe_by_year": sharpe,
        "cumulative_long_short": cumulative_long_short(deciles),
        "decile_monotonicity": decile_monotonicity(deciles, n_deciles),
        "stress_windows": stress_window_table(ic, sharpe),
    }


def build_scorecard_table(runs: dict[str, dict]) -> pd.DataFrame:
    """The final tidy dataframe -> main report table. One row per named
    run (e.g. "elastic_net__d_full_pool__5d", "xgboost__c_pruned__1d"),
    columns: mean IC, IC std, IR, t-stat, n active years, mean gross
    Sharpe, and whether the decile spread is monotonic.
    `runs` maps a run name to the dict returned by `score_one_run`.
    """
    rows = []
    for name, scored in runs.items():
        stats = scored["ic_stats"]
        sharpe = scored["sharpe_by_year"]
        mono = scored["decile_monotonicity"]
        is_monotonic = bool(mono.is_monotonic_increasing)
        rows.append({
            "run": name,
            "mean_ic": stats["mean_ic"],
            "ic_std": stats["ic_std"],
            "ir": stats["ir"],
            "t_stat": stats["t_stat"],
            "n_active_years": stats["n_years"],
            "mean_gross_sharpe": sharpe.mean(),
            "decile_monotonic": is_monotonic,
        })
    return pd.DataFrame(rows).set_index("run").sort_values("mean_ic", ascending=False)


# --- Figures ---


def plot_cumulative_long_short(runs: dict[str, dict], *, title: str = "Cumulative long-short growth of $1"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, scored in runs.items():
        curve = scored["cumulative_long_short"]
        if len(curve):
            ax.plot(curve.index, curve.values, label=name)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    return fig


def plot_ic_by_year(runs: dict[str, dict], *, title: str = "Mean daily Spearman IC by year"):
    import matplotlib.pyplot as plt

    table = pd.DataFrame({name: scored["ic_by_year"] for name, scored in runs.items()})
    fig, ax = plt.subplots(figsize=(10, 5))
    table.plot.bar(ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Mean daily IC")
    ax.legend(fontsize=8)
    return fig


def plot_decile_monotonicity(runs: dict[str, dict], *, title: str = "Mean return by decile"):
    import matplotlib.pyplot as plt

    table = pd.DataFrame({name: scored["decile_monotonicity"] for name, scored in runs.items()})
    fig, ax = plt.subplots(figsize=(9, 5))
    table.plot(ax=ax, marker="o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Decile (1 = lowest predicted, 10 = highest)")
    ax.set_ylabel("Mean realized return")
    ax.legend(fontsize=8)
    return fig


if __name__ == "__main__":
    # Smoke test against real saved predictions, if present.
    import glob

    files = glob.glob(str(_REPO_ROOT / "outputs" / "model_runs_trees" / "*__preds.parquet"))
    if not files:
        print("No preds files found under outputs/model_runs_trees/ -- run the tree-model script first.")
    else:
        for f in files:
            name = Path(f).name.replace("__preds.parquet", "")
            horizon = int(name.split("__")[-1].replace("d", ""))
            preds = pd.read_parquet(f)
            scored = score_one_run(preds, horizon=horizon)
            print(f"\n=== {name} ===")
            print("IC stats:", {k: round(v, 4) if isinstance(v, float) else v for k, v in scored["ic_stats"].items()})
            print("gross Sharpe by year:\n", scored["sharpe_by_year"].round(3))
            print("decile monotonicity:\n", scored["decile_monotonicity"].round(5))
