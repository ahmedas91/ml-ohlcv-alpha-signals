"""Build report-only tables and figures from frozen elastic-net artifacts.

Run after ``uv run python run_models.py``. The script does not fit or select
models; it summarizes the four predeclared specifications at both horizons.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

MODEL_DIR = Path("outputs/model_runs")
OUT_DIR = Path("outputs/report")
ABLATION_SUMMARY = Path("outputs/ablation_normalization/summary.csv")
TEST_YEARS = list(range(2013, 2026))
TRADING_DAYS = 252
ZERO_TOL = 1e-8

CONFIGS = {
    "a_baseline": "Baseline",
    "b_baseline_interactions": "Baseline + interactions",
    "c_pruned": "Pruned technicals",
    "d_full_pool": "Full technicals",
}
COLORS = {
    "a_baseline": "#4C78A8",
    "b_baseline_interactions": "#F58518",
    "c_pruned": "#54A24B",
    "d_full_pool": "#E45756",
}


def _preds_path(config: str, horizon: int) -> Path:
    return MODEL_DIR / f"{config}__{horizon}d__preds.parquet"


def _coefs_path(config: str, horizon: int) -> Path:
    return MODEL_DIR / f"{config}__{horizon}d__coefs.parquet"


def _params_path(config: str, horizon: int) -> Path:
    return MODEL_DIR / f"{config}__{horizon}d__params.parquet"


def _load_coefs(config: str, horizon: int) -> pd.Series:
    coefs = pd.read_parquet(_coefs_path(config, horizon))["coef"]
    coefs.index = coefs.index.set_names(["test_year", "feature"])
    return coefs


def _daily_ic(preds: pd.DataFrame) -> pd.DataFrame:
    """Return one cross-sectional Spearman IC per active test date."""

    def corr(day: pd.DataFrame) -> float:
        if day["yhat"].nunique() < 2 or day["y"].nunique() < 2:
            return np.nan
        return float(day["y"].corr(day["yhat"], method="spearman"))

    daily = (
        preds.groupby(["test_year", "date"], sort=True)
        .apply(corr, include_groups=False)
        .rename("ic")
        .reset_index()
    )
    return daily


def _decile_returns(preds: pd.DataFrame, horizon: int, offset: int = 0) -> pd.DataFrame:
    """Return equal-weight decile returns on non-overlapping formation dates."""
    if offset < 0 or offset >= horizon:
        raise ValueError(f"offset must be in [0, {horizon}), got {offset}")
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for test_year, year in preds.groupby("test_year", sort=True):
        dates = np.sort(year["date"].unique())
        keep_dates = set(dates[offset::horizon])
        for date, day in year.groupby("date", sort=True):
            if date not in keep_dates or day["yhat"].nunique() < 2:
                continue
            rank = day["yhat"].rank(method="first")
            decile = pd.qcut(rank, q=10, labels=False) + 1
            means = day.groupby(decile)["y"].mean()
            row: dict[str, float | int | pd.Timestamp] = {
                "test_year": int(test_year),
                "date": pd.Timestamp(date),
            }
            for d in range(1, 11):
                row[f"decile_{d}"] = float(means.loc[d])
            row["long_short"] = float(means.loc[10] - means.loc[1])
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize_run(
    config: str,
    horizon: int,
    preds: pd.DataFrame,
    daily_ic: pd.DataFrame,
    deciles: pd.DataFrame,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    yearly_ic = daily_ic.groupby("test_year")["ic"].mean().reindex(TEST_YEARS)
    active = yearly_ic.dropna()
    all_year = yearly_ic.fillna(0.0)

    period_return = 0.5 * deciles["long_short"]
    periods_per_year = TRADING_DAYS / horizon
    gross_sharpe = (
        period_return.mean() / period_return.std(ddof=1) * np.sqrt(periods_per_year)
        if len(period_return) > 1 and period_return.std(ddof=1) > 0
        else np.nan
    )

    coefs = _load_coefs(config, horizon)
    nonzero = (
        (coefs.abs() > ZERO_TOL).groupby(level="test_year").sum().reindex(TEST_YEARS, fill_value=0)
    )
    yearly_table = pd.DataFrame(
        {
            "config": config,
            "horizon": horizon,
            "test_year": TEST_YEARS,
            "ic": yearly_ic.to_numpy(),
            "n_nonzero": nonzero.to_numpy(),
        }
    )

    decile_cols = [f"decile_{d}" for d in range(1, 11)]
    yearly_deciles = deciles.groupby("test_year")[decile_cols].mean()
    profile = pd.DataFrame(
        {
            "config": config,
            "horizon": horizon,
            "decile": range(1, 11),
            "mean_return": yearly_deciles.mean().to_numpy(),
            "se_across_years": (
                yearly_deciles.std(ddof=1) / np.sqrt(yearly_deciles.notna().sum())
            ).to_numpy(),
            "n_years": yearly_deciles.notna().sum().to_numpy(),
        }
    )

    row: dict[str, float | int | str] = {
        "config": config,
        "label": CONFIGS[config],
        "horizon": horizon,
        "mean_ic_active": float(active.mean()),
        "ic_std_active": float(active.std(ddof=1)),
        "n_active_years": int(active.shape[0]),
        "mean_ic_all_years_null_as_zero": float(all_year.mean()),
        "positive_ic_years": int((active > 0).sum()),
        "gross_sharpe": float(gross_sharpe),
        "mean_long_short_bps_100pct_gross": float(period_return.mean() * 10_000),
        "n_formations": int(deciles.shape[0]),
        "mean_test_return": float(preds["y"].mean()),
    }
    return row, yearly_table, profile


def _plot_yearly_ic(yearly: pd.DataFrame, *, notebook_mode: bool = False) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    for ax, horizon in zip(axes, (1, 5), strict=True):
        panel = yearly[yearly["horizon"] == horizon]
        for config, label in CONFIGS.items():
            data = panel[panel["config"] == config].set_index("test_year")
            ax.plot(
                data.index,
                data["ic"],
                color=COLORS[config],
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                label=label,
            )
            null_years = data.index[data["ic"].isna()]
            ax.scatter(
                null_years,
                np.full(len(null_years), -0.0015),
                color=COLORS[config],
                marker="x",
                s=22,
            )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_ylabel("Mean daily IC")
        ax.set_title(f"{horizon}-day horizon")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(ncol=2, frameon=False, fontsize=8)
    axes[1].set_xlabel("Test year (× denotes a null model)")
    axes[1].set_xticks(TEST_YEARS)
    fig.tight_layout()
    if notebook_mode:
        plt.show()
    else:
        fig.savefig(OUT_DIR / "yearly_ic.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / "yearly_ic.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _plot_decile_profiles(profiles: pd.DataFrame, *, notebook_mode: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True)
    for ax, horizon in zip(axes, (1, 5), strict=True):
        panel = profiles[profiles["horizon"] == horizon]
        for config, label in CONFIGS.items():
            data = panel[panel["config"] == config]
            ax.plot(
                data["decile"],
                10_000 * data["mean_return"],
                color=COLORS[config],
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                label=label,
            )
            ci = 1.96 * 10_000 * data["se_across_years"]
            ax.fill_between(
                data["decile"],
                10_000 * data["mean_return"] - ci,
                10_000 * data["mean_return"] + ci,
                color=COLORS[config],
                alpha=0.08,
                linewidth=0,
            )
        ax.set_title(f"{horizon}-day horizon")
        ax.set_xlabel("Predicted-return decile")
        ax.set_xticks(range(1, 11))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean realized return (bp)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    if notebook_mode:
        plt.show()
    else:
        fig.savefig(OUT_DIR / "decile_profiles.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / "decile_profiles.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _plot_coefficient_stability(*, notebook_mode: bool = False) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.0))
    for ax, horizon in zip(axes, (1, 5), strict=True):
        coefs = _load_coefs("c_pruned", horizon)
        matrix = coefs.unstack("test_year").reindex(columns=TEST_YEARS, fill_value=0.0)
        importance = matrix.abs().mean(axis=1)
        top = importance.nlargest(12).index
        shown = matrix.loc[top]
        limit = float(np.nanmax(np.abs(shown.to_numpy())))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = ax.imshow(shown, cmap="RdBu_r", aspect="auto", norm=norm)
        nonzero = (matrix.abs() > ZERO_TOL).sum(axis=0)
        ax.set_xticks(range(len(TEST_YEARS)))
        ax.set_xticklabels(
            [f"{str(year)[-2:]}\n{int(nonzero.loc[year])}" for year in TEST_YEARS],
            fontsize=6,
        )
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top, fontsize=7)
        ax.set_title(f"{horizon}-day horizon")
        ax.set_xlabel("Test year / active coefficients", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

        tidy = (
            shown.rename_axis(index="feature", columns="test_year")
            .stack(future_stack=True)
            .rename("coefficient")
            .reset_index()
        )
        tidy.insert(0, "horizon", horizon)
        rows.append(tidy)
    fig.tight_layout()
    if notebook_mode:
        plt.show()
    else:
        fig.savefig(OUT_DIR / "coefficient_stability.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / "coefficient_stability.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    return pd.concat(rows, ignore_index=True)


def _plot_tuning_complexity(*, notebook_mode: bool = False) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.2), sharex=True)
    for ax, horizon in zip(axes, (1, 5), strict=True):
        params = pd.read_parquet(_params_path("c_pruned", horizon)).reindex(TEST_YEARS)
        coefs = _load_coefs("c_pruned", horizon)
        nonzero = (coefs.abs() > ZERO_TOL).groupby(level="test_year").sum().reindex(TEST_YEARS)
        ax.bar(TEST_YEARS, nonzero, color="#9ecae1", label="Nonzero coefficients")
        ax.set_ylabel("Active coefficients")
        ax.set_title(f"Pruned technical model, {horizon}-day horizon")
        ax2 = ax.twinx()
        ax2.plot(
            TEST_YEARS,
            params["l1_ratio"],
            color="#d62728",
            marker="o",
            linewidth=1.1,
            label=r"Selected $\ell_1$ ratio",
        )
        ax2.set_ylabel(r"$\ell_1$ ratio", color="#d62728")
        ax2.set_ylim(-0.05, 1.05)
        tidy = params.reset_index().rename(columns={"index": "test_year"})
        tidy["horizon"] = horizon
        tidy["n_nonzero"] = nonzero.to_numpy()
        rows.append(tidy)
    axes[-1].set_xlabel("Test year")
    axes[-1].set_xticks(TEST_YEARS)
    fig.tight_layout()
    if notebook_mode:
        plt.show()
    else:
        fig.savefig(OUT_DIR / "tuning_complexity.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / "tuning_complexity.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    """Create all report tables and figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    required = [_preds_path(config, h) for config in CONFIGS for h in (1, 5)]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing frozen prediction artifacts. Run `uv run python run_models.py` first: "
            + ", ".join(str(path) for path in missing)
        )

    metric_rows: list[dict[str, float | int | str]] = []
    yearly_tables: list[pd.DataFrame] = []
    profiles: list[pd.DataFrame] = []
    stress_rows: list[dict[str, float | int | str]] = []
    offset_rows: list[dict[str, float | int | str]] = []

    for config in CONFIGS:
        for horizon in (1, 5):
            preds = pd.read_parquet(_preds_path(config, horizon))
            daily_ic = _daily_ic(preds)
            deciles = _decile_returns(preds, horizon)
            row, yearly, profile = _summarize_run(
                config,
                horizon,
                preds,
                daily_ic,
                deciles,
            )
            metric_rows.append(row)
            yearly_tables.append(yearly)
            profiles.append(profile)

            ic_by_year = yearly.set_index("test_year")["ic"]
            spread_by_year = deciles.groupby("test_year")["long_short"]
            for year in (2020, 2022, 2025):
                spread = (
                    0.5 * spread_by_year.get_group(year) if year in spread_by_year.groups else None
                )
                sharpe = (
                    float(spread.mean() / spread.std(ddof=1) * np.sqrt(TRADING_DAYS / horizon))
                    if spread is not None and len(spread) > 1 and spread.std(ddof=1) > 0
                    else np.nan
                )
                stress_rows.append(
                    {
                        "config": config,
                        "horizon": horizon,
                        "test_year": year,
                        "ic": ic_by_year.get(year, np.nan),
                        "gross_sharpe": sharpe,
                    }
                )

            for offset in range(horizon):
                offset_deciles = _decile_returns(preds, horizon, offset=offset)
                offset_return = 0.5 * offset_deciles["long_short"]
                offset_sharpe = (
                    float(
                        offset_return.mean()
                        / offset_return.std(ddof=1)
                        * np.sqrt(TRADING_DAYS / horizon)
                    )
                    if len(offset_return) > 1 and offset_return.std(ddof=1) > 0
                    else np.nan
                )
                offset_rows.append(
                    {
                        "config": config,
                        "horizon": horizon,
                        "offset": offset,
                        "gross_sharpe": offset_sharpe,
                        "mean_long_short_bps_100pct_gross": float(offset_return.mean() * 10_000),
                        "n_formations": int(offset_deciles.shape[0]),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    yearly = pd.concat(yearly_tables, ignore_index=True)
    decile_profiles = pd.concat(profiles, ignore_index=True)
    stress = pd.DataFrame(stress_rows)
    offset_sensitivity = pd.DataFrame(offset_rows)

    metrics.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    yearly.to_csv(OUT_DIR / "yearly_ic.csv", index=False)
    decile_profiles.to_csv(OUT_DIR / "decile_profiles.csv", index=False)
    stress.to_csv(OUT_DIR / "stress_years.csv", index=False)
    offset_sensitivity.to_csv(OUT_DIR / "offset_sensitivity.csv", index=False)
    ablation = pd.read_csv(ABLATION_SUMMARY)
    normalization_cols = [
        "config",
        "daily_spearman_ic__paired_rank",
        "daily_spearman_ic__paired_standardize",
        "daily_spearman_ic__paired_delta_std_minus_rank",
        "daily_spearman_ic__n_paired_years",
    ]
    ablation[normalization_cols].to_csv(
        OUT_DIR / "normalization_ablation.csv",
        index=False,
    )

    _plot_yearly_ic(yearly)
    _plot_decile_profiles(decile_profiles)
    _plot_coefficient_stability().to_csv(
        OUT_DIR / "coefficient_stability.csv",
        index=False,
    )
    _plot_tuning_complexity().to_csv(OUT_DIR / "tuning_complexity.csv", index=False)
    print(f"Wrote report artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
