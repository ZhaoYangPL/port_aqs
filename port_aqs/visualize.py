"""Compact, publication-friendly analysis for PORT-AQS experiment outputs.

The default path deliberately reads only ``run_metrics.csv``.  Request traces
are large (and contain JSON diagnostics), so temporal trace plots are opt-in.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

# The analysis command must also work on headless experiment machines.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from scipy import stats

from .metrics import aggregate_runs


REFERENCE_POLICY = "rpm_aware"
DEFAULT_EXCLUDED_POLICIES = ("available",)
POLICY_ORDER = (
    "rpm_aware",
    "quality_cost",
    "static_latency",
    "min_latency_risk",
    "best_quality",
    "random",
    "rpm_aware_no_gamma",
    "rpm_aware_lambda_0",
    "static_latency_no_gamma",
    "available",
)
POLICY_LABELS = {
    "rpm_aware": "RPM-aware",
    "quality_cost": "Quality-cost",
    "static_latency": "Static latency",
    "min_latency_risk": "Min latency risk",
    "best_quality": "Best quality",
    "random": "Random",
    "rpm_aware_no_gamma": "RPM-aware (no gamma)",
    "rpm_aware_lambda_0": "RPM-aware (lambda=0)",
    "static_latency_no_gamma": "Static latency (no gamma)",
    "available": "Available",
}
POLICY_COLORS = {
    "rpm_aware": "#2563EB",
    "quality_cost": "#F59E0B",
    "static_latency": "#DC2626",
    "min_latency_risk": "#10B981",
    "best_quality": "#8B5CF6",
    "random": "#64748B",
    "rpm_aware_no_gamma": "#0891B2",
    "rpm_aware_lambda_0": "#D97706",
    "static_latency_no_gamma": "#B91C1C",
    "available": "#94A3B8",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    higher_is_better: bool
    multiplier: float = 1.0
    unit: str = ""
    number_format: str = ".3f"
    axis_scale: str = "linear"

    def format_value(self, value: float) -> str:
        if not np.isfinite(value):
            return "NA"
        return f"{value * self.multiplier:{self.number_format}}{self.unit}"


METRICS: dict[str, MetricSpec] = {
    "quality_mean": MetricSpec(
        "quality_mean", "Quality", True, number_format=".4f"
    ),
    "monetary_cost_mean": MetricSpec(
        "monetary_cost_mean",
        "Cost / 1k requests",
        False,
        multiplier=1000.0,
        unit=" USD",
        number_format=".3f",
    ),
    "slo_violation_rate": MetricSpec(
        "slo_violation_rate",
        "SLO violation",
        False,
        multiplier=100.0,
        unit="%",
        number_format=".2f",
    ),
    "quota_wait_p95": MetricSpec(
        "quota_wait_p95",
        "Quota wait p95",
        False,
        unit=" s",
        number_format=".2f",
        axis_scale="symlog",
    ),
    "quota_wait_p99": MetricSpec(
        "quota_wait_p99",
        "Quota wait p99",
        False,
        unit=" s",
        number_format=".2f",
        axis_scale="symlog",
    ),
    "e2e_p95": MetricSpec(
        "e2e_p95",
        "E2E p95",
        False,
        unit=" s",
        number_format=".2f",
        axis_scale="log",
    ),
    "e2e_p99": MetricSpec(
        "e2e_p99",
        "E2E p99",
        False,
        unit=" s",
        number_format=".2f",
        axis_scale="log",
    ),
    "cvar95_e2e": MetricSpec(
        "cvar95_e2e",
        "E2E CVaR95",
        False,
        unit=" s",
        number_format=".2f",
        axis_scale="log",
    ),
    "routing_hhi": MetricSpec(
        "routing_hhi", "Routing HHI", False, number_format=".3f"
    ),
    "rpm_binding_rate": MetricSpec(
        "rpm_binding_rate",
        "RPM binding rate",
        False,
        multiplier=100.0,
        unit="%",
        number_format=".2f",
    ),
}

SCORECARD_KEYS = (
    "quality_mean",
    "monetary_cost_mean",
    "slo_violation_rate",
    "quota_wait_p95",
    "e2e_p95",
    "cvar95_e2e",
)
STABILITY_KEYS = (
    "quality_mean",
    "monetary_cost_mean",
    "slo_violation_rate",
    "e2e_p95",
)
EFFECT_KEYS = (
    "quality_mean",
    "monetary_cost_mean",
    "slo_violation_rate",
    "quota_wait_p95",
    "e2e_p95",
    "cvar95_e2e",
)
EFFECT_DISPLAY = {
    "quality_mean": (100.0, "percentage points"),
    "monetary_cost_mean": (1000.0, "USD / 1k requests"),
    "slo_violation_rate": (100.0, "percentage points"),
    "quota_wait_p95": (1.0, "seconds"),
    "e2e_p95": (1.0, "seconds"),
    "cvar95_e2e": (1.0, "seconds"),
}
REQUIRED_COLUMNS = {
    "condition",
    "seed",
    "policy",
    "quota_mode",
    "quota_load",
    *SCORECARD_KEYS,
    "routing_hhi",
    "rpm_binding_rate",
}


def _policy_order(policies: Iterable[str]) -> list[str]:
    present = set(map(str, policies))
    ordered = [policy for policy in POLICY_ORDER if policy in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _policy_label(policy: str) -> str:
    return POLICY_LABELS.get(policy, policy.replace("_", " ").title())


def _policy_color(policy: str) -> str:
    return POLICY_COLORS.get(policy, "#475569")


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "condition"


def _mean_interval(values: np.ndarray) -> tuple[int, float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    count = int(finite.size)
    if count == 0:
        return 0, float("nan"), float("nan"), float("nan")
    mean = float(np.mean(finite))
    if count < 2:
        return count, mean, float("nan"), float("nan")
    half_width = float(stats.t.ppf(0.975, df=count - 1) * stats.sem(finite))
    return count, mean, mean - half_width, mean + half_width


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = (
        "route_share__",
        "rpm_utilization__",
        "tpm_utilization__",
        "actual_tpm_utilization__",
    )
    columns = {
        "seed",
        "quota_load",
        "requests",
        "completed",
        *METRICS,
    }
    columns.update(
        column for column in frame.columns if column.startswith(prefixes)
    )
    return sorted(columns.intersection(frame.columns))


def load_run_metrics(
    result_dirs: Sequence[str | Path],
    *,
    include_available: bool = False,
    policies: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load, filter, and validate one or more experiment result directories."""

    if not result_dirs:
        raise ValueError("at least one result directory is required")
    frames: list[pd.DataFrame] = []
    for raw_dir in result_dirs:
        result_dir = Path(raw_dir).expanduser().resolve()
        metrics_path = result_dir / "run_metrics.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing run metrics: {metrics_path}")
        frame = pd.read_csv(metrics_path)
        missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"{metrics_path} is missing columns: {missing}")
        for column in _numeric_columns(frame):
            try:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{metrics_path} has non-numeric values in {column!r}"
                ) from exc
        frame["_source_dir"] = str(result_dir)
        frames.append(frame)

    runs = pd.concat(frames, ignore_index=True, sort=False)
    if not include_available:
        runs = runs[~runs["policy"].isin(DEFAULT_EXCLUDED_POLICIES)]
    if policies is not None:
        requested = list(dict.fromkeys(map(str, policies)))
        unknown = sorted(set(requested).difference(set(runs["policy"])))
        if unknown:
            raise ValueError(f"requested policies are absent: {unknown}")
        runs = runs[runs["policy"].isin(requested)]
    if runs.empty:
        raise ValueError("no runs remain after policy filtering")

    keys = ["condition", "seed", "policy"]
    duplicates = runs[runs.duplicated(keys, keep=False)]
    if not duplicates.empty:
        examples = duplicates[keys].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"duplicate condition/seed/policy runs: {examples}")
    if {"requests", "completed"}.issubset(runs.columns):
        incomplete = runs[runs["requests"] != runs["completed"]]
        if not incomplete.empty:
            raise ValueError(
                f"{len(incomplete)} runs are not fully completed; refusing to plot"
            )

    for condition, group in runs.groupby("condition", sort=False):
        seed_sets = {
            str(policy): set(policy_runs["seed"].astype(int))
            for policy, policy_runs in group.groupby("policy", sort=False)
        }
        expected = next(iter(seed_sets.values()))
        mismatched = {
            policy: sorted(seeds)
            for policy, seeds in seed_sets.items()
            if seeds != expected
        }
        if mismatched:
            raise ValueError(
                f"condition {condition!r} does not use paired seed sets: {mismatched}"
            )
    return runs.sort_values(keys, kind="stable").reset_index(drop=True)


def _condition_metadata(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition, group in runs.groupby("condition", sort=False):
        modes = group["quota_mode"].dropna().astype(str).unique()
        loads = group["quota_load"].dropna().astype(float).unique()
        arrivals = (
            group["arrival"].dropna().astype(str).unique()
            if "arrival" in group.columns
            else np.asarray([], dtype=str)
        )
        if len(modes) != 1 or len(loads) != 1:
            raise ValueError(
                f"condition {condition!r} has inconsistent quota metadata"
            )
        rows.append(
            {
                "condition": str(condition),
                "quota_mode": modes[0],
                "quota_load": float(loads[0]),
                "arrival": arrivals[0] if len(arrivals) == 1 else "",
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["quota_load", "condition"], kind="stable"
    )


def build_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Return one compact, raw-valued row per condition and policy."""

    aggregate = aggregate_runs(runs)
    metadata = _condition_metadata(runs)
    aggregate = aggregate.merge(metadata, on="condition", how="left", validate="many_to_one")
    records: list[dict[str, object]] = []
    for _, row in aggregate.iterrows():
        record: dict[str, object] = {
            "condition": row["condition"],
            "quota_mode": row["quota_mode"],
            "quota_load": row["quota_load"],
            "policy": row["policy"],
            "runs": int(row["runs"]),
        }
        for key in SCORECARD_KEYS:
            for suffix in ("mean", "ci_low", "ci_high"):
                source = f"{key}__{suffix}"
                record[f"{key}__{suffix}"] = (
                    float(row[source]) if source in row and pd.notna(row[source]) else np.nan
                )
        records.append(record)
    summary = pd.DataFrame.from_records(records)
    policy_rank = {policy: index for index, policy in enumerate(POLICY_ORDER)}
    summary["_policy_rank"] = summary["policy"].map(policy_rank).fillna(999)
    summary = summary.sort_values(
        ["quota_load", "condition", "_policy_rank", "policy"], kind="stable"
    ).drop(columns="_policy_rank")
    return summary.reset_index(drop=True)


def build_paired_effects(
    runs: pd.DataFrame,
    *,
    reference: str = REFERENCE_POLICY,
    metric_keys: Sequence[str] = EFFECT_KEYS,
) -> pd.DataFrame:
    """Paired effects with a direction-normalised benefit.

    ``raw_difference`` is reference minus baseline. ``benefit`` changes the
    sign for lower-is-better metrics, so positive always favours the reference.
    Display-valued columns additionally apply the metric's multiplier.
    """

    records: list[dict[str, object]] = []
    metadata = _condition_metadata(runs).set_index("condition")
    for condition, group in runs.groupby("condition", sort=False):
        policies = _policy_order(group["policy"].unique())
        if reference not in policies:
            raise ValueError(
                f"reference policy {reference!r} is absent from {condition!r}"
            )
        for baseline in (policy for policy in policies if policy != reference):
            pair = group[group["policy"].isin([reference, baseline])]
            for metric_key in metric_keys:
                spec = METRICS[metric_key]
                wide = pair.pivot(index="seed", columns="policy", values=metric_key)
                wide = wide.reindex(columns=[reference, baseline]).dropna()
                raw = (
                    wide[reference].to_numpy(dtype=np.float64)
                    - wide[baseline].to_numpy(dtype=np.float64)
                )
                count, raw_mean, raw_low, raw_high = _mean_interval(raw)
                sign = 1.0 if spec.higher_is_better else -1.0
                benefit = raw * sign
                _, benefit_mean, benefit_low, benefit_high = _mean_interval(benefit)
                scale, display_unit = EFFECT_DISPLAY.get(
                    metric_key, (spec.multiplier, spec.unit.strip())
                )
                records.append(
                    {
                        "condition": condition,
                        "quota_mode": metadata.loc[condition, "quota_mode"],
                        "quota_load": float(metadata.loc[condition, "quota_load"]),
                        "metric": metric_key,
                        "metric_label": spec.label,
                        "reference": reference,
                        "baseline": baseline,
                        "pairs": count,
                        "raw_difference": raw_mean,
                        "raw_ci_low": raw_low,
                        "raw_ci_high": raw_high,
                        "benefit": benefit_mean,
                        "benefit_ci_low": benefit_low,
                        "benefit_ci_high": benefit_high,
                        "display_multiplier": scale,
                        "benefit_display": benefit_mean * scale,
                        "benefit_display_ci_low": benefit_low * scale,
                        "benefit_display_ci_high": benefit_high * scale,
                        "display_unit": display_unit,
                    }
                )
    return pd.DataFrame.from_records(records)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#0F172A",
            "axes.titlecolor": "#0F172A",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "font.size": 9,
            "grid.color": "#CBD5E1",
            "grid.alpha": 0.45,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def _condition_title(runs: pd.DataFrame, condition: str) -> str:
    group = runs[runs["condition"] == condition]
    mode = str(group["quota_mode"].iloc[0])
    load = float(group["quota_load"].iloc[0])
    seeds = group["seed"].nunique()
    return f"{condition} | {mode} | rho={load:.2f} | {seeds} paired seeds"


def _condition_suffix(condition: str, many_conditions: bool) -> str:
    return f"_{_slug(condition)}" if many_conditions else ""


def _save_figure(
    fig: plt.Figure,
    path: Path,
    *,
    pdf: PdfPages,
    dpi: int,
) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_scorecard(
    summary: pd.DataFrame,
    *,
    condition: str,
    reference: str,
) -> plt.Figure:
    subset = summary[summary["condition"] == condition].copy()
    order = _policy_order(subset["policy"])
    subset = subset.set_index("policy").loc[order].reset_index()
    rows, columns = len(subset), len(SCORECARD_KEYS)
    fig, ax = plt.subplots(figsize=(15.5, 1.0 * rows + 2.6))
    ax.set_xlim(-2.7, columns)
    ax.set_ylim(-1.15, rows + 0.85)
    ax.axis("off")
    cmap = plt.get_cmap("RdYlGn")

    for column_index, key in enumerate(SCORECARD_KEYS):
        spec = METRICS[key]
        values = subset[f"{key}__mean"].to_numpy(dtype=float)
        utility = values if spec.higher_is_better else -values
        finite = np.isfinite(utility)
        scores = np.full(rows, 0.5, dtype=float)
        if finite.any() and np.nanmax(utility) > np.nanmin(utility):
            scores[finite] = (utility[finite] - np.nanmin(utility)) / (
                np.nanmax(utility) - np.nanmin(utility)
            )
        for row_index, value in enumerate(values):
            y = rows - row_index - 1
            color = cmap(0.16 + 0.72 * scores[row_index])
            ax.add_patch(
                Rectangle(
                    (column_index, y),
                    1.0,
                    0.86,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.5,
                )
            )
            ax.text(
                column_index + 0.5,
                y + 0.43,
                spec.format_value(value),
                ha="center",
                va="center",
                color="#0F172A",
                fontsize=9,
                fontweight="bold" if subset.iloc[row_index]["policy"] == reference else "normal",
            )
        direction = "higher is better" if spec.higher_is_better else "lower is better"
        ax.text(
            column_index + 0.5,
            rows + 0.25,
            f"{spec.label}\n{direction}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#0F172A",
        )

    for row_index, policy in enumerate(subset["policy"]):
        y = rows - row_index - 1
        group = "main method" if policy == reference else (
            "core baseline" if policy in {"quality_cost", "static_latency"} else "diagnostic"
        )
        ax.text(
            -0.12,
            y + 0.43,
            f"{_policy_label(policy)}\n{group}",
            ha="right",
            va="center",
            color=_policy_color(policy),
            fontsize=9.5,
            fontweight="bold" if policy == reference else "normal",
        )
        if policy == reference:
            ax.add_patch(
                Rectangle(
                    (-2.55, y - 0.02),
                    columns + 2.55,
                    0.90,
                    fill=False,
                    edgecolor=_policy_color(policy),
                    linewidth=2.2,
                )
            )

    ax.text(
        -2.55,
        rows + 0.75,
        "Decision scorecard",
        fontsize=17,
        fontweight="bold",
        color="#0F172A",
        ha="left",
    )
    ax.text(
        -2.55,
        -0.68,
        "Cell colour is a within-condition rank, not a global success verdict. "
        "Run-level 95% intervals are shown in the detailed figures.",
        fontsize=8.5,
        color="#475569",
        ha="left",
    )
    return fig


def _plot_mean_point(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    policy: str,
) -> None:
    xerr = None
    yerr = None
    if all(np.isfinite([x, x_low, x_high])):
        xerr = np.asarray([[max(0.0, x - x_low)], [max(0.0, x_high - x)]])
    if all(np.isfinite([y, y_low, y_high])):
        yerr = np.asarray([[max(0.0, y - y_low)], [max(0.0, y_high - y)]])
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        markersize=8.5 if policy == REFERENCE_POLICY else 7,
        markerfacecolor=_policy_color(policy),
        markeredgecolor="white",
        markeredgewidth=1.0,
        ecolor=_policy_color(policy),
        elinewidth=1.4,
        capsize=2.5,
        zorder=5,
    )


def plot_tradeoffs(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    condition: str,
    reference: str,
) -> plt.Figure:
    condition_runs = runs[runs["condition"] == condition]
    condition_summary = summary[summary["condition"] == condition].set_index("policy")
    policies = _policy_order(condition_runs["policy"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    panels = (
        ("slo_violation_rate", "quality_mean", "Quality vs SLO risk"),
        ("monetary_cost_mean", "slo_violation_rate", "SLO risk vs monetary cost"),
    )
    offsets = {
        "rpm_aware": (6, 7),
        "quality_cost": (6, -13),
        "static_latency": (6, 7),
        "min_latency_risk": (6, -13),
        "best_quality": (6, 7),
        "random": (6, 7),
    }
    for ax, (x_key, y_key, title) in zip(axes, panels, strict=True):
        x_spec, y_spec = METRICS[x_key], METRICS[y_key]
        for policy in policies:
            policy_runs = condition_runs[condition_runs["policy"] == policy]
            x_values = policy_runs[x_key].to_numpy(dtype=float) * x_spec.multiplier
            y_values = policy_runs[y_key].to_numpy(dtype=float) * y_spec.multiplier
            alpha = 0.22 if policy in {reference, "quality_cost", "static_latency"} else 0.12
            ax.scatter(
                x_values,
                y_values,
                s=22,
                color=_policy_color(policy),
                alpha=alpha,
                linewidths=0,
                zorder=2,
            )
            row = condition_summary.loc[policy]
            x = float(row[f"{x_key}__mean"]) * x_spec.multiplier
            y = float(row[f"{y_key}__mean"]) * y_spec.multiplier
            _plot_mean_point(
                ax,
                x=x,
                y=y,
                x_low=float(row[f"{x_key}__ci_low"]) * x_spec.multiplier,
                x_high=float(row[f"{x_key}__ci_high"]) * x_spec.multiplier,
                y_low=float(row[f"{y_key}__ci_low"]) * y_spec.multiplier,
                y_high=float(row[f"{y_key}__ci_high"]) * y_spec.multiplier,
                policy=policy,
            )
            dx, dy = offsets.get(policy, (6, 6))
            ax.annotate(
                _policy_label(policy),
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.3,
                color=_policy_color(policy),
                fontweight="bold" if policy == reference else "normal",
            )
        ax.set_xlabel(f"{x_spec.label} ({x_spec.unit.strip() or 'score'})")
        ax.set_ylabel(f"{y_spec.label} ({y_spec.unit.strip() or 'score'})")
        ax.set_title(title)
        ax.grid(True, which="both")
        if x_key == "slo_violation_rate" and (condition_runs[x_key] > 0).all():
            ax.set_xscale("log")
        if y_key == "slo_violation_rate" and (condition_runs[y_key] > 0).all():
            ax.set_yscale("log")
    fig.suptitle(
        f"Trade-off map — {_condition_title(runs, condition)}\n"
        "Small dots are seeds; large dots and whiskers are run-level mean and 95% CI",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_paired_effects(
    effects: pd.DataFrame,
    *,
    condition: str,
    reference: str,
    baselines: Sequence[str] | None = None,
) -> plt.Figure:
    subset = effects[effects["condition"] == condition].copy()
    present = _policy_order(subset["baseline"].unique())
    if baselines is None:
        preferred = ["quality_cost", "static_latency"]
        baselines = [baseline for baseline in preferred if baseline in present]
        if not baselines:
            baselines = present
    else:
        baselines = [baseline for baseline in baselines if baseline in present]
    subset = subset[subset["baseline"].isin(baselines)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2))
    for ax, metric_key in zip(axes.flat, EFFECT_KEYS, strict=True):
        spec = METRICS[metric_key]
        metric_rows = subset[subset["metric"] == metric_key].set_index("baseline")
        y_positions = np.arange(len(baselines), dtype=float)
        for y, baseline in zip(y_positions, baselines, strict=True):
            if baseline not in metric_rows.index:
                continue
            row = metric_rows.loc[baseline]
            mean = float(row["benefit_display"])
            low = float(row["benefit_display_ci_low"])
            high = float(row["benefit_display_ci_high"])
            xerr = None
            if all(np.isfinite([mean, low, high])):
                xerr = np.asarray([[max(0.0, mean - low)], [max(0.0, high - mean)]])
            ax.errorbar(
                mean,
                y,
                xerr=xerr,
                fmt="o",
                color=_policy_color(baseline),
                markerfacecolor=_policy_color(reference),
                markeredgecolor="white",
                markersize=8,
                elinewidth=2,
                capsize=3,
                zorder=3,
            )
            ax.annotate(
                f"{mean:+.3g}",
                (mean, y),
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=8,
                color="#334155",
            )
        ax.axvline(0.0, color="#0F172A", linewidth=1.0, linestyle="--")
        ax.set_yticks(y_positions, [_policy_label(item) for item in baselines])
        ax.invert_yaxis()
        unit = (
            str(metric_rows.iloc[0]["display_unit"])
            if not metric_rows.empty
            else (spec.unit.strip() or "score")
        )
        ax.set_xlabel(f"Benefit to {_policy_label(reference)} ({unit})")
        ax.set_title(spec.label)
        ax.grid(True, axis="x")
    fig.suptitle(
        f"Paired effects — {_policy_label(reference)} against core baselines\n"
        "Common-seed Student-t 95% CI; every panel is oriented so right is better",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_seed_stability(
    runs: pd.DataFrame,
    *,
    condition: str,
    reference: str,
) -> plt.Figure:
    subset = runs[runs["condition"] == condition]
    policies = _policy_order(subset["policy"].unique())
    x = np.arange(len(policies), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, metric_key in zip(axes.flat, STABILITY_KEYS, strict=True):
        spec = METRICS[metric_key]
        wide = subset.pivot(index="seed", columns="policy", values=metric_key)
        wide = wide.reindex(columns=policies) * spec.multiplier
        for _, seed_values in wide.iterrows():
            ax.plot(x, seed_values, color="#94A3B8", alpha=0.28, linewidth=0.8, zorder=1)
            ax.scatter(x, seed_values, color="#94A3B8", alpha=0.28, s=10, zorder=1)
        for policy_index, policy in enumerate(policies):
            values = wide[policy].dropna().to_numpy(dtype=float)
            _, mean, low, high = _mean_interval(values)
            yerr = None
            if all(np.isfinite([mean, low, high])):
                yerr = np.asarray([[max(0.0, mean - low)], [max(0.0, high - mean)]])
            ax.errorbar(
                policy_index,
                mean,
                yerr=yerr,
                fmt="o",
                color=_policy_color(policy),
                markeredgecolor="white",
                markersize=8 if policy == reference else 6.5,
                capsize=2.5,
                zorder=4,
            )
        ax.set_xticks(x, [_policy_label(policy) for policy in policies], rotation=25, ha="right")
        ax.set_ylabel(f"{spec.label} ({spec.unit.strip() or 'score'})")
        ax.set_title(spec.label)
        if spec.axis_scale == "log" and (wide.to_numpy() > 0).all():
            ax.set_yscale("log")
        ax.grid(True, axis="y", which="both")
    fig.suptitle(
        f"Seed stability — {_condition_title(runs, condition)}\n"
        "Gray lines connect the same common-random-number seed",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def _arm_columns(runs: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in runs.columns if column.startswith(prefix)]
    return sorted(columns, key=lambda value: value.removeprefix(prefix))


def _annotated_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    value_format: str,
    vmax: float,
) -> None:
    image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(len(column_labels)), column_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title)
    threshold = vmax * 0.55
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text = "NA" if not np.isfinite(value) else format(value, value_format)
            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=6.8,
                color="white" if np.isfinite(value) and value > threshold else "#0F172A",
            )
    plt.colorbar(image, ax=ax, fraction=0.025, pad=0.02)


def plot_routing_mechanism(
    runs: pd.DataFrame,
    *,
    condition: str,
    reference: str,
) -> plt.Figure | None:
    subset = runs[runs["condition"] == condition]
    share_columns = _arm_columns(subset, "route_share__")
    utilization_columns = _arm_columns(subset, "rpm_utilization__")
    if not share_columns or not utilization_columns:
        return None
    policies = _policy_order(subset["policy"].unique())
    grouped = subset.groupby("policy", sort=False).mean(numeric_only=True)
    share = grouped.reindex(policies)[share_columns].to_numpy(dtype=float) * 100.0
    utilization = grouped.reindex(policies)[utilization_columns].to_numpy(dtype=float) * 100.0
    arm_labels = [column.removeprefix("route_share__") for column in share_columns]

    fig = plt.figure(figsize=(17, 10.5))
    grid = fig.add_gridspec(2, 2, width_ratios=(4.4, 1.35), hspace=0.38, wspace=0.23)
    ax_share = fig.add_subplot(grid[0, 0])
    ax_util = fig.add_subplot(grid[1, 0])
    ax_hhi = fig.add_subplot(grid[0, 1])
    ax_binding = fig.add_subplot(grid[1, 1])
    labels = [_policy_label(policy) for policy in policies]

    _annotated_heatmap(
        ax_share,
        share,
        row_labels=labels,
        column_labels=arm_labels,
        title="Mean route share (%)",
        value_format=".1f",
        vmax=max(1.0, float(np.nanmax(share))),
    )
    _annotated_heatmap(
        ax_util,
        utilization,
        row_labels=labels,
        column_labels=arm_labels,
        title="RPM utilization in the common arrival window (%)",
        value_format=".0f",
        vmax=max(100.0, float(np.nanmax(utilization))),
    )

    y = np.arange(len(policies), dtype=float)
    hhi = grouped.reindex(policies)["routing_hhi"].to_numpy(dtype=float)
    binding = grouped.reindex(policies)["rpm_binding_rate"].to_numpy(dtype=float) * 100.0
    for index, policy in enumerate(policies):
        size = 75 if policy == reference else 48
        ax_hhi.scatter(hhi[index], y[index], s=size, color=_policy_color(policy), zorder=3)
        ax_binding.scatter(binding[index], y[index], s=size, color=_policy_color(policy), zorder=3)
    ax_hhi.axvline(1.0 / len(arm_labels), color="#64748B", linestyle="--", linewidth=1)
    ax_hhi.set_title("Routing concentration (HHI)")
    ax_hhi.set_xlabel("HHI (uniform reference is dashed)")
    ax_binding.set_title("RPM binding rate")
    ax_binding.set_xlabel("Requests with RPM wait (%)")
    for ax in (ax_hhi, ax_binding):
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(True, axis="x")
    fig.suptitle(
        f"Routing mechanism — {_condition_title(runs, condition)}\n"
        "HHI and utilization are diagnostics, not universal higher/lower-is-better objectives",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    return fig


def plot_load_sweep(summary: pd.DataFrame, *, reference: str) -> plt.Figure | None:
    loads = np.sort(summary["quota_load"].dropna().unique())
    if len(loads) < 2:
        return None
    policies = _policy_order(summary["policy"].unique())
    keys = ("quality_mean", "slo_violation_rate", "e2e_p95", "monetary_cost_mean")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, key in zip(axes.flat, keys, strict=True):
        spec = METRICS[key]
        for policy in policies:
            rows = summary[summary["policy"] == policy].sort_values("quota_load")
            if rows.empty:
                continue
            x = rows["quota_load"].to_numpy(dtype=float)
            y = rows[f"{key}__mean"].to_numpy(dtype=float) * spec.multiplier
            low = rows[f"{key}__ci_low"].to_numpy(dtype=float) * spec.multiplier
            high = rows[f"{key}__ci_high"].to_numpy(dtype=float) * spec.multiplier
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.4 if policy == reference else 1.3,
                color=_policy_color(policy),
                label=_policy_label(policy),
            )
            if np.isfinite(low).all() and np.isfinite(high).all():
                ax.fill_between(x, low, high, color=_policy_color(policy), alpha=0.10)
        ax.set_title(spec.label)
        ax.set_xlabel("Nominal request-equivalent load (rho)")
        ax.set_ylabel(f"{spec.label} ({spec.unit.strip() or 'score'})")
        if spec.axis_scale == "log" and (summary[f"{key}__mean"] > 0).all():
            ax.set_yscale("log")
        ax.grid(True, which="both")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle(
        "Stable-load sweep — run-level means and 95% confidence bands",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def _trace_path(source_dir: Path, condition: str, seed: int, policy: str) -> Path:
    exact = source_dir / "traces" / f"{condition}__seed{seed}__{policy}.pkl.gz"
    if exact.is_file():
        return exact
    matches = list((source_dir / "traces").glob(f"*__seed{seed}__{policy}.pkl.gz"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"cannot find one trace for condition={condition}, seed={seed}, policy={policy} "
        f"under {source_dir / 'traces'}"
    )


def plot_trace_dynamics(
    runs: pd.DataFrame,
    *,
    condition: str,
    seed: int,
    rolling_window: int,
    reference: str,
) -> plt.Figure:
    subset = runs[(runs["condition"] == condition) & (runs["seed"] == seed)]
    if subset.empty:
        raise ValueError(f"seed {seed} is absent from condition {condition!r}")
    policies = _policy_order(subset["policy"].unique())
    series_by_policy: dict[str, pd.DataFrame] = {}
    needed = ["arrival_time", "admission_wait", "e2e_latency", "slo_violated", "quality"]
    for policy in policies:
        source_dir = Path(subset[subset["policy"] == policy]["_source_dir"].iloc[0])
        trace = pd.read_pickle(_trace_path(source_dir, condition, seed, policy))
        missing = sorted(set(needed).difference(trace.columns))
        if missing:
            raise ValueError(f"trace for {policy!r} is missing columns: {missing}")
        light = trace[needed].copy()
        del trace
        minimum_periods = max(5, rolling_window // 4)
        rolling = light.rolling(rolling_window, min_periods=minimum_periods)
        reduced = pd.DataFrame(
            {
                "minutes": (light["arrival_time"] - light["arrival_time"].iloc[0]) / 60.0,
                "slo_rate": rolling["slo_violated"].mean() * 100.0,
                "wait_mean": rolling["admission_wait"].mean(),
                "e2e_p95": rolling["e2e_latency"].quantile(0.95),
                "quality_mean": rolling["quality"].mean(),
            }
        )
        stride = max(1, math.ceil(len(reduced) / 800))
        series_by_policy[policy] = reduced.iloc[::stride].dropna()
        del light, rolling, reduced
        gc.collect()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    panels = (
        ("slo_rate", "Rolling SLO violation (%)", "linear"),
        ("wait_mean", "Rolling mean admission wait (s)", "symlog"),
        ("e2e_p95", "Rolling E2E p95 (s)", "log"),
        ("quality_mean", "Rolling mean quality", "linear"),
    )
    for ax, (column, label, scale) in zip(axes.flat, panels, strict=True):
        for policy, frame in series_by_policy.items():
            ax.plot(
                frame["minutes"],
                frame[column],
                color=_policy_color(policy),
                linewidth=2.2 if policy == reference else 1.1,
                alpha=0.95 if policy == reference else 0.75,
                label=_policy_label(policy),
            )
        ax.set_title(label)
        ax.set_xlabel("Arrival time (minutes)")
        ax.set_ylabel(label)
        values = np.concatenate(
            [frame[column].dropna().to_numpy(dtype=float) for frame in series_by_policy.values()]
        )
        if scale == "log" and (values > 0).all():
            ax.set_yscale("log")
        elif scale == "symlog":
            ax.set_yscale("symlog", linthresh=0.1)
        ax.grid(True, which="both")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle(
        f"Trace dynamics — {condition}, seed {seed}, rolling window={rolling_window}\n"
        "Trace loading is opt-in and processed one policy at a time",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def _formatted_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        record: dict[str, object] = {
            "Condition": row["condition"],
            "Policy": _policy_label(str(row["policy"])),
            "Seeds": int(row["runs"]),
        }
        for key in SCORECARD_KEYS:
            spec = METRICS[key]
            record[spec.label] = spec.format_value(float(row[f"{key}__mean"]))
        records.append(record)
    return pd.DataFrame.from_records(records)


def _write_html_report(
    path: Path,
    *,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    images: Sequence[Path],
    reference: str,
    pdf_name: str,
) -> None:
    formatted = _formatted_summary_table(summary)
    core_effects = effects[effects["baseline"].isin(["quality_cost", "static_latency"])].copy()
    if not core_effects.empty:
        core_effects["Comparison"] = core_effects["baseline"].map(
            lambda value: f"{_policy_label(reference)} vs {_policy_label(str(value))}"
        )
        core_effects["Metric"] = core_effects["metric"].map(
            lambda value: METRICS[str(value)].label
        )
        core_effects["Benefit (right-is-better convention)"] = core_effects.apply(
            lambda row: (
                f"{float(row['benefit_display']):+.4g} "
                f"[{float(row['benefit_display_ci_low']):+.4g}, "
                f"{float(row['benefit_display_ci_high']):+.4g}] "
                f"{row['display_unit']}"
            ),
            axis=1,
        )
        effect_display = core_effects[
            ["condition", "Comparison", "Metric", "Benefit (right-is-better convention)", "pairs"]
        ].rename(columns={"condition": "Condition", "pairs": "Pairs"})
    else:
        effect_display = pd.DataFrame()

    image_html = "\n".join(
        f'<figure><img src="{html.escape(image.name)}" alt="{html.escape(image.stem)}">'
        f"<figcaption>{html.escape(image.stem.replace('_', ' '))}</figcaption></figure>"
        for image in images
    )
    table_html = formatted.to_html(index=False, border=0, classes="metrics-table")
    effects_html = (
        effect_display.to_html(index=False, border=0, classes="metrics-table")
        if not effect_display.empty
        else "<p>No paired core-baseline comparison is available.</p>"
    )
    source_dirs = sorted(set(runs["_source_dir"].astype(str)))
    sources_html = "".join(f"<li><code>{html.escape(item)}</code></li>" for item in source_dirs)
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PORT-AQS 可视化分析</title>
<style>
body {{ max-width: 1500px; margin: 0 auto; padding: 32px; font-family: Inter, "Microsoft YaHei", sans-serif; color: #0f172a; background: #f8fafc; }}
h1, h2 {{ color: #0f172a; }}
.notice {{ background: #fff7ed; border-left: 5px solid #f59e0b; padding: 14px 18px; margin: 18px 0; }}
.links a {{ display: inline-block; margin-right: 16px; }}
figure {{ margin: 28px 0; padding: 14px; background: white; border: 1px solid #e2e8f0; border-radius: 10px; }}
img {{ width: 100%; height: auto; }}
figcaption {{ color: #475569; margin-top: 8px; }}
.metrics-table {{ border-collapse: collapse; width: 100%; background: white; font-size: 13px; }}
.metrics-table th, .metrics-table td {{ border: 1px solid #e2e8f0; padding: 7px 9px; text-align: right; }}
.metrics-table th:first-child, .metrics-table td:first-child, .metrics-table th:nth-child(2), .metrics-table td:nth-child(2) {{ text-align: left; }}
code {{ word-break: break-all; }}
</style>
</head>
<body>
<h1>PORT-AQS 可视化分析</h1>
<p>默认只读取 run-level CSV；没有批量载入大型 trace。图中 95% CI 的独立统计单位是 seed，而不是单个请求。</p>
<div class="notice"><strong>整体结论未自动判定：</strong>可视化输入没有预先声明“最大允许质量下降”和“最大允许成本上升”。图表只呈现权衡与配对区间，不能据此自动写成 Stage 1 成功。</div>
<p class="links"><a href="{html.escape(pdf_name)}">打开单文件 PDF</a> <a href="analysis_summary.csv">下载精简汇总 CSV</a> <a href="paired_effects.csv">下载配对效应 CSV</a></p>
<h2>建议阅读顺序</h2>
<ol><li>Scorecard：先看各策略绝对量级。</li><li>Trade-offs：看质量、SLO、成本的权衡。</li><li>Paired effects：只用共同 seed 判断主方法相对核心 baseline 的差值。</li><li>Seed stability：检查结论是否由少数 seed 驱动。</li><li>Routing mechanism：解释 route share、利用率与集中度；这些是诊断量，不做统一好坏排名。</li></ol>
<h2>精简数值表</h2>
{table_html}
<h2>主方法相对核心 baseline 的配对效应</h2>
<p>Benefit 已统一定向为正值更有利于 {_policy_label(reference)}；括号为 paired Student-t 95% CI。</p>
{effects_html}
<h2>图表</h2>
{image_html}
<h2>数据来源</h2><ul>{sources_html}</ul>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def generate_analysis(
    result_dirs: Sequence[str | Path],
    *,
    output_dir: str | Path | None = None,
    reference: str = REFERENCE_POLICY,
    include_available: bool = False,
    policies: Sequence[str] | None = None,
    paired_baselines: Sequence[str] | None = None,
    trace_seed: int | None = None,
    rolling_window: int = 250,
    dpi: int = 160,
) -> list[Path]:
    """Generate a compact report and return every created artifact path."""

    if rolling_window < 10:
        raise ValueError("rolling_window must be at least 10")
    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    runs = load_run_metrics(
        result_dirs, include_available=include_available, policies=policies
    )
    if reference not in set(runs["policy"]):
        raise ValueError(f"reference policy {reference!r} is absent")
    summary = build_summary(runs)
    effects = build_paired_effects(runs, reference=reference)

    resolved_dirs = [Path(item).expanduser().resolve() for item in result_dirs]
    if output_dir is None:
        if len(resolved_dirs) == 1:
            destination = resolved_dirs[0] / "analysis"
        else:
            common = Path(os.path.commonpath([str(item.parent) for item in resolved_dirs]))
            destination = common / "analysis_combined"
    else:
        destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    summary_path = destination / "analysis_summary.csv"
    effects_path = destination / "paired_effects.csv"
    summary.to_csv(summary_path, index=False)
    effects.to_csv(effects_path, index=False)
    artifacts: list[Path] = [summary_path, effects_path]

    _configure_style()
    conditions = list(_condition_metadata(runs)["condition"])
    many_conditions = len(conditions) > 1
    images: list[Path] = []
    pdf_path = destination / "analysis_report.pdf"
    with PdfPages(pdf_path) as pdf:
        for condition in conditions:
            suffix = _condition_suffix(condition, many_conditions)
            figures = (
                (f"00_scorecard{suffix}.png", plot_scorecard(summary, condition=condition, reference=reference)),
                (f"01_tradeoffs{suffix}.png", plot_tradeoffs(runs, summary, condition=condition, reference=reference)),
                (
                    f"02_paired_effects{suffix}.png",
                    plot_paired_effects(
                        effects,
                        condition=condition,
                        reference=reference,
                        baselines=paired_baselines,
                    ),
                ),
                (f"03_seed_stability{suffix}.png", plot_seed_stability(runs, condition=condition, reference=reference)),
            )
            for filename, figure in figures:
                path = destination / filename
                _save_figure(figure, path, pdf=pdf, dpi=dpi)
                images.append(path)
            routing = plot_routing_mechanism(runs, condition=condition, reference=reference)
            if routing is not None:
                path = destination / f"04_routing_mechanism{suffix}.png"
                _save_figure(routing, path, pdf=pdf, dpi=dpi)
                images.append(path)
            if trace_seed is not None:
                trace_figure = plot_trace_dynamics(
                    runs,
                    condition=condition,
                    seed=trace_seed,
                    rolling_window=rolling_window,
                    reference=reference,
                )
                path = destination / f"06_trace_dynamics_seed{trace_seed}{suffix}.png"
                _save_figure(trace_figure, path, pdf=pdf, dpi=dpi)
                images.append(path)
        sweep = plot_load_sweep(summary, reference=reference)
        if sweep is not None:
            path = destination / "05_load_sweep.png"
            _save_figure(sweep, path, pdf=pdf, dpi=dpi)
            images.append(path)
    artifacts.extend(images)
    artifacts.append(pdf_path)

    html_path = destination / "analysis_report.html"
    _write_html_report(
        html_path,
        runs=runs,
        summary=summary,
        effects=effects,
        images=images,
        reference=reference,
        pdf_name=pdf_path.name,
    )
    artifacts.append(html_path)

    manifest_path = destination / "analysis_manifest.json"
    manifest = {
        "source_result_dirs": [str(item) for item in resolved_dirs],
        "reference_policy": reference,
        "included_policies": _policy_order(runs["policy"].unique()),
        "conditions": conditions,
        "trace_seed": trace_seed,
        "rolling_window": rolling_window if trace_seed is not None else None,
        "generated_files": [path.name for path in artifacts],
        "overall_verdict": "not_assigned_without_predeclared_quality_and_cost_tolerances",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    artifacts.append(manifest_path)
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a compact PORT-AQS visual analysis report."
    )
    parser.add_argument(
        "result_dirs",
        nargs="+",
        type=Path,
        help="one or more experiment result directories containing run_metrics.csv",
    )
    parser.add_argument("--output", type=Path, help="analysis output directory")
    parser.add_argument(
        "--reference",
        default=REFERENCE_POLICY,
        help=f"reference policy for paired effects (default: {REFERENCE_POLICY})",
    )
    parser.add_argument("--policies", nargs="+", help="optional policy subset")
    parser.add_argument(
        "--paired-baselines",
        nargs="+",
        help="baselines shown in the paired-effects figure (default: core baselines)",
    )
    parser.add_argument(
        "--include-available",
        action="store_true",
        help="include the available heuristic (excluded by default)",
    )
    parser.add_argument(
        "--trace-seed",
        type=int,
        help="opt in to per-request rolling diagnostics for one seed",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=250,
        help="request window for optional trace diagnostics (default: 250)",
    )
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = generate_analysis(
        args.result_dirs,
        output_dir=args.output,
        reference=args.reference,
        include_available=args.include_available,
        policies=args.policies,
        paired_baselines=args.paired_baselines,
        trace_seed=args.trace_seed,
        rolling_window=args.rolling_window,
        dpi=args.dpi,
    )
    output_dir = artifacts[0].parent
    print(f"PORT-AQS analysis: {output_dir}")
    print(f"Generated {len(artifacts)} artifacts")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
