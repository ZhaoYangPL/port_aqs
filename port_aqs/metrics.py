"""Metrics and paired uncertainty summaries for PORT-AQS experiments."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

TRACE_REQUIRED = {
    "request_id",
    "model_id",
    "arrival_time",
    "dispatch_time",
    "completion_time",
    "admission_wait",
    "api_latency",
    "quality",
    "monetary_cost",
    "predicted_violation",
    "rpm_wait",
    "tpm_wait",
    "predicted_token_demand",
    "reserved_token_demand",
    "actual_token_demand",
}


def _quantile(values: np.ndarray, probability: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, probability)) if finite.size else float("nan")


def _mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def expected_calibration_error(
    predicted_probability: Sequence[float],
    outcome: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    prediction = np.clip(np.asarray(predicted_probability, dtype=np.float64), 0.0, 1.0)
    observed = np.asarray(outcome, dtype=np.float64)
    mask = np.isfinite(prediction) & np.isfinite(observed)
    prediction, observed = prediction[mask], observed[mask]
    if not prediction.size:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(prediction, edges[1:-1]), bins - 1)
    result = 0.0
    for index in range(bins):
        members = assignments == index
        if np.any(members):
            result += float(members.mean()) * abs(
                float(prediction[members].mean()) - float(observed[members].mean())
            )
    return result


def _endpoint_value(endpoint: Any, field: str, fallback: float) -> float:
    if isinstance(endpoint, Mapping):
        return float(endpoint.get(field, fallback))
    return float(getattr(endpoint, field, fallback))


def _endpoint_first(endpoint: Any, fields: Sequence[str], fallback: float) -> float:
    """Read the first available endpoint field across core API revisions."""

    for field in fields:
        if isinstance(endpoint, Mapping) and field in endpoint:
            return float(endpoint[field])
        if not isinstance(endpoint, Mapping) and hasattr(endpoint, field):
            return float(getattr(endpoint, field))
    return float(fallback)


def compute_metrics(
    trace: pd.DataFrame,
    *,
    endpoints: Mapping[str, Any] | None = None,
    calibration_bins: int = 10,
) -> dict[str, float | int]:
    """Compute one policy/seed result from its drained request trace."""

    missing = sorted(TRACE_REQUIRED.difference(trace.columns))
    if missing:
        raise ValueError(f"trace is missing required columns: {missing}")
    if trace.empty:
        return {"requests": 0}

    data = trace.copy()
    if "status" in data:
        completed = data[data["status"] == "completed"].copy()
    else:
        completed = data.copy()
    if completed.empty:
        raise ValueError("drained trace contains no completed requests")

    e2e = (
        completed["e2e_latency"].to_numpy(dtype=np.float64)
        if "e2e_latency" in completed
        else (
            completed["completion_time"].to_numpy(dtype=np.float64)
            - completed["arrival_time"].to_numpy(dtype=np.float64)
        )
    )
    wait = completed["admission_wait"].to_numpy(dtype=np.float64)
    rpm_wait = completed["rpm_wait"].to_numpy(dtype=np.float64)
    tpm_wait = completed["tpm_wait"].to_numpy(dtype=np.float64)
    response = completed["api_latency"].to_numpy(dtype=np.float64)
    deadline = (
        completed["deadline"].to_numpy(dtype=np.float64)
        if "deadline" in completed
        else np.full(len(completed), np.inf)
    )
    violated = (
        completed["slo_violated"].to_numpy(dtype=np.float64)
        if "slo_violated" in completed
        else (e2e > deadline).astype(np.float64)
    )
    p95 = _quantile(e2e, 0.95)
    tail = e2e[np.isfinite(e2e) & (e2e >= p95)] if np.isfinite(p95) else np.asarray([])

    result: dict[str, float | int] = {
        "requests": int(len(data)),
        "completed": int(len(completed)),
        "quality_mean": _mean(completed["quality"].to_numpy(dtype=np.float64)),
        "monetary_cost_total": float(completed["monetary_cost"].sum()),
        "monetary_cost_mean": _mean(
            completed["monetary_cost"].to_numpy(dtype=np.float64)
        ),
        "quota_wait_mean": _mean(wait),
        "quota_wait_p95": _quantile(wait, 0.95),
        "quota_wait_p99": _quantile(wait, 0.99),
        "rpm_wait_mean": _mean(rpm_wait),
        "rpm_wait_p95": _quantile(rpm_wait, 0.95),
        "rpm_wait_p99": _quantile(rpm_wait, 0.99),
        "tpm_wait_mean": _mean(tpm_wait),
        "tpm_wait_p95": _quantile(tpm_wait, 0.95),
        "tpm_wait_p99": _quantile(tpm_wait, 0.99),
        "rpm_binding_rate": _mean(
            completed["rpm_binding"].to_numpy(dtype=np.float64)
        ) if "rpm_binding" in completed else float("nan"),
        "tpm_binding_rate": _mean(
            completed["tpm_binding"].to_numpy(dtype=np.float64)
        ) if "tpm_binding" in completed else float("nan"),
        "api_response_mean": _mean(response),
        "api_response_p95": _quantile(response, 0.95),
        "e2e_mean": _mean(e2e),
        "e2e_p95": p95,
        "e2e_p99": _quantile(e2e, 0.99),
        "slo_violation_rate": _mean(violated),
        "cvar95_e2e": _mean(tail),
        "risk_brier": _mean(
            (
                completed["predicted_violation"].to_numpy(dtype=np.float64)
                - violated
            )
            ** 2
        ),
        "risk_ece": expected_calibration_error(
            completed["predicted_violation"], violated, bins=calibration_bins
        ),
    }
    predicted_tokens = completed["predicted_token_demand"].to_numpy(dtype=np.float64)
    reserved_tokens = completed["reserved_token_demand"].to_numpy(dtype=np.float64)
    actual_tokens = completed["actual_token_demand"].to_numpy(dtype=np.float64)
    prediction_error = actual_tokens - predicted_tokens
    reservation_error = actual_tokens - reserved_tokens
    result.update(
        {
            "token_prediction_bias": _mean(prediction_error),
            "token_prediction_mae": _mean(np.abs(prediction_error)),
            "token_prediction_rmse": float(
                np.sqrt(_mean(np.square(prediction_error)))
            ),
            "token_prediction_mape": _mean(
                np.abs(prediction_error) / np.maximum(actual_tokens, 1.0)
            ),
            "token_reservation_bias": _mean(reservation_error),
            "token_reservation_mae": _mean(np.abs(reservation_error)),
            "token_reservation_coverage_rate": _mean(
                (actual_tokens <= reserved_tokens + 1e-9).astype(np.float64)
            ),
            "token_underreservation_rate": _mean(
                (actual_tokens > reserved_tokens + 1e-9).astype(np.float64)
            ),
            "actual_token_demand_total": float(np.sum(actual_tokens)),
            "actual_token_demand_mean": _mean(actual_tokens),
        }
    )
    if "routing_latency" in completed:
        result["routing_latency_mean"] = _mean(
            completed["routing_latency"].to_numpy(dtype=np.float64)
        )

    route_share = completed["model_id"].value_counts(normalize=True, sort=False)
    route_models = sorted(endpoints) if endpoints else sorted(route_share.index)
    for model_id in route_models:
        result[f"route_share__{model_id}"] = float(route_share.get(model_id, 0.0))
    result["routing_hhi"] = float(
        sum(float(route_share.get(model_id, 0.0)) ** 2 for model_id in route_models)
    )

    if endpoints:
        horizon_start = float(completed["arrival_time"].min())
        # Every policy in a paired run sees the same arrival trace.  Restrict
        # utilization to that common, exogenous observation window; requests
        # dispatched during the subsequent drain do not enlarge the denominator.
        horizon_end = float(completed["arrival_time"].max())
        horizon = max(0.0, horizon_end - horizon_start)
        rpm_utilizations: list[float] = []
        tpm_utilizations: list[float] = []
        actual_tpm_utilizations: list[float] = []
        for model_id, endpoint in sorted(endpoints.items()):
            rpm = _endpoint_value(endpoint, "rpm", 0.0)
            rpm_capacity = _endpoint_first(
                endpoint, ("rpm_bucket_capacity", "bucket_capacity"), 0.0
            )
            available_tokens = rpm_capacity + rpm / 60.0 * horizon
            selected = completed[completed["model_id"] == model_id]
            selected_in_window = selected[
                (selected["dispatch_time"] >= horizon_start - 1e-9)
                & (selected["dispatch_time"] <= horizon_end + 1e-9)
            ]
            dispatches = int(len(selected_in_window))
            rpm_utilization = (
                dispatches / available_tokens
                if available_tokens > 0 and np.isfinite(available_tokens)
                else float("nan")
            )
            result[f"rpm_utilization__{model_id}"] = float(rpm_utilization)
            if np.isfinite(rpm_utilization):
                rpm_utilizations.append(float(rpm_utilization))

            tpm = _endpoint_first(endpoint, ("tpm",), float("inf"))
            tpm_capacity = _endpoint_first(
                endpoint, ("tpm_bucket_capacity",), float("inf")
            )
            available_tpm = tpm_capacity + tpm / 60.0 * horizon
            reserved = float(selected_in_window["reserved_token_demand"].sum())
            actual = float(selected_in_window["actual_token_demand"].sum())
            tpm_utilization = (
                reserved / available_tpm
                if available_tpm > 0 and np.isfinite(available_tpm)
                else float("nan")
            )
            result[f"tpm_utilization__{model_id}"] = float(tpm_utilization)
            if np.isfinite(tpm_utilization):
                tpm_utilizations.append(float(tpm_utilization))
            actual_tpm_utilization = (
                actual / available_tpm
                if available_tpm > 0 and np.isfinite(available_tpm)
                else float("nan")
            )
            result[f"actual_tpm_utilization__{model_id}"] = float(
                actual_tpm_utilization
            )
            if np.isfinite(actual_tpm_utilization):
                actual_tpm_utilizations.append(float(actual_tpm_utilization))
        result["rpm_utilization_mean"] = _mean(np.asarray(rpm_utilizations))
        result["tpm_utilization_mean"] = _mean(np.asarray(tpm_utilizations))
        result["actual_tpm_utilization_mean"] = _mean(
            np.asarray(actual_tpm_utilizations)
        )
    return result


def paired_confidence_interval(
    runs: pd.DataFrame,
    *,
    metric: str,
    treatment: str,
    baseline: str,
    confidence: float = 0.95,
    seed_columns: Sequence[str] = ("seed",),
) -> dict[str, float | int | str]:
    """Paired Student-t interval using common-random-number run keys."""

    required = {"policy", metric, *seed_columns}
    missing = sorted(required.difference(runs.columns))
    if missing:
        raise ValueError(f"run table is missing columns: {missing}")
    subset = runs[runs["policy"].isin([treatment, baseline])]
    present = set(subset["policy"])
    if treatment not in present or baseline not in present:
        raise ValueError("both treatment and baseline must be present")
    wide = subset.pivot_table(index=list(seed_columns), columns="policy", values=metric)
    # All-NaN metrics (for example a disabled resource's utilization) disappear
    # from pivot_table columns; retain them as zero-pair summaries.
    wide = wide.reindex(columns=[treatment, baseline])
    difference = (wide[treatment] - wide[baseline]).dropna().to_numpy(dtype=np.float64)
    count = int(difference.size)
    mean = _mean(difference)
    if count < 2:
        half_width = float("nan")
    else:
        standard_error = float(stats.sem(difference))
        half_width = float(
            stats.t.ppf((1.0 + confidence) / 2.0, df=count - 1) * standard_error
        )
    return {
        "metric": metric,
        "treatment": treatment,
        "baseline": baseline,
        "pairs": count,
        "mean_difference": mean,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
    }


def aggregate_runs(
    runs: pd.DataFrame,
    *,
    metrics: Sequence[str] = (
        "quality_mean",
        "monetary_cost_total",
        "monetary_cost_mean",
        "quota_wait_mean",
        "quota_wait_p95",
        "quota_wait_p99",
        "rpm_wait_mean",
        "rpm_wait_p95",
        "rpm_wait_p99",
        "tpm_wait_mean",
        "tpm_wait_p95",
        "tpm_wait_p99",
        "rpm_binding_rate",
        "tpm_binding_rate",
        "api_response_mean",
        "api_response_p95",
        "e2e_mean",
        "e2e_p95",
        "e2e_p99",
        "slo_violation_rate",
        "cvar95_e2e",
        "risk_brier",
        "risk_ece",
        "routing_hhi",
        "routing_latency_mean",
        "rpm_utilization_mean",
        "tpm_utilization_mean",
        "actual_tpm_utilization_mean",
        "token_prediction_bias",
        "token_prediction_mae",
        "token_prediction_rmse",
        "token_prediction_mape",
        "token_reservation_bias",
        "token_reservation_mae",
        "token_reservation_coverage_rate",
        "token_underreservation_rate",
        "actual_token_demand_total",
        "actual_token_demand_mean",
    ),
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Per-condition/per-policy means and run-level confidence intervals."""

    metric_names = list(metrics)
    metric_names.extend(
        sorted(
            column
            for column in runs.columns
            if column.startswith(
                (
                    "route_share__",
                    "rpm_utilization__",
                    "tpm_utilization__",
                    "actual_tpm_utilization__",
                )
            )
            and column not in metric_names
        )
    )
    records: list[dict[str, float | int | str]] = []
    group_columns = ["policy"]
    if "condition" in runs.columns:
        group_columns.insert(0, "condition")
    for keys, group in runs.groupby(group_columns, sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        record: dict[str, float | int | str] = {
            column: str(value)
            for column, value in zip(group_columns, key_tuple, strict=True)
        }
        record["runs"] = len(group)
        for metric in metric_names:
            if metric not in group.columns:
                continue
            values = group[metric].dropna().to_numpy(dtype=np.float64)
            mean = _mean(values)
            record[f"{metric}__n"] = int(values.size)
            if values.size < 2:
                half_width = float("nan")
            else:
                half_width = float(
                    stats.t.ppf((1 + confidence) / 2, df=len(values) - 1)
                    * stats.sem(values)
                )
            record[f"{metric}__mean"] = mean
            record[f"{metric}__ci_low"] = mean - half_width
            record[f"{metric}__ci_high"] = mean + half_width
        records.append(record)
    return pd.DataFrame.from_records(records)


def json_ready(value: Any) -> Any:
    """Recursively convert numpy/dataclass values before ``json.dump``."""

    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
