"""Command-line runner for feedback-free PORT-AQS Stage 1 experiments."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .data import (
    KNNPrediction,
    SplitSpec,
    TfidfKNNProfiles,
    calibration_cost_scale,
    load_routerbench,
    stratified_split,
)
from .metrics import aggregate_runs, compute_metrics, json_ready, paired_confidence_interval
from .simulator import (
    ALL_POLICIES,
    POLICY_ALIASES,
    QuotaProxyPrices,
    SimulationConfig,
    calibration_proxy_duals,
    simulate_policy,
)
from .synthetic import (
    LatencyNoiseConfig,
    endpoint_rpms,
    generate_burst_arrivals,
    generate_poisson_arrivals,
    make_speed_profiles,
    materialize_latency_table,
)
from .types import EndpointSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DemandStatistics:
    mean_by_arm: Mapping[str, float]
    p99_by_arm: Mapping[str, float]
    max_by_arm: Mapping[str, float]


@dataclass(frozen=True)
class CalibrationResult:
    beta: float
    lambda_penalty: float
    cost_budget: float
    quota_prices: QuotaProxyPrices
    trials: pd.DataFrame


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def _resolve_from_project(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    smoke = result.get("smoke", {})
    result.setdefault("data", {}).setdefault("splits", {}).update(
        smoke.get(
            "splits",
            {"historical": 400, "calibration": 100, "streaming": 120},
        )
    )
    result.setdefault("profiles", {}).update(
        smoke.get("profiles", {"neighbors": 8, "max_features": 512})
    )
    result.setdefault("experiment", {})["seeds"] = smoke.get("seeds", [0])
    result["experiment"]["conditions"] = smoke.get(
        "conditions",
        [
            {
                "name": "smoke_joint",
                "arrival": "poisson",
                "quota_mode": "joint",
                "quota_load": 0.9,
            },
            {
                "name": "smoke_rpm_only",
                "arrival": "poisson",
                "quota_mode": "rpm_only",
                "quota_load": 0.9,
            },
            {
                "name": "smoke_tpm_only",
                "arrival": "poisson",
                "quota_mode": "tpm_only",
                "quota_load": 0.9,
            },
        ],
    )
    result["experiment"]["robustness"] = {"enabled": False}
    result["experiment"]["save_traces"] = True
    if "calibration" in result:
        result["calibration"]["cost_budget_normalized_per_task"] = smoke.get(
            "cost_budget_normalized_per_task", 0.15
        )
        result["calibration"]["lambda_grid"] = smoke.get("lambda_grid", [0.0, 0.5])
    result["experiment"]["output_dir"] = smoke.get("output_dir", "results/smoke")
    return result


def _select_conditions(
    config: dict[str, Any], condition_names: Sequence[str]
) -> dict[str, Any]:
    """Return a config containing only the requested expanded conditions."""

    requested = list(dict.fromkeys(map(str, condition_names)))
    if not requested:
        raise ValueError("at least one condition name is required")
    result = deepcopy(config)
    experiment = result.setdefault("experiment", {})
    expanded = _expanded_conditions(experiment)
    by_name = {str(condition["name"]): condition for condition in expanded}
    unknown = sorted(set(requested).difference(by_name))
    if unknown:
        raise ValueError(
            f"unknown conditions: {unknown}; choices are {sorted(by_name)}"
        )
    experiment["conditions"] = [by_name[name] for name in requested]
    # Robustness conditions have already been expanded above. Disable a second
    # expansion when run_experiment later resolves the selected configuration.
    experiment["robustness"] = {"enabled": False}
    return result


def _noise_config(config: Mapping[str, Any]) -> LatencyNoiseConfig:
    return LatencyNoiseConfig(
        lognormal_sigma=float(config.get("lognormal_sigma", 0.22)),
        tail_probability=float(config.get("tail_probability", 0.01)),
        tail_multiplier=float(config.get("tail_multiplier", 4.0)),
        tail_pareto_shape=float(config.get("tail_pareto_shape", 3.0)),
        maximum_multiplier=float(config.get("maximum_multiplier", 25.0)),
    )


def _expanded_conditions(experiment: Mapping[str, Any]) -> list[dict[str, Any]]:
    conditions = [dict(item) for item in experiment.get("conditions", [])]
    robustness = experiment.get("robustness", {})
    if robustness.get("enabled", False):
        for permutation in robustness.get("quota_permutations", [0, 1, 2, 3, 4]):
            conditions.append(
                {
                    "name": f"heterogeneous_joint_p{int(permutation)}",
                    "arrival": robustness.get("arrival", "poisson"),
                    "quota_mode": "joint",
                    "quota_load": float(robustness.get("quota_load", 0.7)),
                    "heterogeneous": True,
                    "quota_permutation": int(permutation),
                }
            )
    names = [condition.get("name") for condition in conditions]
    if not conditions or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("experiment conditions need unique non-empty names")
    allowed_modes = {"joint", "rpm_only", "tpm_only"}
    for condition in conditions:
        mode = str(condition.get("quota_mode", "joint"))
        if mode not in allowed_modes:
            raise ValueError(f"unknown quota_mode {mode!r}; expected {sorted(allowed_modes)}")
        condition["quota_mode"] = mode
    return conditions


def _predicted_token_demand(frame: pd.DataFrame, profiles: KNNPrediction) -> np.ndarray:
    inputs = frame["input_tokens"].to_numpy(dtype=np.float64)[:, None]
    demand = inputs + np.maximum(profiles.output_tokens, 0.0)
    if np.any(~np.isfinite(demand)) or np.any(demand <= 0):
        raise ValueError("predicted token demand must be positive and finite")
    return demand


def _demand_statistics(
    frame: pd.DataFrame, profiles: KNNPrediction, arms: Sequence[str]
) -> DemandStatistics:
    demand = _predicted_token_demand(frame, profiles)
    return DemandStatistics(
        mean_by_arm={str(arm): float(demand[:, index].mean()) for index, arm in enumerate(arms)},
        p99_by_arm={str(arm): float(np.quantile(demand[:, index], 0.99)) for index, arm in enumerate(arms)},
        max_by_arm={str(arm): float(demand[:, index].max()) for index, arm in enumerate(arms)},
    )


def _make_endpoints(
    arms: Sequence[str],
    quota_config: Mapping[str, Any],
    condition: Mapping[str, Any],
    demand_statistics: DemandStatistics,
) -> tuple[EndpointSpec, ...]:
    mode = str(condition.get("quota_mode", "joint"))
    heterogeneous = bool(condition.get("heterogeneous", False))
    rpms = endpoint_rpms(
        arms,
        homogeneous_rpm=float(quota_config.get("homogeneous_rpm", 60.0)),
        heterogeneous=heterogeneous,
        values=tuple(
            float(value)
            for value in quota_config.get(
                "heterogeneous_rpm_values", [30, 60, 120, 240]
            )
        ),
        permutation_index=int(condition.get("quota_permutation", 0)),
    )
    rpm_bucket = float(quota_config.get("rpm_bucket_capacity", 6.0))
    request_equivalents = float(quota_config.get("tpm_bucket_request_equivalents", 6.0))
    demand_quantile_floor = float(quota_config.get("tpm_demand_quantile", 0.99))
    if not math.isclose(demand_quantile_floor, 0.99, abs_tol=1e-12):
        raise ValueError("the current frozen DemandStatistics stores the required p99 only")
    max_headroom = float(quota_config.get("tpm_calibration_max_headroom", 1.10))
    if max_headroom < 1.0 or not math.isfinite(max_headroom):
        raise ValueError("tpm_calibration_max_headroom must be finite and at least 1")
    tpm_rate_scale = float(quota_config.get("tpm_rate_scale", 1.0))
    if tpm_rate_scale <= 0 or not math.isfinite(tpm_rate_scale):
        raise ValueError("tpm_rate_scale must be positive and finite")

    endpoints: list[EndpointSpec] = []
    for arm in arms:
        model_id = str(arm)
        nominal_rpm = float(rpms[model_id])
        mean_demand = demand_statistics.mean_by_arm[model_id]
        tpm = nominal_rpm * mean_demand * tpm_rate_scale
        tpm_bucket = max(
            request_equivalents * mean_demand,
            demand_statistics.p99_by_arm[model_id],
            max_headroom * demand_statistics.max_by_arm[model_id],
        )
        endpoints.append(
            EndpointSpec(
                model_id=model_id,
                rpm=(float("inf") if mode == "tpm_only" else nominal_rpm),
                rpm_bucket_capacity=(float("inf") if mode == "tpm_only" else rpm_bucket),
                tpm=(float("inf") if mode == "rpm_only" else tpm),
                tpm_bucket_capacity=(float("inf") if mode == "rpm_only" else tpm_bucket),
            )
        )
    return tuple(endpoints)


def _aggregate_request_capacity(
    endpoints: Sequence[EndpointSpec], demand_statistics: DemandStatistics
) -> float:
    total = 0.0
    for endpoint in endpoints:
        rpm_equivalent = endpoint.rpm_refill_rate
        tpm_equivalent = (
            endpoint.tpm_refill_rate
            / demand_statistics.mean_by_arm[endpoint.model_id]
        )
        endpoint_rate = min(rpm_equivalent, tpm_equivalent)
        if not math.isfinite(endpoint_rate) and endpoint_rate > 0:
            raise ValueError("at least one finite quota resource is required per endpoint")
        total += endpoint_rate
    if total <= 0 or not math.isfinite(total):
        raise ValueError("aggregate request-equivalent quota capacity must be finite/positive")
    return total


def _arrivals(
    count: int,
    endpoints: Sequence[EndpointSpec],
    demand_statistics: DemandStatistics,
    condition: Mapping[str, Any],
    *,
    seed: int,
) -> np.ndarray:
    aggregate = _aggregate_request_capacity(endpoints, demand_statistics)
    mode = str(condition.get("arrival", "poisson"))
    if mode == "poisson":
        return generate_poisson_arrivals(
            count,
            aggregate_refill_rate=aggregate,
            quota_load=float(condition.get("quota_load", 0.7)),
            seed=seed,
        )
    if mode == "burst":
        return generate_burst_arrivals(
            count,
            aggregate_refill_rate=aggregate,
            loads=tuple(
                float(value)
                for value in condition.get("burst_loads", [0.5, 1.2, 0.5])
            ),
            task_fractions=tuple(
                float(value)
                for value in condition.get("burst_fractions", [0.25, 0.5, 0.25])
            ),
            seed=seed,
        )
    raise ValueError(f"unknown arrival mode {mode!r}")


def _expected_calibration_horizon(
    count: int,
    endpoints: Sequence[EndpointSpec],
    demand_statistics: DemandStatistics,
    condition: Mapping[str, Any],
) -> float:
    if count <= 1:
        return 0.0
    aggregate = _aggregate_request_capacity(endpoints, demand_statistics)
    if condition.get("arrival", "poisson") == "burst":
        loads = np.asarray(condition.get("burst_loads", [0.5, 1.2, 0.5]), dtype=float)
        fractions = np.asarray(
            condition.get("burst_fractions", [0.25, 0.5, 0.25]), dtype=float
        )
        fractions /= fractions.sum()
        return float(np.sum((count * fractions) / (aggregate * loads)))
    return float(
        (count - 1)
        / (aggregate * float(condition.get("quota_load", 0.7)))
    )


def _simulation_config(
    routing: Mapping[str, Any], *, beta: float, lambda_penalty: float
) -> SimulationConfig:
    rpm_initial = routing.get("initial_rpm_tokens")
    tpm_initial = routing.get("initial_tpm_tokens")
    return SimulationConfig(
        deadline_seconds=float(routing.get("deadline_seconds", 8.0)),
        beta=float(beta),
        lambda_penalty=float(lambda_penalty),
        initial_rpm_tokens=None if rpm_initial is None else float(rpm_initial),
        initial_tpm_tokens=None if tpm_initial is None else float(tpm_initial),
    )


def _reorder_profiles(profiles: KNNPrediction, order: np.ndarray) -> KNNPrediction:
    return KNNPrediction(
        quality=profiles.quality[order],
        cost=profiles.cost[order],
        output_tokens=profiles.output_tokens[order],
        latency_samples=profiles.latency_samples[order],
        neighbor_weights=profiles.neighbor_weights[order],
    )


def _zero_prices(arms: Sequence[str]) -> QuotaProxyPrices:
    return QuotaProxyPrices(
        {str(arm): 0.0 for arm in arms},
        {str(arm): 0.0 for arm in arms},
    )


def _preflight_demands(
    frame: pd.DataFrame,
    profiles: KNNPrediction,
    endpoints: Sequence[EndpointSpec],
    *,
    label: str,
) -> None:
    """Fail explicitly if a frozen prediction cannot fit a finite TPM bucket."""

    demand = _predicted_token_demand(frame, profiles)
    failures: list[str] = []
    for arm_index, endpoint in enumerate(endpoints):
        if math.isfinite(endpoint.tpm_bucket_capacity):
            maximum = float(demand[:, arm_index].max(initial=0.0))
            if maximum > endpoint.tpm_bucket_capacity + 1e-9:
                failures.append(
                    f"{endpoint.model_id}: predicted max {maximum:.3f} > "
                    f"TPM bucket {endpoint.tpm_bucket_capacity:.3f}"
                )
    if failures:
        raise ValueError(
            f"{label} has unreservable predicted token demand; actual token outcomes "
            f"were not consulted: {'; '.join(failures)}"
        )


def _calibrate_hyperparameters(
    *,
    calibration_config: Mapping[str, Any],
    routing_config: Mapping[str, Any],
    quota_config: Mapping[str, Any],
    calibration_frame: pd.DataFrame,
    arms: Sequence[str],
    actual_quality: np.ndarray,
    actual_cost: np.ndarray,
    actual_output_tokens: np.ndarray,
    profiles: KNNPrediction,
    speeds: Mapping[str, Any],
    noise: LatencyNoiseConfig,
    cost_scale: float,
    demand_statistics: DemandStatistics,
    condition: Mapping[str, Any] | None = None,
) -> CalibrationResult:
    """Calibrate beta/gamma/lambda for one fixed deployment condition."""

    if not calibration_config.get("enabled", True):
        return CalibrationResult(
            beta=float(routing_config.get("beta", 0.1)),
            lambda_penalty=float(routing_config.get("lambda_penalty", 0.5)),
            cost_budget=float("nan"),
            quota_prices=_zero_prices(arms),
            trials=pd.DataFrame(),
        )
    if condition is None:
        calibration_condition: dict[str, Any] = {
            "name": "calibration",
            "arrival": "poisson",
            "quota_mode": str(calibration_config.get("quota_mode", "joint")),
            "quota_load": float(calibration_config.get("quota_load", 0.7)),
        }
    else:
        calibration_condition = dict(condition)
        calibration_condition["name"] = f"calibration__{condition.get('name', 'condition')}"
    endpoints = _make_endpoints(
        arms, quota_config, calibration_condition, demand_statistics
    )
    _preflight_demands(
        calibration_frame,
        profiles,
        endpoints,
        label=str(calibration_condition["name"]),
    )
    seed = int(calibration_config.get("seed", 17))
    arrivals = _arrivals(
        len(calibration_frame),
        endpoints,
        demand_statistics,
        calibration_condition,
        seed=seed,
    )
    world = materialize_latency_table(
        calibration_frame,
        arms,
        actual_output_tokens,
        speeds,
        seed=seed,
        noise=noise,
    )
    horizon = _expected_calibration_horizon(
        len(calibration_frame), endpoints, demand_statistics, calibration_condition
    )
    demand = _predicted_token_demand(calibration_frame, profiles)
    normalised_cost = profiles.cost / cost_scale
    if "cost_budget_normalized_total" in calibration_config:
        cost_budget = float(calibration_config["cost_budget_normalized_total"])
    else:
        cost_budget = len(calibration_frame) * float(
            calibration_config.get("cost_budget_normalized_per_task", 0.15)
        )
    cheapest_feasible = float(np.min(normalised_cost, axis=1).sum())
    if cost_budget < cheapest_feasible - 1e-9:
        raise ValueError(
            "calibration cost budget is infeasible: "
            f"B^C={cost_budget:.6g} < cheapest predicted total {cheapest_feasible:.6g}"
        )
    if routing_config.get("gamma_proxy", True):
        duals = calibration_proxy_duals(
            profiles.quality,
            normalised_cost,
            demand,
            endpoints,
            horizon_seconds=horizon,
            cost_budget=cost_budget,
        )
        calibrated_beta = float(duals.beta)
        calibrated_prices = duals.quota_prices
    else:
        calibrated_beta = float(routing_config.get("beta", 0.1))
        calibrated_prices = _zero_prices(arms)
    lambda_grid = [
        float(value)
        for value in calibration_config.get(
            "lambda_grid",
            [float(routing_config.get("lambda_penalty", 1.0))],
        )
    ]
    if (
        not lambda_grid
        or any(not math.isfinite(value) or value < 0 for value in lambda_grid)
    ):
        raise ValueError("lambda_grid must contain finite non-negative values")

    records: list[dict[str, Any]] = []
    endpoint_map = {endpoint.model_id: endpoint for endpoint in endpoints}
    for lambda_penalty in lambda_grid:
        trace = simulate_policy(
            policy="rpm_aware",
            frame=calibration_frame,
            arms=arms,
            arrivals=arrivals,
            actual_quality=actual_quality,
            actual_cost=actual_cost,
            actual_output_tokens=actual_output_tokens,
            profiles=profiles,
            world=world,
            endpoints=endpoints,
            cost_scale=cost_scale,
            quota_prices=calibrated_prices,
            config=_simulation_config(
                routing_config,
                beta=calibrated_beta,
                lambda_penalty=lambda_penalty,
            ),
            seed=seed,
        )
        metrics = compute_metrics(trace, endpoints=endpoint_map)
        records.append(
            {
                "calibration_method": "lp_beta_gamma_lambda_replay",
                "cost_budget_normalized_total": cost_budget,
                "cost_budget_normalized_per_task": cost_budget / len(calibration_frame),
                "cheapest_predicted_cost_total": cheapest_feasible,
                "beta": calibrated_beta,
                "lambda_penalty": lambda_penalty,
                **metrics,
                "normalized_cost_mean": float(metrics["monetary_cost_mean"])
                / cost_scale,
            }
        )

    trials = pd.DataFrame.from_records(records)
    selection_cost_weight = float(calibration_config.get("selection_cost_weight", 0.1))
    selection_slo_weight = float(calibration_config.get("selection_slo_weight", 1.0))
    if (
        not np.isfinite(selection_cost_weight)
        or not np.isfinite(selection_slo_weight)
        or selection_cost_weight < 0
        or selection_slo_weight < 0
    ):
        raise ValueError("calibration selection weights must be finite and non-negative")
    trials["selection_utility"] = (
        trials["quality_mean"]
        - selection_cost_weight * trials["normalized_cost_mean"]
        - selection_slo_weight * trials["slo_violation_rate"]
    )
    ranked = trials.sort_values(
        [
            "selection_utility",
            "quality_mean",
            "monetary_cost_mean",
            "slo_violation_rate",
            "lambda_penalty",
        ],
        ascending=[False, False, True, True, True],
    )
    selected = ranked.iloc[0]
    trials["selected"] = trials["lambda_penalty"] == selected["lambda_penalty"]
    return CalibrationResult(
        beta=calibrated_beta,
        lambda_penalty=float(selected["lambda_penalty"]),
        cost_budget=float(cost_budget),
        quota_prices=calibrated_prices,
        trials=trials,
    )


def run_experiment(config: Mapping[str, Any]) -> Path:
    data_config = config.get("data", {})
    split_config = data_config.get("splits", {})
    dataset = load_routerbench(
        _resolve_from_project(
            data_config.get("path", "../../PORT/data/routerbench_0shot.pkl")
        )
    )
    splits = stratified_split(
        dataset.frame,
        SplitSpec(
            historical=int(split_config.get("historical", 26_481)),
            calibration=int(split_config.get("calibration", 500)),
            streaming=int(split_config.get("streaming", 9_500)),
            seed=int(split_config.get("seed", 2025)),
        ),
    )
    output_dir = _resolve_from_project(
        config.get("experiment", {}).get("output_dir", "results/mvp")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    synthetic_config = config.get("synthetic", {})
    speeds = make_speed_profiles(
        dataset.arms,
        permutation_index=int(synthetic_config.get("speed_permutation", 0)),
        intercept_range=tuple(
            float(value)
            for value in synthetic_config.get("intercept_range", [0.20, 0.85])
        ),
        tokens_per_second_range=tuple(
            float(value)
            for value in synthetic_config.get(
                "tokens_per_second_range", [16.0, 64.0]
            )
        ),
    )
    noise = _noise_config(synthetic_config.get("noise", {}))
    historical_latency = materialize_latency_table(
        splits.historical,
        dataset.arms,
        dataset.output_token_matrix(splits.historical),
        speeds,
        seed=int(synthetic_config.get("historical_seed", 101)),
        noise=noise,
    ).static_response_seconds
    profile_config = config.get("profiles", {})
    profile_index = TfidfKNNProfiles(
        neighbors=int(profile_config.get("neighbors", 16)),
        max_features=int(profile_config.get("max_features", 4_096)),
        query_batch_size=int(profile_config.get("query_batch_size", 256)),
        n_jobs=int(profile_config.get("n_jobs", 1)),
    ).fit(
        splits.historical["prompt"],
        dataset.quality_matrix(splits.historical),
        dataset.cost_matrix(splits.historical),
        historical_latency,
        dataset.output_token_matrix(splits.historical),
    )
    calibration_profiles = profile_index.predict(splits.calibration["prompt"])
    streaming_profiles = profile_index.predict(splits.streaming["prompt"])
    calibration_quality = dataset.quality_matrix(splits.calibration)
    calibration_cost = dataset.cost_matrix(splits.calibration)
    calibration_output = dataset.output_token_matrix(splits.calibration)
    cost_scale = calibration_cost_scale(calibration_cost)
    demand_statistics = _demand_statistics(
        splits.calibration, calibration_profiles, dataset.arms
    )

    routing_config = config.get("routing", {})
    quota_config = config.get("quota", {})
    calibration_config = config.get("calibration", {})
    if calibration_config.get("enabled", True):
        if "cost_budget_normalized_total" in calibration_config:
            configured_cost_budget = float(
                calibration_config["cost_budget_normalized_total"]
            )
        else:
            configured_cost_budget = len(splits.calibration) * float(
                calibration_config.get("cost_budget_normalized_per_task", 0.15)
            )
    else:
        configured_cost_budget = float("nan")

    metadata = {
        "dataset_path": str(dataset.source_path),
        "source_rows": dataset.source_rows,
        "deduplicated_rows": splits.deduplicated_size,
        "duplicate_prompts_removed": dataset.duplicate_prompts_removed,
        "requested_split_sizes": asdict(splits.requested),
        "actual_split_sizes": splits.actual_sizes,
        "arms": list(dataset.arms),
        "source_model_names_persisted": False,
        "speed_profiles": {arm: asdict(speed) for arm, speed in speeds.items()},
        "cost_scale": cost_scale,
        "calibration_cost_budget_normalized_total": configured_cost_budget,
        "calibration_cost_budget_normalized_per_task": (
            None
            if not math.isfinite(configured_cost_budget)
            else configured_cost_budget / max(1, len(splits.calibration))
        ),
        "calibration_parameter_scope": "condition_specific_fixed_within_quota_load",
        "frozen_beta": None,
        "frozen_lambda_penalty": None,
        "calibration_demand_statistics": asdict(demand_statistics),
        "tpm_capacity_rule": (
            "max(bucket_request_equivalents*calibration_arm_mean, "
            "calibration_arm_p99, calibration_arm_max*headroom)"
        ),
        "streaming_outcomes_used_for_quota_configuration": False,
        "streaming_completion_feedback": False,
        "quality_feedback": False,
        "latency_health_feedback": False,
        "streaming_order": "deterministic common permutation per seed",
        "synthetic_routing_latency_seconds": 0.0,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(json_ready(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = pd.concat(
        [
            splits.historical[["task_id", "prompt_hash", "eval_name"]].assign(split="historical"),
            splits.calibration[["task_id", "prompt_hash", "eval_name"]].assign(split="calibration"),
            splits.streaming[["task_id", "prompt_hash", "eval_name"]].assign(split="streaming"),
        ],
        ignore_index=True,
    )
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)

    experiment_config = config.get("experiment", {})
    conditions = _expanded_conditions(experiment_config)
    raw_policies = [
        str(policy) for policy in experiment_config.get("policies", ALL_POLICIES)
    ]
    unknown = sorted(
        set(raw_policies).difference(set(ALL_POLICIES).union(POLICY_ALIASES))
    )
    if unknown:
        raise ValueError(f"unknown policies: {unknown}; choices are {ALL_POLICIES}")
    policies = list(
        dict.fromkeys(POLICY_ALIASES.get(policy, policy) for policy in raw_policies)
    )
    seeds = [int(seed) for seed in experiment_config.get("seeds", range(10))]
    save_traces = bool(experiment_config.get("save_traces", True))
    trace_dir = output_dir / "traces"
    if save_traces:
        trace_dir.mkdir(exist_ok=True)

    streaming_quality = dataset.quality_matrix(splits.streaming)
    streaming_cost = dataset.cost_matrix(splits.streaming)
    streaming_output = dataset.output_token_matrix(splits.streaming)
    metrics_records: list[dict[str, Any]] = []
    condition_parameters: list[dict[str, Any]] = []
    calibration_trial_frames: list[pd.DataFrame] = []
    prepared_conditions: list[
        tuple[int, dict[str, Any], tuple[EndpointSpec, ...], CalibrationResult]
    ] = []
    for condition_index, condition in enumerate(conditions):
        endpoints = _make_endpoints(
            dataset.arms, quota_config, condition, demand_statistics
        )
        _preflight_demands(
            splits.streaming,
            streaming_profiles,
            endpoints,
            label=f"streaming condition {condition['name']}",
        )
        calibration = _calibrate_hyperparameters(
            calibration_config=calibration_config,
            routing_config=routing_config,
            quota_config=quota_config,
            calibration_frame=splits.calibration,
            arms=dataset.arms,
            actual_quality=calibration_quality,
            actual_cost=calibration_cost,
            actual_output_tokens=calibration_output,
            profiles=calibration_profiles,
            speeds=speeds,
            noise=noise,
            cost_scale=cost_scale,
            demand_statistics=demand_statistics,
            condition=condition,
        )
        if not calibration.trials.empty:
            trials = calibration.trials.copy()
            trials.insert(0, "condition_index", condition_index)
            trials.insert(0, "condition", str(condition["name"]))
            calibration_trial_frames.append(trials)
        horizon = _expected_calibration_horizon(
            len(splits.calibration), endpoints, demand_statistics, condition
        )
        condition_parameters.append(
            {
                "condition": str(condition["name"]),
                "condition_index": condition_index,
                "configuration": dict(condition),
                "cost_budget_normalized_total": calibration.cost_budget,
                "cost_budget_normalized_per_task": (
                    None
                    if not math.isfinite(calibration.cost_budget)
                    else calibration.cost_budget / max(1, len(splits.calibration))
                ),
                "frozen_beta": calibration.beta,
                "frozen_lambda_penalty": calibration.lambda_penalty,
                "request_equivalent_arrival_capacity_per_second": (
                    _aggregate_request_capacity(endpoints, demand_statistics)
                ),
                "calibration_proxy_horizon_seconds": horizon,
                "endpoints": [asdict(endpoint) for endpoint in endpoints],
                "frozen_gamma_rpm": dict(calibration.quota_prices.gamma_rpm),
                "frozen_gamma_tpm": dict(calibration.quota_prices.gamma_tpm),
            }
        )
        prepared_conditions.append(
            (condition_index, dict(condition), endpoints, calibration)
        )

    if calibration_trial_frames:
        pd.concat(calibration_trial_frames, ignore_index=True).to_csv(
            output_dir / "calibration_trials.csv", index=False
        )

    for condition_index, condition, endpoints, calibration in prepared_conditions:
        for seed in seeds:
            common_seed = int(experiment_config.get("seed_offset", 10_000)) + seed
            task_order = np.random.default_rng(common_seed + 1_000_003).permutation(
                len(splits.streaming)
            )
            seed_frame = splits.streaming.iloc[task_order].reset_index(drop=True)
            seed_quality = streaming_quality[task_order]
            seed_cost = streaming_cost[task_order]
            seed_output = streaming_output[task_order]
            seed_profiles = _reorder_profiles(streaming_profiles, task_order)
            arrivals = _arrivals(
                len(splits.streaming),
                endpoints,
                demand_statistics,
                condition,
                seed=common_seed,
            )
            world = materialize_latency_table(
                seed_frame,
                dataset.arms,
                seed_output,
                speeds,
                seed=common_seed,
                noise=noise,
            )
            sim_config = _simulation_config(
                routing_config,
                beta=calibration.beta,
                lambda_penalty=calibration.lambda_penalty,
            )
            endpoint_map = {endpoint.model_id: endpoint for endpoint in endpoints}
            for policy in policies:
                trace = simulate_policy(
                    policy=policy,
                    frame=seed_frame,
                    arms=dataset.arms,
                    arrivals=arrivals,
                    actual_quality=seed_quality,
                    actual_cost=seed_cost,
                    actual_output_tokens=seed_output,
                    profiles=seed_profiles,
                    world=world,
                    endpoints=endpoints,
                    cost_scale=cost_scale,
                    quota_prices=calibration.quota_prices,
                    config=sim_config,
                    seed=common_seed,
                )
                metrics = compute_metrics(trace, endpoints=endpoint_map)
                metrics_records.append(
                    {
                        "condition": str(condition["name"]),
                        "condition_index": condition_index,
                        "seed": seed,
                        "policy": policy,
                        "arrival": condition.get("arrival", "poisson"),
                        "quota_mode": condition.get("quota_mode", "joint"),
                        "quota_load": condition.get("quota_load", np.nan),
                        "heterogeneous": bool(condition.get("heterogeneous", False)),
                        "quota_permutation": int(condition.get("quota_permutation", 0)),
                        "frozen_beta": calibration.beta,
                        "frozen_lambda_penalty": calibration.lambda_penalty,
                        **metrics,
                    }
                )
                if save_traces:
                    trace.to_pickle(
                        trace_dir / f"{condition['name']}__seed{seed}__{policy}.pkl.gz",
                        compression="gzip",
                    )
            pd.DataFrame.from_records(metrics_records).to_csv(
                output_dir / "run_metrics.csv", index=False
            )

    runs = pd.DataFrame.from_records(metrics_records)
    (output_dir / "condition_parameters.json").write_text(
        json.dumps(json_ready(condition_parameters), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    aggregate_runs(runs).to_csv(output_dir / "aggregate_metrics.csv", index=False)
    intervals: list[dict[str, Any]] = []
    paired_metrics = [
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
    ]
    paired_metrics.extend(
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
        )
    )
    for condition_name, condition_runs in runs.groupby("condition", sort=True):
        if "rpm_aware" not in set(condition_runs["policy"]):
            continue
        for baseline in sorted(
            set(condition_runs["policy"]).difference({"rpm_aware"})
        ):
            for metric in paired_metrics:
                interval = paired_confidence_interval(
                    condition_runs,
                    metric=metric,
                    treatment="rpm_aware",
                    baseline=baseline,
                )
                intervals.append({"condition": condition_name, **interval})
    pd.DataFrame.from_records(intervals).to_csv(
        output_dir / "paired_confidence_intervals.csv", index=False
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run feedback-free PORT-inspired API RPM/TPM quota routing."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mvp.yaml",
        help="YAML configuration (default: configs/mvp.yaml)",
    )
    parser.add_argument("--smoke", action="store_true", help="run a fast end-to-end check")
    parser.add_argument("--data", type=Path, help="override the RouterBench pickle path")
    parser.add_argument("--output", type=Path, help="override the result directory")
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=ALL_POLICIES + tuple(POLICY_ALIASES),
        help="override policies (old names such as quota_risk/static_risk/min_risk are aliases)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, help="override common random seeds")
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="run only named conditions from the resolved experiment matrix",
    )
    parser.add_argument("--no-traces", action="store_true", help="omit request traces")
    parser.add_argument("--list-policies", action="store_true", help="print policy names and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_policies:
        print("\n".join(ALL_POLICIES))
        return 0
    config = _load_config(args.config.resolve())
    if args.smoke:
        config = _smoke_config(config)
    if args.conditions:
        config = _select_conditions(config, args.conditions)
    if args.data:
        config.setdefault("data", {})["path"] = str(args.data.resolve())
    if args.output:
        config.setdefault("experiment", {})["output_dir"] = str(args.output.resolve())
    if args.policies:
        config.setdefault("experiment", {})["policies"] = args.policies
    if args.seeds:
        config.setdefault("experiment", {})["seeds"] = args.seeds
    if args.no_traces:
        config.setdefault("experiment", {})["save_traces"] = False
    output_dir = run_experiment(config)
    print(f"PORT-AQS Stage 1 results: {output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
