"""Feedback-free, one-shot simulation for PORT-AQS Stage 1.

Only client-side RPM/TPM admission state changes online.  Quality, monetary
cost, output-token demand and response-latency profiles are frozen before the
stream starts.  Completions are recorded for evaluation but never update the
router or either quota bucket.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, sparse

from .data import KNNPrediction
from .latency import LatencyEstimator
from .metrics import json_ready
from .quota import QuotaLimiter
from .router import PortAQSRouter
from .synthetic import SyntheticLatencyTable
from .types import (
    CandidateEstimate,
    EndpointSpec,
    QuotaPrice,
    QuotaReservation,
    RoutingMode,
)


STANDARD_POLICIES = (
    "quality_cost",
    "static_latency",
    "rpm_aware",
    "rpm_aware_no_gamma",
    "rpm_aware_lambda_0",
    "static_latency_no_gamma",
)
REFERENCE_POLICIES = ("available", "random", "best_quality", "min_latency_risk")
ALL_POLICIES = STANDARD_POLICIES + REFERENCE_POLICIES
POLICY_ALIASES: Mapping[str, str] = {
    "admission": "rpm_aware",
    "quota_risk": "rpm_aware",
    "static_risk": "static_latency",
    "min_risk": "min_latency_risk",
}


@dataclass(frozen=True)
class SimulationConfig:
    deadline_seconds: float = 8.0
    beta: float = 0.1
    lambda_penalty: float = 0.5
    initial_rpm_tokens: float | None = None
    initial_tpm_tokens: float | None = None


@dataclass(frozen=True)
class QuotaProxyPrices:
    """Frozen per-endpoint shadow prices from the aggregate calibration LP."""

    gamma_rpm: Mapping[str, float]
    gamma_tpm: Mapping[str, float]

    def for_endpoint(self, model_id: str) -> QuotaPrice:
        return QuotaPrice(
            gamma_rpm=float(self.gamma_rpm.get(model_id, 0.0)),
            gamma_tpm=float(self.gamma_tpm.get(model_id, 0.0)),
        )

    def as_core_mapping(self, arms: Sequence[str]) -> dict[str, QuotaPrice]:
        return {str(arm): self.for_endpoint(str(arm)) for arm in arms}


@dataclass(frozen=True)
class CalibrationDualPrices:
    """Shadow prices produced by the static calibration proxy LP."""

    beta: float
    quota_prices: QuotaProxyPrices
    cost_budget: float


@dataclass(frozen=True)
class _DecisionView:
    selected_model_id: str
    selected_candidate: CandidateEstimate
    reservation: QuotaReservation
    candidates: tuple[CandidateEstimate, ...]


def calibration_proxy_duals(
    predicted_quality: np.ndarray,
    normalized_cost: np.ndarray,
    predicted_token_demand: np.ndarray,
    endpoints: Sequence[EndpointSpec],
    *,
    horizon_seconds: float,
    cost_budget: float,
) -> CalibrationDualPrices:
    """Solve the calibration proxy LP and return beta/gamma shadow prices.

    This static fractional LP is used only before streaming.  It maximises
    predicted quality subject to one assignment per task, a total normalised
    monetary-cost budget, and aggregate quota capacities.  The rolling
    token-bucket simulator remains the only strict online quota mechanism.
    """

    quality = np.asarray(predicted_quality, dtype=np.float64)
    cost = np.asarray(normalized_cost, dtype=np.float64)
    demand = np.asarray(predicted_token_demand, dtype=np.float64)
    if quality.ndim != 2 or cost.shape != quality.shape or demand.shape != quality.shape:
        raise ValueError("quality, normalized_cost and predicted_token_demand must share [task, arm] shape")
    if (
        np.any(~np.isfinite(quality))
        or np.any(~np.isfinite(cost))
        or np.any(~np.isfinite(demand))
        or np.any(cost < 0)
        or np.any(demand <= 0)
    ):
        raise ValueError("quality/cost/demand must be finite, with non-negative cost and positive demand")
    task_count, arm_count = quality.shape
    if len(endpoints) != arm_count:
        raise ValueError("calibration matrix arm count and endpoint count differ")
    if task_count == 0:
        empty = QuotaProxyPrices(
            {endpoint.model_id: 0.0 for endpoint in endpoints},
            {endpoint.model_id: 0.0 for endpoint in endpoints},
        )
        return CalibrationDualPrices(0.0, empty, float(cost_budget))
    if not math.isfinite(horizon_seconds) or horizon_seconds < 0:
        raise ValueError("horizon_seconds must be finite and non-negative")
    if not math.isfinite(cost_budget) or cost_budget < 0:
        raise ValueError("cost_budget must be finite and non-negative")

    variables = task_count * arm_count
    task_rows = np.repeat(np.arange(task_count), arm_count)
    variable_columns = np.arange(variables)
    task_equalities = sparse.coo_matrix(
        (np.ones(variables), (task_rows, variable_columns)),
        shape=(task_count, variables),
    ).tocsr()

    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    capacities: list[float] = []
    constraint_keys: list[tuple[str, int | None]] = []

    def add_constraint(
        resource: str,
        arm_index: int | None,
        columns: np.ndarray,
        values: np.ndarray,
        capacity: float,
    ) -> None:
        if math.isinf(capacity):
            return
        if not math.isfinite(capacity) or capacity < 0:
            raise ValueError(f"invalid {resource} proxy capacity")
        row = len(capacities)
        row_indices.extend([row] * len(columns))
        column_indices.extend(columns.tolist())
        coefficients.extend(np.asarray(values, dtype=np.float64).tolist())
        capacities.append(float(capacity))
        constraint_keys.append((resource, arm_index))

    add_constraint(
        "cost",
        None,
        np.arange(variables, dtype=np.int64),
        cost.reshape(-1),
        float(cost_budget),
    )

    for arm_index, endpoint in enumerate(endpoints):
        columns = np.arange(task_count, dtype=np.int64) * arm_count + arm_index
        rpm_capacity = (
            endpoint.rpm_bucket_capacity
            + endpoint.rpm_refill_rate * horizon_seconds
        )
        tpm_capacity = (
            endpoint.tpm_bucket_capacity
            + endpoint.tpm_refill_rate * horizon_seconds
        )
        add_constraint(
            "rpm",
            arm_index,
            columns,
            np.ones(task_count),
            (
                float("inf")
                if math.isinf(rpm_capacity)
                else min(float(task_count), rpm_capacity)
            ),
        )
        add_constraint(
            "tpm",
            arm_index,
            columns,
            demand[:, arm_index],
            (
                float("inf")
                if math.isinf(tpm_capacity)
                else min(float(demand[:, arm_index].sum()), tpm_capacity)
            ),
        )

    inequalities = sparse.coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(len(capacities), variables),
    ).tocsr()
    result = optimize.linprog(
        c=-quality.reshape(-1),
        A_ub=inequalities,
        b_ub=np.asarray(capacities, dtype=np.float64),
        A_eq=task_equalities,
        b_eq=np.ones(task_count),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"calibration proxy LP failed: {result.message}")

    beta = 0.0
    gamma_rpm = {endpoint.model_id: 0.0 for endpoint in endpoints}
    gamma_tpm = {endpoint.model_id: 0.0 for endpoint in endpoints}
    marginals = np.maximum(0.0, -np.asarray(result.ineqlin.marginals, dtype=np.float64))
    for value, (resource, arm_index) in zip(marginals, constraint_keys, strict=True):
        if resource == "cost":
            beta = float(value)
        elif resource == "rpm":
            gamma_rpm[endpoints[int(arm_index)].model_id] = float(value)
        elif resource == "tpm":
            gamma_tpm[endpoints[int(arm_index)].model_id] = float(value)
    return CalibrationDualPrices(
        beta=beta,
        quota_prices=QuotaProxyPrices(gamma_rpm, gamma_tpm),
        cost_budget=float(cost_budget),
    )


def quota_proxy_gamma(
    calibration_utility: np.ndarray,
    predicted_token_demand: np.ndarray,
    endpoints: Sequence[EndpointSpec],
    *,
    horizon_seconds: float,
) -> QuotaProxyPrices:
    """Solve a joint RPM/TPM aggregate-capacity proxy LP.

    Each fractional task assignment consumes one RPM request and its frozen
    predicted input-plus-output token demand.  Disabled resources have infinite
    rate/capacity and are omitted from the LP.  HiGHS upper-bound marginals are
    mapped directly to non-negative max-utility shadow prices; no common-offset
    subtraction is valid for task-varying TPM demand.
    """

    utility = np.asarray(calibration_utility, dtype=np.float64)
    demand = np.asarray(predicted_token_demand, dtype=np.float64)
    if utility.ndim != 2 or demand.shape != utility.shape:
        raise ValueError("utility and predicted_token_demand must share [task, arm] shape")
    if np.any(~np.isfinite(utility)) or np.any(~np.isfinite(demand)) or np.any(demand <= 0):
        raise ValueError("utility must be finite and predicted token demand positive/finite")
    task_count, arm_count = utility.shape
    if len(endpoints) != arm_count:
        raise ValueError("calibration utility arm count and endpoint count differ")
    zero_rpm = {endpoint.model_id: 0.0 for endpoint in endpoints}
    zero_tpm = {endpoint.model_id: 0.0 for endpoint in endpoints}
    if task_count == 0:
        return QuotaProxyPrices(zero_rpm, zero_tpm)
    if not math.isfinite(horizon_seconds) or horizon_seconds < 0:
        raise ValueError("horizon_seconds must be finite and non-negative")

    variables = task_count * arm_count
    task_rows = np.repeat(np.arange(task_count), arm_count)
    variable_columns = np.arange(variables)
    task_equalities = sparse.coo_matrix(
        (np.ones(variables), (task_rows, variable_columns)),
        shape=(task_count, variables),
    ).tocsr()

    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    capacities: list[float] = []
    constraint_keys: list[tuple[str, int]] = []

    def add_constraint(resource: str, arm_index: int, values: np.ndarray, capacity: float) -> None:
        if math.isinf(capacity):
            return
        if not math.isfinite(capacity) or capacity < 0:
            raise ValueError(f"invalid {resource} proxy capacity for arm {arm_index}")
        row = len(capacities)
        columns = np.arange(task_count, dtype=np.int64) * arm_count + arm_index
        row_indices.extend([row] * task_count)
        column_indices.extend(columns.tolist())
        coefficients.extend(np.asarray(values, dtype=np.float64).tolist())
        capacities.append(float(capacity))
        constraint_keys.append((resource, arm_index))

    for arm_index, endpoint in enumerate(endpoints):
        rpm_capacity = (
            endpoint.rpm_bucket_capacity
            + endpoint.rpm_refill_rate * horizon_seconds
        )
        tpm_capacity = (
            endpoint.tpm_bucket_capacity
            + endpoint.tpm_refill_rate * horizon_seconds
        )
        add_constraint(
            "rpm",
            arm_index,
            np.ones(task_count),
            (
                float("inf")
                if math.isinf(rpm_capacity)
                else min(float(task_count), rpm_capacity)
            ),
        )
        add_constraint(
            "tpm",
            arm_index,
            demand[:, arm_index],
            (
                float("inf")
                if math.isinf(tpm_capacity)
                else min(float(demand[:, arm_index].sum()), tpm_capacity)
            ),
        )

    if capacities:
        endpoint_inequalities = sparse.coo_matrix(
            (coefficients, (row_indices, column_indices)),
            shape=(len(capacities), variables),
        ).tocsr()
        a_ub: sparse.csr_matrix | None = endpoint_inequalities
        b_ub: np.ndarray | None = np.asarray(capacities, dtype=np.float64)
    else:
        a_ub = None
        b_ub = None
    result = optimize.linprog(
        c=-utility.reshape(-1),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=task_equalities,
        b_eq=np.ones(task_count),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"joint quota proxy LP failed: {result.message}")

    gamma_rpm = dict(zero_rpm)
    gamma_tpm = dict(zero_tpm)
    if constraint_keys:
        marginals = np.maximum(
            0.0, -np.asarray(result.ineqlin.marginals, dtype=np.float64)
        )
        for value, (resource, arm_index) in zip(
            marginals, constraint_keys, strict=True
        ):
            model_id = endpoints[arm_index].model_id
            if resource == "rpm":
                gamma_rpm[model_id] = float(value)
            else:
                gamma_tpm[model_id] = float(value)
    return QuotaProxyPrices(gamma_rpm, gamma_tpm)


def _weights(candidate: CandidateEstimate) -> np.ndarray:
    if candidate.neighbor_weights:
        return np.asarray(candidate.neighbor_weights, dtype=np.float64)
    count = len(candidate.static_latency_samples)
    return np.full(count, 1.0 / count, dtype=np.float64)


def _weighted_cdf(candidate: CandidateEstimate, threshold: float) -> float:
    if threshold < 0:
        return 0.0
    samples = np.asarray(candidate.static_latency_samples, dtype=np.float64)
    return float(_weights(candidate)[samples <= threshold].sum())


def _weighted_quantile(candidate: CandidateEstimate, probability: float) -> float:
    samples = np.asarray(candidate.static_latency_samples, dtype=np.float64)
    weights = _weights(candidate)
    order = np.argsort(samples, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = min(int(np.searchsorted(cumulative, probability, side="left")), len(order) - 1)
    return float(samples[order[position]])


def _binding_flags(value: Any) -> tuple[str, bool, bool]:
    binding = str(getattr(value, "value", value)).lower()
    rpm = binding in {"rpm", "both", "joint", "rpm+tpm"}
    tpm = binding in {"tpm", "both", "joint", "rpm+tpm"}
    return binding, rpm, tpm


class _PolicyController:
    """Common static-profile controller for main and diagnostic policies."""

    def __init__(
        self,
        policy: str,
        endpoints: Sequence[EndpointSpec],
        *,
        beta: float,
        lambda_penalty: float,
        prices: QuotaProxyPrices,
        seed: int,
        initial_rpm_tokens: float | None,
        initial_tpm_tokens: float | None,
    ) -> None:
        canonical = POLICY_ALIASES.get(policy, policy)
        if canonical not in REFERENCE_POLICIES:
            raise ValueError(f"unsupported policy {policy!r}")
        self.policy = canonical
        self.endpoints = {endpoint.model_id: endpoint for endpoint in endpoints}
        self.beta = float(beta)
        self.lambda_penalty = float(lambda_penalty)
        self.prices = prices
        self.rng = np.random.default_rng(seed)
        kwargs: dict[str, float] = {}
        if initial_rpm_tokens is not None:
            kwargs["initial_rpm_tokens"] = float(initial_rpm_tokens)
        if initial_tpm_tokens is not None:
            kwargs["initial_tpm_tokens"] = float(initial_tpm_tokens)
        self.limiters = {
            endpoint.model_id: QuotaLimiter(endpoint, **kwargs) for endpoint in endpoints
        }

    def _score(
        self,
        candidate: CandidateEstimate,
        *,
        task_index: int,
        arrival_time: float,
        deadline: float,
    ) -> CandidateEstimate:
        try:
            preview = self.limiters[candidate.model_id].preview(
                arrival_time, candidate.predicted_token_demand
            )
        except ValueError as error:
            raise ValueError(
                f"request task_index={task_index} model={candidate.model_id} has "
                f"predicted_token_demand={candidate.predicted_token_demand:g} that "
                "cannot be reserved by the configured TPM bucket"
            ) from error

        response_p95 = _weighted_quantile(candidate, 0.95)
        residual_deadline = deadline - preview.admission_wait
        risk = 1.0 if residual_deadline <= 0 else 1.0 - _weighted_cdf(
            candidate, residual_deadline
        )
        quota_price = self.prices.for_endpoint(candidate.model_id)
        scarcity = (
            quota_price.gamma_rpm
            + quota_price.gamma_tpm * candidate.predicted_token_demand
        )
        score = (
            candidate.predicted_quality
            - self.beta * candidate.normalized_cost
            - scarcity
            - self.lambda_penalty * risk
        )
        return replace(
            candidate,
            quota_preview=preview,
            predicted_risk=float(min(1.0, max(0.0, risk))),
            predicted_response_p95=response_p95,
            # Diagnostic E2E p95 always includes the real previewed admission
            # wait. Static-risk differs only in the risk threshold it scores.
            predicted_e2e_p95=response_p95 + preview.admission_wait,
            scarcity_penalty=scarcity,
            score=score,
        )

    def _choose(self, scored: tuple[CandidateEstimate, ...]) -> CandidateEstimate:
        if self.policy == "random":
            return scored[int(self.rng.integers(0, len(scored)))]
        if self.policy == "best_quality":
            return min(scored, key=lambda item: (-item.predicted_quality, item.model_id))
        if self.policy == "min_latency_risk":
            return min(
                scored,
                key=lambda item: (
                    item.predicted_risk,
                    item.predicted_e2e_p95,
                    item.model_id,
                ),
            )
        if self.policy == "available":
            def qc(item: CandidateEstimate) -> float:
                return (
                    item.predicted_quality
                    - self.beta * item.normalized_cost
                    - item.scarcity_penalty
                )

            immediate = [
                item
                for item in scored
                if item.quota_preview is not None
                and item.quota_preview.admission_wait <= 1e-12
            ]
            if immediate:
                return min(immediate, key=lambda item: (-qc(item), item.model_id))
            return min(
                scored,
                key=lambda item: (
                    item.quota_preview.admission_wait,
                    -qc(item),
                    item.model_id,
                ),
            )

        best_score = max(item.score for item in scored)
        tied = tuple(
            item
            for item in scored
            if math.isclose(item.score, best_score, rel_tol=0.0, abs_tol=1e-12)
        )
        if (
            self.lambda_penalty > 0.0
            and len(tied) > 1
            and all(math.isclose(item.predicted_risk, 1.0, abs_tol=1e-12) for item in tied)
        ):
            return min(tied, key=lambda item: (item.predicted_e2e_p95, item.model_id))
        return min(tied, key=lambda item: item.model_id)

    def route(
        self,
        request_id: str,
        task_index: int,
        arrival_time: float,
        deadline: float,
        candidates: Sequence[CandidateEstimate],
    ) -> _DecisionView:
        scored = tuple(
            self._score(
                candidate,
                task_index=task_index,
                arrival_time=arrival_time,
                deadline=deadline,
            )
            for candidate in candidates
        )
        selected = self._choose(scored)
        reservation = self.limiters[selected.model_id].commit(
            arrival_time,
            selected.predicted_token_demand,
            request_id=request_id,
        )
        return _DecisionView(selected.model_id, selected, reservation, scored)


class _CoreControllerAdapter:
    """Expose the public core Router.route through the simulator controller shape."""

    _MODES: Mapping[str, RoutingMode] = {
        "quality_cost": RoutingMode.NO_LATENCY,
        "static_latency": RoutingMode.STATIC,
        "rpm_aware": RoutingMode.ADMISSION,
        "rpm_aware_no_gamma": RoutingMode.ADMISSION,
        "rpm_aware_lambda_0": RoutingMode.ADMISSION,
        "static_latency_no_gamma": RoutingMode.STATIC,
    }

    def __init__(
        self,
        policy: str,
        endpoints: Sequence[EndpointSpec],
        *,
        beta: float,
        lambda_penalty: float,
        prices: QuotaProxyPrices,
        initial_rpm_tokens: float | None,
        initial_tpm_tokens: float | None,
    ) -> None:
        if policy not in self._MODES:
            raise ValueError(f"unsupported core policy {policy!r}")
        balances = None
        if initial_rpm_tokens is not None or initial_tpm_tokens is not None:
            balances = {
                endpoint.model_id: (
                    endpoint.rpm_bucket_capacity
                    if initial_rpm_tokens is None
                    else float(initial_rpm_tokens),
                    endpoint.tpm_bucket_capacity
                    if initial_tpm_tokens is None
                    else float(initial_tpm_tokens),
                )
                for endpoint in endpoints
            }
        arm_ids = tuple(endpoint.model_id for endpoint in endpoints)
        self.router = PortAQSRouter(
            endpoints,
            # Every route supplies its current kNN samples/weights.  These
            # constructor-only dummies avoid aggregating future stream profiles.
            LatencyEstimator({arm: (1.0,) for arm in arm_ids}),
            beta=beta,
            lambda_penalty=lambda_penalty,
            quota_prices=prices.as_core_mapping(arm_ids),
            mode=self._MODES[policy],
            initial_quota_balances=balances,
        )

    def route(
        self,
        request_id: str,
        task_index: int,
        arrival_time: float,
        deadline: float,
        candidates: Sequence[CandidateEstimate],
    ) -> Any:
        for candidate in candidates:
            endpoint = self.router.endpoints[candidate.model_id]
            if (
                math.isfinite(endpoint.tpm_bucket_capacity)
                and candidate.predicted_token_demand
                > endpoint.tpm_bucket_capacity + 1e-9
            ):
                raise ValueError(
                    f"request task_index={task_index} model={candidate.model_id} has "
                    f"predicted_token_demand={candidate.predicted_token_demand:g} that "
                    "cannot be reserved by the configured TPM bucket"
                )
        try:
            return self.router.route(request_id, arrival_time, deadline, candidates)
        except ValueError as error:
            if "tpm_bucket_capacity" not in str(error) and "token_demand" not in str(error):
                raise
            raise ValueError(
                f"request task_index={task_index} contains predicted token demand "
                "that cannot be reserved by the configured TPM bucket"
            ) from error


def _snapshot_by_resource(snapshot: Any, resource: str) -> dict[str, Any]:
    common = {
        "model_id",
        "timestamp",
        "queued",
        "next_dispatch_time",
        "total_committed",
        "total_dispatched",
    }
    values = asdict(snapshot)
    return {
        key: value
        for key, value in values.items()
        if key in common or key.startswith(resource)
    }


def simulate_policy(
    *,
    policy: str,
    frame: pd.DataFrame,
    arms: Sequence[str],
    arrivals: Sequence[float],
    actual_quality: np.ndarray,
    actual_cost: np.ndarray,
    actual_output_tokens: np.ndarray,
    profiles: KNNPrediction,
    world: SyntheticLatencyTable,
    endpoints: Sequence[EndpointSpec],
    cost_scale: float,
    quota_prices: QuotaProxyPrices | None = None,
    config: SimulationConfig = SimulationConfig(),
    seed: int = 2025,
) -> pd.DataFrame:
    """Run a policy and return a fully drained, one-row-per-request trace.

    Actual output tokens and API latency are read only *after* the endpoint is
    selected.  They are evaluation outcomes: neither completion nor reservation
    error reconciles or mutates RPM/TPM quota state.
    """

    canonical_policy = POLICY_ALIASES.get(policy, policy)
    if canonical_policy not in ALL_POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choices are {ALL_POLICIES}")
    arm_tuple = tuple(str(arm) for arm in arms)
    arrival_array = np.asarray(arrivals, dtype=np.float64)
    task_count, arm_count = len(frame), len(arm_tuple)
    expected_shape = (task_count, arm_count)
    if arrival_array.shape != (task_count,) or np.any(np.diff(arrival_array) < 0):
        raise ValueError("arrivals must be one non-decreasing timestamp per task")
    for name, values in (
        ("actual_quality", actual_quality),
        ("actual_cost", actual_cost),
        ("actual_output_tokens", actual_output_tokens),
    ):
        if np.asarray(values).shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    if (
        profiles.quality.shape != expected_shape
        or profiles.cost.shape != expected_shape
        or profiles.output_tokens.shape != expected_shape
    ):
        raise ValueError(f"profile predictions must have shape {expected_shape}")
    if tuple(endpoint.model_id for endpoint in endpoints) != arm_tuple:
        raise ValueError("endpoint order must exactly match arms")
    if cost_scale <= 0 or not math.isfinite(cost_scale):
        raise ValueError("cost_scale must be positive and finite")
    if "input_tokens" not in frame:
        raise ValueError("frame must contain observed input_tokens")

    prices = quota_prices or QuotaProxyPrices(
        {arm: 0.0 for arm in arm_tuple}, {arm: 0.0 for arm in arm_tuple}
    )
    effective_prices = prices
    effective_lambda = config.lambda_penalty
    if canonical_policy in {"rpm_aware_no_gamma", "static_latency_no_gamma"}:
        effective_prices = QuotaProxyPrices(
            {arm: 0.0 for arm in arm_tuple}, {arm: 0.0 for arm in arm_tuple}
        )
    if canonical_policy == "rpm_aware_lambda_0":
        effective_lambda = 0.0
    if canonical_policy in STANDARD_POLICIES:
        controller: _CoreControllerAdapter | _PolicyController = _CoreControllerAdapter(
            canonical_policy,
            endpoints,
            beta=config.beta,
            lambda_penalty=effective_lambda,
            prices=effective_prices,
            initial_rpm_tokens=config.initial_rpm_tokens,
            initial_tpm_tokens=config.initial_tpm_tokens,
        )
    else:
        controller = _PolicyController(
            canonical_policy,
            endpoints,
            beta=config.beta,
            lambda_penalty=effective_lambda,
            prices=effective_prices,
            seed=seed,
            initial_rpm_tokens=config.initial_rpm_tokens,
            initial_tpm_tokens=config.initial_tpm_tokens,
        )
    input_tokens = frame["input_tokens"].to_numpy(dtype=np.float64)
    predicted_demand = input_tokens[:, None] + np.maximum(profiles.output_tokens, 0.0)
    records: list[dict[str, Any]] = []

    for task_index, (_, task) in enumerate(frame.reset_index(drop=True).iterrows()):
        arrival_time = float(arrival_array[task_index])
        candidates: list[CandidateEstimate] = []
        for arm_index, arm in enumerate(arm_tuple):
            samples = tuple(
                float(max(value, 1e-9))
                for value in profiles.latency_samples[task_index, :, arm_index]
            )
            candidates.append(
                CandidateEstimate(
                    model_id=arm,
                    predicted_quality=float(profiles.quality[task_index, arm_index]),
                    normalized_cost=float(profiles.cost[task_index, arm_index] / cost_scale),
                    monetary_cost=float(max(0.0, profiles.cost[task_index, arm_index])),
                    predicted_token_demand=float(max(1.0, predicted_demand[task_index, arm_index])),
                    static_latency_samples=samples,
                    neighbor_weights=tuple(
                        float(value) for value in profiles.neighbor_weights[task_index]
                    ),
                )
            )

        request_id = str(task["task_id"])
        decision = controller.route(
            request_id,
            task_index,
            arrival_time,
            config.deadline_seconds,
            candidates,
        )
        selected = decision.selected_candidate
        reservation = decision.reservation
        arm_index = arm_tuple.index(decision.selected_model_id)
        dispatch_time = reservation.dispatch_time

        # Potential outcomes become visible only after route.  They are never
        # passed to QuotaLimiter and completion has no state-transition hook.
        api_latency = world.response_seconds(task_index, decision.selected_model_id)
        actual_token_demand = float(
            input_tokens[task_index] + actual_output_tokens[task_index, arm_index]
        )
        completion_time = dispatch_time + api_latency
        e2e = completion_time - arrival_time
        snapshot = reservation.snapshot_before
        binding, rpm_binding, tpm_binding = _binding_flags(reservation.binding_resource)

        candidate_diagnostics: list[dict[str, Any]] = []
        for scored in decision.candidates:
            preview = scored.quota_preview
            preview_binding, preview_rpm_binding, preview_tpm_binding = _binding_flags(
                preview.binding_resource
            )
            candidate_diagnostics.append(
                {
                    "model_id": scored.model_id,
                    "predicted_quality": scored.predicted_quality,
                    "predicted_cost": scored.monetary_cost,
                    "normalized_cost": scored.normalized_cost,
                    "predicted_output_tokens": max(
                        0.0,
                        scored.predicted_token_demand - input_tokens[task_index],
                    ),
                    "predicted_token_demand": scored.predicted_token_demand,
                    "scarcity_penalty": scored.scarcity_penalty,
                    "quota_snapshot": asdict(preview.snapshot),
                    "admission_wait": preview.admission_wait,
                    "rpm_wait": preview.rpm_ready_wait,
                    "tpm_wait": preview.tpm_ready_wait,
                    "binding_resource": preview_binding,
                    "rpm_binding": preview_rpm_binding,
                    "tpm_binding": preview_tpm_binding,
                    "predicted_violation": scored.predicted_risk,
                    "predicted_response_p95": scored.predicted_response_p95,
                    "predicted_e2e_p95": scored.predicted_e2e_p95,
                    "score": scored.score,
                }
            )

        reserved_token_demand = float(reservation.token_demand)
        record: dict[str, Any] = {
            "request_id": request_id,
            "eval_name": str(task["eval_name"]),
            "policy": canonical_policy,
            "model_id": decision.selected_model_id,
            "arrival_time": arrival_time,
            "deadline": config.deadline_seconds,
            "routing_latency": 0.0,
            "dispatch_time": dispatch_time,
            "completion_time": completion_time,
            "admission_wait": reservation.admission_wait,
            "rpm_wait": reservation.rpm_ready_wait,
            "tpm_wait": reservation.tpm_ready_wait,
            "binding_resource": binding,
            "rpm_binding": rpm_binding,
            "tpm_binding": tpm_binding,
            "api_latency": api_latency,
            "e2e_latency": e2e,
            "slo_violated": bool(e2e > config.deadline_seconds),
            "predicted_quality": selected.predicted_quality,
            "predicted_cost": selected.monetary_cost,
            "normalized_cost": selected.normalized_cost,
            "predicted_violation": selected.predicted_risk,
            "predicted_response_p95": selected.predicted_response_p95,
            "predicted_e2e_p95": selected.predicted_e2e_p95,
            "score": selected.score,
            "scarcity_penalty": selected.scarcity_penalty,
            "quality": float(actual_quality[task_index, arm_index]),
            "monetary_cost": float(actual_cost[task_index, arm_index]),
            "input_tokens": float(input_tokens[task_index]),
            "predicted_output_tokens": float(profiles.output_tokens[task_index, arm_index]),
            "actual_output_tokens": float(actual_output_tokens[task_index, arm_index]),
            "predicted_token_demand": selected.predicted_token_demand,
            "reserved_token_demand": reserved_token_demand,
            "actual_token_demand": actual_token_demand,
            "token_prediction_error": actual_token_demand - selected.predicted_token_demand,
            "token_reservation_error": actual_token_demand - reserved_token_demand,
            "quota_snapshot": json.dumps(
                json_ready(asdict(snapshot)), sort_keys=True, allow_nan=False
            ),
            "rpm_quota_snapshot": json.dumps(
                json_ready(_snapshot_by_resource(snapshot, "rpm")),
                sort_keys=True,
                allow_nan=False,
            ),
            "tpm_quota_snapshot": json.dumps(
                json_ready(_snapshot_by_resource(snapshot, "tpm")),
                sort_keys=True,
                allow_nan=False,
            ),
            "candidate_diagnostics": json.dumps(
                json_ready(candidate_diagnostics), sort_keys=True, allow_nan=False
            ),
            "status": "completed",
        }
        records.append(record)

    trace = pd.DataFrame.from_records(records)
    if len(trace) != task_count or not (trace["status"] == "completed").all():
        raise RuntimeError("simulator failed to drain every committed request")
    return trace
