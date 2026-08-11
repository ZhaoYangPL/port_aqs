"""Typed records shared by the feedback-free PORT-AQS Stage 1 core.

Times and durations are seconds.  ``deadline`` is an SLO duration measured
from request arrival.  "RPM token" below means an API-call permit; "TPM
token" means an estimated LLM input/output token.  Keeping those names
separate avoids treating requests with different token demands as identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Final


_EPS: Final[float] = 1e-9


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _nonnegative(name: str, value: float) -> None:
    _finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _positive_or_infinite(name: str, value: float) -> None:
    if math.isnan(value) or value <= 0:
        raise ValueError(f"{name} must be positive or +inf, got {value!r}")
    if math.isinf(value) and value < 0:  # pragma: no cover - made explicit
        raise ValueError(f"{name} cannot be -inf")


def _nonnegative_or_infinite(name: str, value: float) -> None:
    if math.isnan(value) or value < 0:
        raise ValueError(f"{name} must be non-negative or +inf, got {value!r}")


class RoutingMode(str, Enum):
    """Latency information available to a feedback-free Stage 1 policy."""

    NO_LATENCY = "no_latency"
    STATIC = "static"
    ADMISSION = "admission"

    @classmethod
    def parse(cls, value: RoutingMode | str) -> RoutingMode:
        if isinstance(value, cls):
            return value
        aliases = {
            "none": cls.NO_LATENCY,
            "quality_cost": cls.NO_LATENCY,
            "static_latency": cls.STATIC,
            "static_response": cls.STATIC,
            "rpm_aware": cls.ADMISSION,
            "quota_admission": cls.ADMISSION,
            "dual_quota_admission": cls.ADMISSION,
        }
        if value in aliases:
            return aliases[value]
        try:
            return cls(value)
        except ValueError as error:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"unknown routing mode {value!r}; expected one of {choices}"
            ) from error


@dataclass(frozen=True, slots=True)
class EndpointSpec:
    """Independent RPM and TPM token-bucket configuration for one endpoint.

    A rate and its capacity may both be ``+inf`` to disable that resource in
    an ablation.  Finite TPM capacity must cover every predicted request
    reservation; the limiter rejects an oversized reservation rather than
    clipping it silently.
    """

    model_id: str
    rpm: float
    rpm_bucket_capacity: float
    tpm: float
    tpm_bucket_capacity: float

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        _positive_or_infinite("rpm", self.rpm)
        _positive_or_infinite("tpm", self.tpm)
        _positive_or_infinite("rpm_bucket_capacity", self.rpm_bucket_capacity)
        _positive_or_infinite("tpm_bucket_capacity", self.tpm_bucket_capacity)
        if self.rpm_bucket_capacity < 1:
            raise ValueError("rpm_bucket_capacity must be at least one request")

    @property
    def rpm_refill_rate(self) -> float:
        """API-call permits replenished per second."""

        return self.rpm / 60.0

    @property
    def tpm_refill_rate(self) -> float:
        """LLM-token permits replenished per second."""

        return self.tpm / 60.0

    # Read-only aliases ease migration of code that only needs the RPM view.
    @property
    def refill_rate(self) -> float:
        return self.rpm_refill_rate

    @property
    def bucket_capacity(self) -> float:
        return self.rpm_bucket_capacity


@dataclass(frozen=True, slots=True)
class QuotaPrice:
    """Frozen scarcity prices obtained only from calibration data."""

    gamma_rpm: float = 0.0
    gamma_tpm: float = 0.0

    def __post_init__(self) -> None:
        _nonnegative("gamma_rpm", self.gamma_rpm)
        _nonnegative("gamma_tpm", self.gamma_tpm)

    def penalty(self, predicted_token_demand: float) -> float:
        _nonnegative("predicted_token_demand", predicted_token_demand)
        return self.gamma_rpm + self.gamma_tpm * predicted_token_demand


@dataclass(frozen=True, slots=True)
class QuotaSnapshot:
    """A causal, physical two-bucket view at one timestamp.

    Balances never become negative.  Future committed calls are represented by
    the FIFO fields rather than by negative balances.  TPM conservation here
    concerns predicted reservations, not the post-call actual token count.
    """

    model_id: str
    timestamp: float
    rpm_capacity: float
    rpm_refill_rate: float
    rpm_tokens: float
    tpm_capacity: float
    tpm_refill_rate: float
    tpm_tokens: float
    queued: int
    queued_tpm_demand: float
    next_dispatch_time: float | None
    total_committed: int
    total_dispatched: int
    total_tpm_committed: float
    total_tpm_dispatched: float

    def __post_init__(self) -> None:
        _finite("timestamp", self.timestamp)
        for name, value in (
            ("rpm_capacity", self.rpm_capacity),
            ("rpm_refill_rate", self.rpm_refill_rate),
            ("tpm_capacity", self.tpm_capacity),
            ("tpm_refill_rate", self.tpm_refill_rate),
        ):
            _positive_or_infinite(name, value)
        _nonnegative_or_infinite("rpm_tokens", self.rpm_tokens)
        _nonnegative_or_infinite("tpm_tokens", self.tpm_tokens)
        if self.rpm_tokens > self.rpm_capacity + _EPS:
            raise ValueError("rpm_tokens cannot exceed RPM bucket capacity")
        if self.tpm_tokens > self.tpm_capacity + _EPS:
            raise ValueError("tpm_tokens cannot exceed TPM bucket capacity")
        if self.queued < 0:
            raise ValueError("queued cannot be negative")
        _nonnegative("queued_tpm_demand", self.queued_tpm_demand)
        if self.next_dispatch_time is not None:
            _finite("next_dispatch_time", self.next_dispatch_time)
            if self.next_dispatch_time < self.timestamp - _EPS:
                raise ValueError("next_dispatch_time cannot precede the snapshot")
        if self.total_committed < 0 or self.total_dispatched < 0:
            raise ValueError("quota counters cannot be negative")
        _nonnegative("total_tpm_committed", self.total_tpm_committed)
        _nonnegative("total_tpm_dispatched", self.total_tpm_dispatched)
        if self.total_committed != self.total_dispatched + self.queued:
            raise ValueError("RPM conservation failed: committed != dispatched + queued")
        if not math.isclose(
            self.total_tpm_committed,
            self.total_tpm_dispatched + self.queued_tpm_demand,
            rel_tol=1e-10,
            abs_tol=_EPS,
        ):
            raise ValueError(
                "TPM reservation conservation failed: committed != dispatched + queued"
            )

    @property
    def rpm_virtual_balance(self) -> float:
        return self.rpm_tokens - float(self.queued)

    @property
    def tpm_virtual_balance(self) -> float:
        return self.tpm_tokens - self.queued_tpm_demand

    # Compatibility aliases: these are deliberately the RPM bucket only.
    @property
    def capacity(self) -> float:
        return self.rpm_capacity

    @property
    def refill_rate(self) -> float:
        return self.rpm_refill_rate

    @property
    def tokens(self) -> float:
        return self.rpm_tokens


@dataclass(frozen=True, slots=True)
class QuotaPreview:
    """Non-mutating prediction of the next joint FIFO admission."""

    model_id: str
    arrival_time: float
    dispatch_time: float
    admission_wait: float
    rpm_ready_wait: float
    tpm_ready_wait: float
    binding_resource: str
    token_demand: float
    queue_position: int
    snapshot: QuotaSnapshot

    def __post_init__(self) -> None:
        _finite("arrival_time", self.arrival_time)
        _finite("dispatch_time", self.dispatch_time)
        for name, value in (
            ("admission_wait", self.admission_wait),
            ("rpm_ready_wait", self.rpm_ready_wait),
            ("tpm_ready_wait", self.tpm_ready_wait),
            ("token_demand", self.token_demand),
        ):
            _nonnegative(name, value)
        if self.token_demand <= 0:
            raise ValueError("token_demand must be positive")
        if self.dispatch_time < self.arrival_time - _EPS:
            raise ValueError("dispatch_time cannot precede arrival_time")
        if not math.isclose(
            self.admission_wait,
            max(self.rpm_ready_wait, self.tpm_ready_wait),
            rel_tol=0.0,
            abs_tol=_EPS,
        ):
            raise ValueError("admission_wait must be max(RPM ready wait, TPM ready wait)")
        if self.binding_resource not in {"none", "rpm", "tpm", "both"}:
            raise ValueError("binding_resource must be none, rpm, tpm, or both")
        if self.queue_position < 0:
            raise ValueError("queue_position cannot be negative")


@dataclass(frozen=True, slots=True)
class QuotaReservation:
    """A committed joint RPM/TPM consumption in endpoint FIFO order."""

    request_id: str
    model_id: str
    sequence: int
    arrival_time: float
    dispatch_time: float
    admission_wait: float
    rpm_ready_wait: float
    tpm_ready_wait: float
    binding_resource: str
    token_demand: float
    queue_position: int
    snapshot_before: QuotaSnapshot
    snapshot_after: QuotaSnapshot

    def __post_init__(self) -> None:
        if self.sequence <= 0:
            raise ValueError("sequence must be positive")
        # Reuse the preview validation for all shared scheduling fields.
        QuotaPreview(
            model_id=self.model_id,
            arrival_time=self.arrival_time,
            dispatch_time=self.dispatch_time,
            admission_wait=self.admission_wait,
            rpm_ready_wait=self.rpm_ready_wait,
            tpm_ready_wait=self.tpm_ready_wait,
            binding_resource=self.binding_resource,
            token_demand=self.token_demand,
            queue_position=self.queue_position,
            snapshot=self.snapshot_before,
        )


@dataclass(frozen=True, slots=True)
class CandidateEstimate:
    """Static, task-conditioned routing input and its scored diagnostics."""

    model_id: str
    predicted_quality: float
    normalized_cost: float
    predicted_token_demand: float = 1.0
    monetary_cost: float = 0.0
    static_latency_samples: tuple[float, ...] = field(default_factory=tuple)
    neighbor_weights: tuple[float, ...] = field(default_factory=tuple)
    quota_preview: QuotaPreview | None = None
    predicted_risk: float = 0.0
    predicted_response_p95: float = math.nan
    predicted_e2e_p95: float = math.nan
    scarcity_penalty: float = 0.0
    score: float = math.nan

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        _finite("predicted_quality", self.predicted_quality)
        _finite("normalized_cost", self.normalized_cost)
        _finite("predicted_token_demand", self.predicted_token_demand)
        if self.predicted_token_demand <= 0:
            raise ValueError("predicted_token_demand must be positive")
        _nonnegative("monetary_cost", self.monetary_cost)
        samples = tuple(float(value) for value in self.static_latency_samples)
        object.__setattr__(self, "static_latency_samples", samples)
        if any((not math.isfinite(value) or value <= 0) for value in samples):
            raise ValueError("static_latency_samples must contain positive finite seconds")
        weights = tuple(float(value) for value in self.neighbor_weights)
        if weights:
            if len(weights) != len(samples):
                raise ValueError(
                    "neighbor_weights must have one value per static latency sample"
                )
            if any((not math.isfinite(value) or value < 0) for value in weights):
                raise ValueError("neighbor_weights must contain finite non-negative values")
            total = sum(weights)
            if total <= 0:
                raise ValueError("neighbor_weights must have positive total mass")
            weights = tuple(value / total for value in weights)
        object.__setattr__(self, "neighbor_weights", weights)
        _nonnegative("predicted_risk", self.predicted_risk)
        if self.predicted_risk > 1 + _EPS:
            raise ValueError("predicted_risk cannot exceed one")
        _nonnegative("scarcity_penalty", self.scarcity_penalty)
        for name, value in (
            ("predicted_response_p95", self.predicted_response_p95),
            ("predicted_e2e_p95", self.predicted_e2e_p95),
            ("score", self.score),
        ):
            if not (math.isnan(value) or math.isfinite(value)):
                raise ValueError(f"{name} must be finite or NaN")


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """A deterministic, one-shot routing choice and quota reservation."""

    request_id: str
    arrival_time: float
    deadline: float
    mode: RoutingMode
    selected_model_id: str
    selected_candidate: CandidateEstimate
    reservation: QuotaReservation
    candidates: tuple[CandidateEstimate, ...]

    def __post_init__(self) -> None:
        _finite("arrival_time", self.arrival_time)
        _finite("deadline", self.deadline)
        if self.deadline <= 0:
            raise ValueError("deadline must be positive")
        if self.selected_candidate.model_id != self.selected_model_id:
            raise ValueError("selected candidate and model id disagree")
        if self.reservation.model_id != self.selected_model_id:
            raise ValueError("reservation and selected model id disagree")


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """One completed external-API call kept only as an evaluation trace.

    Stage 1 never feeds this record back into either latency or quality
    estimation.  It remains a useful neutral schema for simulator output.
    """

    request_id: str
    model_id: str
    arrival_time: float
    dispatch_time: float
    completion_time: float
    api_latency: float
    selected: bool = True
    status: str = "ok"
    quality: float | None = None
    monetary_cost: float | None = None
    predicted_token_demand: float | None = None
    actual_token_count: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("arrival_time", self.arrival_time),
            ("dispatch_time", self.dispatch_time),
            ("completion_time", self.completion_time),
        ):
            _finite(name, value)
        _nonnegative("api_latency", self.api_latency)
        if self.dispatch_time < self.arrival_time - _EPS:
            raise ValueError("dispatch_time cannot precede arrival_time")
        if self.completion_time < self.dispatch_time - _EPS:
            raise ValueError("completion_time cannot precede dispatch_time")
        if self.monetary_cost is not None:
            _nonnegative("monetary_cost", self.monetary_cost)
        if self.quality is not None:
            _finite("quality", self.quality)
        for name, value in (
            ("predicted_token_demand", self.predicted_token_demand),
            ("actual_token_count", self.actual_token_count),
        ):
            if value is not None:
                _nonnegative(name, value)
