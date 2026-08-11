"""Client-side joint RPM/TPM admission control for PORT-AQS Stage 1.

The limiter schedules API *dispatches*, not server execution.  Every call
needs one RPM permit and a task-dependent predicted number of TPM permits.
Both buckets must be ready and prior reservations retain FIFO order.  API
completion never releases capacity or changes either bucket.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
import math
from threading import RLock

from .types import EndpointSpec, QuotaPreview, QuotaReservation, QuotaSnapshot


_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class _Pending:
    sequence: int
    dispatch_time: float
    token_demand: float
    binding_resource: str


class QuotaLimiter:
    """A causal joint token bucket with one FIFO per endpoint.

    ``preview(timestamp, token_demand)`` is non-mutating.  ``commit`` performs
    the same calculation under a lock and reserves the exact dispatch time.
    Mutating arrival timestamps must be non-decreasing.
    """

    def __init__(
        self,
        spec: EndpointSpec,
        *,
        start_time: float = 0.0,
        initial_rpm_tokens: float | None = None,
        initial_tpm_tokens: float | None = None,
    ) -> None:
        if not math.isfinite(start_time):
            raise ValueError("start_time must be finite")
        rpm_tokens = (
            spec.rpm_bucket_capacity
            if initial_rpm_tokens is None
            else float(initial_rpm_tokens)
        )
        tpm_tokens = (
            spec.tpm_bucket_capacity
            if initial_tpm_tokens is None
            else float(initial_tpm_tokens)
        )
        self._validate_initial(
            "initial_rpm_tokens", rpm_tokens, spec.rpm_bucket_capacity
        )
        self._validate_initial(
            "initial_tpm_tokens", tpm_tokens, spec.tpm_bucket_capacity
        )

        self.spec = spec
        self._rpm_balance = rpm_tokens
        self._tpm_balance = tpm_tokens
        self._last_timestamp = float(start_time)
        self._pending: deque[_Pending] = deque()
        self._sequence = 0
        self._total_dispatched = 0
        self._total_tpm_committed = 0.0
        self._total_tpm_dispatched = 0.0
        # RPM-only Stage 1 can be represented by one virtual balance.  Keep an
        # append-only dispatch index solely for exact FIFO diagnostics and
        # conservation snapshots; preview/commit never replay the queue.
        self._rpm_only_fast = (
            math.isfinite(spec.rpm_refill_rate)
            and math.isinf(spec.tpm_refill_rate)
        )
        self._rpm_virtual_balance = rpm_tokens
        self._rpm_dispatch_times: list[float] = []
        self._rpm_tpm_prefix: list[float] = [0.0]
        self._lock = RLock()

    @staticmethod
    def _validate_initial(name: str, value: float, capacity: float) -> None:
        if math.isnan(value) or value < 0 or value > capacity:
            raise ValueError(f"{name} must lie in [0, capacity]")
        if math.isinf(value) and not math.isinf(capacity):
            raise ValueError(f"{name} may be +inf only for an infinite-capacity bucket")

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    def _validate_token_demand(self, token_demand: float) -> float:
        demand = float(token_demand)
        if not math.isfinite(demand) or demand <= 0:
            raise ValueError("token_demand must be positive and finite")
        if (
            math.isfinite(self.spec.tpm_bucket_capacity)
            and demand > self.spec.tpm_bucket_capacity + _EPS
        ):
            raise ValueError(
                f"token_demand {demand} exceeds finite TPM bucket capacity "
                f"{self.spec.tpm_bucket_capacity}; configure a valid reservation "
                "upper bound instead of clipping it"
            )
        return demand

    @staticmethod
    def _refill(balance: float, rate: float, capacity: float, elapsed: float) -> float:
        if elapsed < -_EPS:  # pragma: no cover - internal schedule invariant
            raise RuntimeError("cannot refill backward in time")
        if math.isinf(rate):
            return capacity
        if elapsed <= _EPS or math.isinf(balance):
            return balance
        return min(capacity, balance + rate * elapsed)

    @staticmethod
    def _consume(balance: float, demand: float, rate: float, resource: str) -> float:
        # An infinite rate disables that resource for ablation purposes.
        if math.isinf(rate):
            return balance
        if balance < demand - _EPS:  # pragma: no cover - arithmetic invariant
            raise RuntimeError(f"reserved dispatch lacks {resource} permits")
        return max(0.0, balance - demand)

    def _refill_pair(
        self,
        rpm_balance: float,
        tpm_balance: float,
        elapsed: float,
    ) -> tuple[float, float]:
        return (
            self._refill(
                rpm_balance,
                self.spec.rpm_refill_rate,
                self.spec.rpm_bucket_capacity,
                elapsed,
            ),
            self._refill(
                tpm_balance,
                self.spec.tpm_refill_rate,
                self.spec.tpm_bucket_capacity,
                elapsed,
            ),
        )

    def _consume_pair(
        self,
        rpm_balance: float,
        tpm_balance: float,
        token_demand: float,
    ) -> tuple[float, float]:
        return (
            self._consume(rpm_balance, 1.0, self.spec.rpm_refill_rate, "RPM"),
            self._consume(
                tpm_balance,
                token_demand,
                self.spec.tpm_refill_rate,
                "TPM",
            ),
        )

    def _project(
        self, timestamp: float
    ) -> tuple[float, float, deque[_Pending], int, float]:
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if timestamp < self._last_timestamp - _EPS:
            raise ValueError(
                f"timestamp {timestamp} precedes committed state at {self._last_timestamp}"
            )

        rpm_balance = self._rpm_balance
        tpm_balance = self._tpm_balance
        pending = deque(self._pending)
        dispatched = self._total_dispatched
        tpm_dispatched = self._total_tpm_dispatched
        cursor = self._last_timestamp

        while pending and pending[0].dispatch_time <= timestamp + _EPS:
            item = pending.popleft()
            elapsed = max(0.0, item.dispatch_time - cursor)
            rpm_balance, tpm_balance = self._refill_pair(
                rpm_balance, tpm_balance, elapsed
            )
            rpm_balance, tpm_balance = self._consume_pair(
                rpm_balance, tpm_balance, item.token_demand
            )
            cursor = max(cursor, item.dispatch_time)
            dispatched += 1
            tpm_dispatched += item.token_demand

        rpm_balance, tpm_balance = self._refill_pair(
            rpm_balance, tpm_balance, max(0.0, timestamp - cursor)
        )
        return rpm_balance, tpm_balance, pending, dispatched, tpm_dispatched

    def _snapshot_from(
        self,
        timestamp: float,
        rpm_balance: float,
        tpm_balance: float,
        pending: deque[_Pending],
        dispatched: int,
        tpm_dispatched: float,
    ) -> QuotaSnapshot:
        return QuotaSnapshot(
            model_id=self.model_id,
            timestamp=timestamp,
            rpm_capacity=self.spec.rpm_bucket_capacity,
            rpm_refill_rate=self.spec.rpm_refill_rate,
            rpm_tokens=max(0.0, min(self.spec.rpm_bucket_capacity, rpm_balance)),
            tpm_capacity=self.spec.tpm_bucket_capacity,
            tpm_refill_rate=self.spec.tpm_refill_rate,
            tpm_tokens=max(0.0, min(self.spec.tpm_bucket_capacity, tpm_balance)),
            queued=len(pending),
            queued_tpm_demand=sum(item.token_demand for item in pending),
            next_dispatch_time=pending[0].dispatch_time if pending else None,
            total_committed=self._sequence,
            total_dispatched=dispatched,
            total_tpm_committed=self._total_tpm_committed,
            total_tpm_dispatched=tpm_dispatched,
        )

    def _rpm_fast_project(self, timestamp: float) -> float:
        """Project the RPM-only virtual balance in O(1)."""

        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if timestamp < self._last_timestamp - _EPS:
            raise ValueError(
                f"timestamp {timestamp} precedes committed state at {self._last_timestamp}"
            )
        elapsed = max(0.0, timestamp - self._last_timestamp)
        return min(
            self.spec.rpm_bucket_capacity,
            self._rpm_virtual_balance + self.spec.rpm_refill_rate * elapsed,
        )

    def _rpm_fast_snapshot(
        self, timestamp: float, virtual_balance: float
    ) -> QuotaSnapshot:
        dispatched = bisect_right(self._rpm_dispatch_times, timestamp + _EPS)
        queued = len(self._rpm_dispatch_times) - dispatched
        # With unit RPM demand, virtual_balance = physical_tokens - queued.
        rpm_tokens = max(
            0.0,
            min(self.spec.rpm_bucket_capacity, virtual_balance + float(queued)),
        )
        total_tpm_committed = self._rpm_tpm_prefix[-1]
        total_tpm_dispatched = self._rpm_tpm_prefix[dispatched]
        return QuotaSnapshot(
            model_id=self.model_id,
            timestamp=timestamp,
            rpm_capacity=self.spec.rpm_bucket_capacity,
            rpm_refill_rate=self.spec.rpm_refill_rate,
            rpm_tokens=rpm_tokens,
            tpm_capacity=self.spec.tpm_bucket_capacity,
            tpm_refill_rate=self.spec.tpm_refill_rate,
            tpm_tokens=self.spec.tpm_bucket_capacity,
            queued=queued,
            queued_tpm_demand=total_tpm_committed - total_tpm_dispatched,
            next_dispatch_time=(
                self._rpm_dispatch_times[dispatched] if queued else None
            ),
            total_committed=len(self._rpm_dispatch_times),
            total_dispatched=dispatched,
            total_tpm_committed=total_tpm_committed,
            total_tpm_dispatched=total_tpm_dispatched,
        )

    def _rpm_fast_wait(self, virtual_balance: float) -> float:
        if virtual_balance >= 1.0 - _EPS:
            return 0.0
        return max(
            0.0,
            (1.0 - virtual_balance) / self.spec.rpm_refill_rate,
        )

    def snapshot(self, timestamp: float) -> QuotaSnapshot:
        """Return a non-mutating projection of both quota buckets."""

        with self._lock:
            if self._rpm_only_fast:
                virtual = self._rpm_fast_project(timestamp)
                return self._rpm_fast_snapshot(timestamp, virtual)
            state = self._project(timestamp)
            return self._snapshot_from(timestamp, *state)

    @staticmethod
    def _resource_wait(balance: float, demand: float, rate: float) -> float:
        if math.isinf(rate) or balance >= demand - _EPS:
            return 0.0
        return max(0.0, (demand - balance) / rate)

    def _schedule_from(
        self,
        timestamp: float,
        token_demand: float,
        rpm_balance: float,
        tpm_balance: float,
        pending: deque[_Pending],
    ) -> tuple[float, float, float, str]:
        """Compute exact readiness behind all current FIFO reservations."""

        cursor = timestamp
        inherited_binding = "none"
        for item in pending:
            elapsed = max(0.0, item.dispatch_time - cursor)
            rpm_balance, tpm_balance = self._refill_pair(
                rpm_balance, tpm_balance, elapsed
            )
            rpm_balance, tpm_balance = self._consume_pair(
                rpm_balance, tpm_balance, item.token_demand
            )
            cursor = item.dispatch_time
            inherited_binding = item.binding_resource

        fifo_wait = max(0.0, cursor - timestamp)
        rpm_extra = self._resource_wait(
            rpm_balance, 1.0, self.spec.rpm_refill_rate
        )
        tpm_extra = self._resource_wait(
            tpm_balance, token_demand, self.spec.tpm_refill_rate
        )
        # A disabled resource is ready by definition and therefore reports
        # zero even while the request is FIFO-blocked by the other bucket.
        # For finite resources, readiness remains conditioned on reaching the
        # FIFO tail: refill can be clipped/wasted while an earlier reservation
        # waits for its other resource, so the shared prefix must be replayed.
        rpm_ready_wait = (
            0.0
            if math.isinf(self.spec.rpm_refill_rate)
            else fifo_wait + rpm_extra
        )
        tpm_ready_wait = (
            0.0
            if math.isinf(self.spec.tpm_refill_rate)
            else fifo_wait + tpm_extra
        )
        admission_wait = max(rpm_ready_wait, tpm_ready_wait)
        if admission_wait < _EPS:
            return 0.0, 0.0, 0.0, "none"

        if rpm_ready_wait > tpm_ready_wait + _EPS:
            binding = "rpm"
        elif tpm_ready_wait > rpm_ready_wait + _EPS:
            binding = "tpm"
        elif rpm_extra <= _EPS and tpm_extra <= _EPS and inherited_binding != "none":
            # Both resources are ready at the tail, so the inherited FIFO
            # bottleneck explains the wait more accurately than "both".
            binding = inherited_binding
        else:
            binding = "both"
        return admission_wait, rpm_ready_wait, tpm_ready_wait, binding

    def preview(self, timestamp: float, token_demand: float) -> QuotaPreview:
        """Predict joint FIFO admission without changing limiter state."""

        demand = self._validate_token_demand(token_demand)
        with self._lock:
            if self._rpm_only_fast:
                virtual = self._rpm_fast_project(timestamp)
                snapshot = self._rpm_fast_snapshot(timestamp, virtual)
                wait = self._rpm_fast_wait(virtual)
                return QuotaPreview(
                    model_id=self.model_id,
                    arrival_time=timestamp,
                    dispatch_time=timestamp + wait,
                    admission_wait=wait,
                    rpm_ready_wait=wait,
                    tpm_ready_wait=0.0,
                    binding_resource="none" if wait <= _EPS else "rpm",
                    token_demand=demand,
                    queue_position=snapshot.queued,
                    snapshot=snapshot,
                )
            rpm_balance, tpm_balance, pending, dispatched, tpm_dispatched = (
                self._project(timestamp)
            )
            snapshot = self._snapshot_from(
                timestamp,
                rpm_balance,
                tpm_balance,
                pending,
                dispatched,
                tpm_dispatched,
            )
            wait, rpm_wait, tpm_wait, binding = self._schedule_from(
                timestamp, demand, rpm_balance, tpm_balance, pending
            )
            return QuotaPreview(
                model_id=self.model_id,
                arrival_time=timestamp,
                dispatch_time=timestamp + wait,
                admission_wait=wait,
                rpm_ready_wait=rpm_wait,
                tpm_ready_wait=tpm_wait,
                binding_resource=binding,
                token_demand=demand,
                queue_position=len(pending),
                snapshot=snapshot,
            )

    def _advance(self, timestamp: float) -> None:
        (
            self._rpm_balance,
            self._tpm_balance,
            self._pending,
            self._total_dispatched,
            self._total_tpm_dispatched,
        ) = self._project(timestamp)
        self._last_timestamp = timestamp

    def commit(
        self,
        timestamp: float,
        token_demand: float,
        request_id: str = "",
    ) -> QuotaReservation:
        """Atomically reserve one RPM permit and predicted TPM demand."""

        demand = self._validate_token_demand(token_demand)
        with self._lock:
            if self._rpm_only_fast:
                virtual = self._rpm_fast_project(timestamp)
                before = self._rpm_fast_snapshot(timestamp, virtual)
                wait = self._rpm_fast_wait(virtual)
                dispatch_time = timestamp + wait
                if (
                    self._rpm_dispatch_times
                    and dispatch_time < self._rpm_dispatch_times[-1] - _EPS
                ):
                    raise RuntimeError("FIFO dispatch times must be non-decreasing")
                self._sequence += 1
                self._total_tpm_committed += demand
                self._rpm_dispatch_times.append(dispatch_time)
                self._rpm_tpm_prefix.append(self._rpm_tpm_prefix[-1] + demand)
                self._rpm_virtual_balance = virtual - 1.0
                self._last_timestamp = timestamp
                after = self._rpm_fast_snapshot(timestamp, self._rpm_virtual_balance)
                return QuotaReservation(
                    request_id=request_id,
                    model_id=self.model_id,
                    sequence=self._sequence,
                    arrival_time=timestamp,
                    dispatch_time=dispatch_time,
                    admission_wait=wait,
                    rpm_ready_wait=wait,
                    tpm_ready_wait=0.0,
                    binding_resource="none" if wait <= _EPS else "rpm",
                    token_demand=demand,
                    queue_position=before.queued,
                    snapshot_before=before,
                    snapshot_after=after,
                )
            self._advance(timestamp)
            before = self._snapshot_from(
                timestamp,
                self._rpm_balance,
                self._tpm_balance,
                self._pending,
                self._total_dispatched,
                self._total_tpm_dispatched,
            )
            queue_position = len(self._pending)
            wait, rpm_wait, tpm_wait, binding = self._schedule_from(
                timestamp,
                demand,
                self._rpm_balance,
                self._tpm_balance,
                self._pending,
            )
            dispatch_time = timestamp + wait

            self._sequence += 1
            self._total_tpm_committed += demand
            sequence = self._sequence
            if wait == 0.0:
                self._rpm_balance, self._tpm_balance = self._consume_pair(
                    self._rpm_balance, self._tpm_balance, demand
                )
                self._total_dispatched += 1
                self._total_tpm_dispatched += demand
            else:
                if self._pending and dispatch_time < self._pending[-1].dispatch_time - _EPS:
                    raise RuntimeError("FIFO dispatch times must be non-decreasing")
                self._pending.append(
                    _Pending(sequence, dispatch_time, demand, binding)
                )

            after = self._snapshot_from(
                timestamp,
                self._rpm_balance,
                self._tpm_balance,
                self._pending,
                self._total_dispatched,
                self._total_tpm_dispatched,
            )
            return QuotaReservation(
                request_id=request_id,
                model_id=self.model_id,
                sequence=sequence,
                arrival_time=timestamp,
                dispatch_time=dispatch_time,
                admission_wait=wait,
                rpm_ready_wait=rpm_wait,
                tpm_ready_wait=tpm_wait,
                binding_resource=binding,
                token_demand=demand,
                queue_position=queue_position,
                snapshot_before=before,
                snapshot_after=after,
            )

    def on_completion(self, completion_time: float) -> None:
        """Explicit no-op: completion is evaluation output, not Stage 1 input."""

        if not math.isfinite(completion_time):
            raise ValueError("completion_time must be finite")
        # Deliberately do not advance time, reconcile actual tokens, release a
        # permit, or modify either balance.
