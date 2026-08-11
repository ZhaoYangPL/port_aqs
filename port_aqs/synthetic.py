"""Deterministic synthetic API latency and arrival traces for PORT-AQS.

Response latency is an intentionally indivisible black-box measurement.  The
generator never labels any part of it as provider queueing or service time.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


# Checked-in permutations prevent accidental tuning of endpoint speed/quota to
# the quality ordering in a particular run.  Prefixes are used for <11 arms.
FIXED_PERMUTATIONS_11: tuple[tuple[int, ...], ...] = (
    (7, 0, 9, 3, 10, 2, 5, 1, 8, 4, 6),
    (2, 8, 4, 10, 1, 6, 0, 9, 5, 7, 3),
    (9, 5, 1, 7, 3, 10, 6, 2, 8, 0, 4),
    (4, 10, 6, 0, 8, 3, 9, 5, 1, 7, 2),
    (6, 3, 10, 5, 0, 8, 2, 7, 4, 1, 9),
)


def _permutation(length: int, index: int) -> np.ndarray:
    if length < 1:
        return np.empty(0, dtype=np.int64)
    if length <= 11:
        prefix = [value for value in FIXED_PERMUTATIONS_11[index % 5] if value < length]
        # A prefix-filter is still a permutation but may correlate across small
        # smoke arm counts.  Rotate to keep the five choices visibly distinct.
        shift = index % length
        return np.roll(np.asarray(prefix, dtype=np.int64), shift)
    seed = int(sha256(f"port-aqs-permutation:{index}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed).permutation(length)


@dataclass(frozen=True)
class EndpointSpeed:
    intercept_seconds: float
    output_tokens_per_second: float


@dataclass(frozen=True)
class LatencyNoiseConfig:
    lognormal_sigma: float = 0.22
    tail_probability: float = 0.01
    tail_multiplier: float = 4.0
    tail_pareto_shape: float = 3.0
    maximum_multiplier: float = 25.0


def make_speed_profiles(
    arms: Sequence[str],
    *,
    permutation_index: int = 0,
    intercept_range: tuple[float, float] = (0.20, 0.85),
    tokens_per_second_range: tuple[float, float] = (16.0, 64.0),
) -> dict[str, EndpointSpeed]:
    """Construct and freeze anonymous endpoint speed parameters."""

    count = len(arms)
    order = _permutation(count, permutation_index)
    intercepts = np.linspace(intercept_range[0], intercept_range[1], count)
    # Low intercept and high generation speed describe the same fast endpoint.
    rates = np.linspace(tokens_per_second_range[1], tokens_per_second_range[0], count)
    return {
        str(arms[position]): EndpointSpeed(
            intercept_seconds=float(intercepts[rank]),
            output_tokens_per_second=float(rates[rank]),
        )
        for position, rank in enumerate(order)
    }


def endpoint_rpms(
    arms: Sequence[str],
    *,
    homogeneous_rpm: float = 60.0,
    heterogeneous: bool = False,
    values: Sequence[float] = (30.0, 60.0, 120.0, 240.0),
    permutation_index: int = 0,
) -> dict[str, float]:
    """Return homogeneous quotas or one of five fixed heterogeneous mappings."""

    if not heterogeneous:
        return {str(arm): float(homogeneous_rpm) for arm in arms}
    if not values or any(value <= 0 for value in values):
        raise ValueError("heterogeneous RPM values must be positive")
    repeated = np.resize(np.asarray(values, dtype=np.float64), len(arms))
    order = _permutation(len(arms), permutation_index)
    assigned = repeated[order]
    return {str(arm): float(assigned[index]) for index, arm in enumerate(arms)}


def _stable_row_seed(seed: int, task_id: str) -> int:
    digest = sha256(f"{seed}:{task_id}".encode("utf-8", errors="replace")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


@dataclass(frozen=True)
class SyntheticLatencyTable:
    """Stable task/arm potential outcomes with policy-independent shocks.

    Stage 1 deliberately contains no endpoint-health process and no completion
    feedback.  A task/arm response is therefore fixed by the common random seed
    and does not depend on when a policy dispatches it.
    """

    task_ids: tuple[str, ...]
    arms: tuple[str, ...]
    base_seconds: np.ndarray
    random_multiplier: np.ndarray

    def __post_init__(self) -> None:
        shape = (len(self.task_ids), len(self.arms))
        if self.base_seconds.shape != shape or self.random_multiplier.shape != shape:
            raise ValueError(f"latency matrices must have shape {shape}")

    @property
    def static_response_seconds(self) -> np.ndarray:
        """Stable black-box response samples for the historical profile."""

        return self.base_seconds * self.random_multiplier

    def response_seconds(
        self,
        task_index: int,
        arm: str,
        dispatch_time: float | None = None,
    ) -> float:
        """Return a black-box response potential outcome.

        ``dispatch_time`` is accepted for call-site compatibility but is
        intentionally ignored in the feedback-free Stage 1 world.
        """

        try:
            arm_index = self.arms.index(arm)
        except ValueError as error:
            raise KeyError(f"unknown arm: {arm}") from error
        base = self.base_seconds[task_index, arm_index]
        shock = self.random_multiplier[task_index, arm_index]
        return float(base * shock)

    def potential_response_matrix(
        self, dispatch_times: np.ndarray | None = None
    ) -> np.ndarray:
        """Diagnostic full matrix; policies must not consume this at runtime."""

        if dispatch_times is not None:
            times = np.asarray(dispatch_times, dtype=np.float64)
            if times.shape != (len(self.task_ids), len(self.arms)):
                raise ValueError("dispatch_times must be [task, arm]")
        return self.static_response_seconds.copy()


def materialize_latency_table(
    frame: pd.DataFrame,
    arms: Sequence[str],
    output_tokens: np.ndarray,
    speeds: Mapping[str, EndpointSpeed],
    *,
    seed: int,
    noise: LatencyNoiseConfig = LatencyNoiseConfig(),
) -> SyntheticLatencyTable:
    """Materialise the potential outcome shocks shared by every policy.

    The base formula is ``a_i + .001*n_input + n_output/v_i``.  Each task's RNG
    is keyed by task ID, so producing a calibration table separately does not
    change the streaming table's shocks.
    """

    arm_tuple = tuple(str(arm) for arm in arms)
    task_ids = tuple(frame["task_id"].astype(str))
    output = np.asarray(output_tokens, dtype=np.float64)
    expected_shape = (len(frame), len(arm_tuple))
    if output.shape != expected_shape:
        raise ValueError(f"output_tokens must have shape {expected_shape}")
    missing_speeds = sorted(set(arm_tuple).difference(speeds))
    if missing_speeds:
        raise ValueError(f"missing speed profiles for {missing_speeds}")

    input_tokens = frame["input_tokens"].to_numpy(dtype=np.float64)
    base = np.empty(expected_shape, dtype=np.float64)
    for arm_index, arm in enumerate(arm_tuple):
        speed = speeds[arm]
        base[:, arm_index] = (
            speed.intercept_seconds
            + 0.001 * input_tokens
            + output[:, arm_index] / speed.output_tokens_per_second
        )

    multiplier = np.empty(expected_shape, dtype=np.float64)
    sigma = float(noise.lognormal_sigma)
    for row_index, task_id in enumerate(task_ids):
        rng = np.random.default_rng(_stable_row_seed(seed, task_id))
        values = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=len(arm_tuple))
        tail_mask = rng.random(len(arm_tuple)) < noise.tail_probability
        if np.any(tail_mask):
            tail = noise.tail_multiplier * (
                1.0 + rng.pareto(noise.tail_pareto_shape, int(tail_mask.sum()))
            )
            values[tail_mask] *= tail
        multiplier[row_index] = np.minimum(values, noise.maximum_multiplier)
    return SyntheticLatencyTable(
        task_ids=task_ids,
        arms=arm_tuple,
        base_seconds=base,
        random_multiplier=multiplier,
    )


def generate_poisson_arrivals(
    task_count: int,
    *,
    aggregate_refill_rate: float,
    quota_load: float,
    seed: int,
    start_seconds: float = 0.0,
) -> np.ndarray:
    """Generate arrivals with ``rho_q = Lambda / sum_i r_i``."""

    if task_count < 0 or aggregate_refill_rate <= 0 or quota_load <= 0:
        raise ValueError("task_count must be non-negative and rates/load positive")
    if task_count == 0:
        return np.empty(0, dtype=np.float64)
    rate = aggregate_refill_rate * quota_load
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(scale=1.0 / rate, size=max(0, task_count - 1))
    return np.concatenate(([float(start_seconds)], float(start_seconds) + np.cumsum(gaps)))


def generate_burst_arrivals(
    task_count: int,
    *,
    aggregate_refill_rate: float,
    loads: Sequence[float] = (0.5, 1.2, 0.5),
    task_fractions: Sequence[float] = (0.25, 0.5, 0.25),
    seed: int,
    start_seconds: float = 0.0,
) -> np.ndarray:
    """Generate a finite piecewise-Poisson burst, not a stationary overload."""

    if len(loads) != len(task_fractions) or not loads:
        raise ValueError("loads and task_fractions must have equal non-zero length")
    if aggregate_refill_rate <= 0 or any(load <= 0 for load in loads):
        raise ValueError("arrival rates must be positive")
    fractions = np.asarray(task_fractions, dtype=np.float64)
    if np.any(fractions < 0) or fractions.sum() <= 0:
        raise ValueError("task_fractions must be non-negative with positive sum")
    fractions /= fractions.sum()
    expected = fractions * task_count
    counts = np.floor(expected).astype(int)
    for index in np.argsort(-(expected - counts))[: task_count - int(counts.sum())]:
        counts[index] += 1

    rng = np.random.default_rng(seed)
    arrivals: list[float] = []
    clock = float(start_seconds)
    for segment, (load, count) in enumerate(zip(loads, counts, strict=True)):
        if count == 0:
            continue
        rate = aggregate_refill_rate * float(load)
        # The first task overall arrives at start_seconds; later segments have a
        # sampled gap so their boundaries cannot produce simultaneous artifacts.
        gap_count = int(count) if arrivals else max(0, int(count) - 1)
        gaps = rng.exponential(scale=1.0 / rate, size=gap_count)
        if not arrivals:
            arrivals.append(clock)
            for gap in gaps:
                clock += float(gap)
                arrivals.append(clock)
        else:
            for gap in gaps:
                clock += float(gap)
                arrivals.append(clock)
    return np.asarray(arrivals[:task_count], dtype=np.float64)
