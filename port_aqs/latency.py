"""Frozen black-box API response-latency profiles for PORT-AQS Stage 1."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


class LatencyEstimator:
    """Evaluate a static task-conditioned weighted empirical distribution.

    Samples describe full external-API response time.  They are not server
    service times, and Stage 1 never updates them from streaming completions.
    A caller may supply kNN samples and weights for each task; otherwise the
    frozen endpoint-wide profile passed at construction is used with equal
    mass.
    """

    def __init__(self, static_samples: Mapping[str, Sequence[float]]) -> None:
        if not static_samples:
            raise ValueError("at least one static latency profile is required")
        self._static = {
            model_id: self._validated_samples(samples)
            for model_id, samples in static_samples.items()
        }
        if any(not model_id for model_id in self._static):
            raise ValueError("model_id must be non-empty")

    @staticmethod
    def _validated_samples(samples: Sequence[float]) -> np.ndarray:
        values = np.asarray(tuple(samples), dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("latency samples must be a non-empty one-dimensional sequence")
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("latency samples must contain positive finite seconds")
        values = np.array(values, copy=True)
        values.setflags(write=False)
        return values

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._static))

    def _base_samples(
        self, model_id: str, static_samples: Sequence[float] | None
    ) -> np.ndarray:
        if static_samples is not None and len(static_samples) > 0:
            return self._validated_samples(static_samples)
        try:
            return self._static[model_id]
        except KeyError as error:
            raise KeyError(f"no static latency profile for {model_id!r}") from error

    @staticmethod
    def _validated_weights(
        sample_weights: Sequence[float] | None,
        sample_count: int,
    ) -> np.ndarray:
        if sample_weights is None or len(sample_weights) == 0:
            weights = np.full(sample_count, 1.0 / sample_count, dtype=np.float64)
            weights.setflags(write=False)
            return weights
        weights = np.asarray(tuple(sample_weights), dtype=np.float64)
        if weights.ndim != 1 or weights.size != sample_count:
            raise ValueError("sample_weights must have one value per latency sample")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("sample_weights must contain finite non-negative values")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("sample_weights must have positive total mass")
        weights = np.asarray(weights / total, dtype=np.float64)
        weights.setflags(write=False)
        return weights

    def _profile(
        self,
        model_id: str,
        static_samples: Sequence[float] | None,
        sample_weights: Sequence[float] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        samples = self._base_samples(model_id, static_samples)
        return samples, self._validated_weights(sample_weights, samples.size)

    @staticmethod
    def _weighted_quantile(
        samples: np.ndarray,
        weights: np.ndarray,
        quantile: float,
    ) -> float:
        order = np.argsort(samples, kind="stable")
        sorted_samples = samples[order]
        cumulative = np.cumsum(weights[order])
        cumulative[-1] = 1.0
        index = int(np.searchsorted(cumulative, quantile, side="left"))
        return float(sorted_samples[min(index, sorted_samples.size - 1)])

    def predict_samples(
        self,
        model_id: str,
        *,
        static_samples: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Return a copy of the frozen response-time samples."""

        return np.array(self._base_samples(model_id, static_samples), copy=True)

    def predict_cdf(
        self,
        model_id: str,
        threshold: float,
        *,
        static_samples: Sequence[float] | None = None,
        sample_weights: Sequence[float] | None = None,
    ) -> float:
        """Evaluate the frozen empirical response-time CDF."""

        if math.isnan(threshold):
            raise ValueError("threshold cannot be NaN")
        if threshold <= 0:
            return 0.0
        if math.isinf(threshold):
            return 1.0
        samples, weights = self._profile(model_id, static_samples, sample_weights)
        value = float(weights[samples <= threshold].sum())
        return min(1.0, max(0.0, value))

    def predict_quantile(
        self,
        model_id: str,
        quantile: float,
        *,
        static_samples: Sequence[float] | None = None,
        sample_weights: Sequence[float] | None = None,
    ) -> float:
        if not math.isfinite(quantile) or not 0 <= quantile <= 1:
            raise ValueError("quantile must lie in [0, 1]")
        samples, weights = self._profile(model_id, static_samples, sample_weights)
        return self._weighted_quantile(samples, weights, quantile)

    def static_median(
        self,
        model_id: str,
        static_samples: Sequence[float] | None = None,
        sample_weights: Sequence[float] | None = None,
    ) -> float:
        return self.predict_quantile(
            model_id,
            0.5,
            static_samples=static_samples,
            sample_weights=sample_weights,
        )

    def slo_violation_probability(
        self,
        model_id: str,
        deadline: float,
        admission_wait: float,
        *,
        static_samples: Sequence[float] | None = None,
        sample_weights: Sequence[float] | None = None,
    ) -> float:
        """Predict ``P(admission_wait + API response > deadline)``."""

        if not math.isfinite(deadline) or deadline <= 0:
            raise ValueError("deadline must be positive and finite")
        if not math.isfinite(admission_wait) or admission_wait < 0:
            raise ValueError("admission_wait must be non-negative and finite")
        if deadline <= admission_wait:
            return 1.0
        cdf = self.predict_cdf(
            model_id,
            deadline - admission_wait,
            static_samples=static_samples,
            sample_weights=sample_weights,
        )
        return min(1.0, max(0.0, 1.0 - cdf))
