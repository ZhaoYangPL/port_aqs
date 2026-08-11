"""RouterBench loading, anonymisation, leakage-safe splits and TF-IDF profiles.

The source pickle contains quality, response and monetary-cost columns keyed by
historical model names.  This module deliberately converts those names to
anonymous arms before they enter the simulator.  Nothing in an experiment
result should be read as a measurement of the similarly named API today.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence
import math

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


META_COLUMNS = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}
QUALITY_PREFIX = "quality__"
COST_PREFIX = "cost__"
OUTPUT_PREFIX = "output_tokens__"


def _text_hash(value: object) -> str:
    return sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def _rough_token_count(value: object) -> int:
    """Return a dependency-free, deterministic token-count proxy.

    The synthetic experiment is not claiming tokenizer-accurate billing.  The
    approximation is used only as a covariate in the latency generator.
    """

    text = "" if value is None else str(value)
    return max(1, int(math.ceil(len(text) / 4.0)))


def detect_model_columns(frame: pd.DataFrame) -> list[str]:
    """Detect quality columns that have matching cost and response columns."""

    detected: list[str] = []
    columns = set(frame.columns)
    for column in frame.columns:
        if column in META_COLUMNS or "|" in str(column):
            continue
        if f"{column}|total_cost" in columns and f"{column}|model_response" in columns:
            detected.append(str(column))
    if not detected:
        raise ValueError(
            "No model columns found; expected <model>, <model>|total_cost and "
            "<model>|model_response columns."
        )
    return detected


@dataclass(frozen=True)
class RouterBenchData:
    """An anonymised, compact view of RouterBench."""

    frame: pd.DataFrame
    arms: tuple[str, ...]
    source_model_by_arm: Mapping[str, str]
    source_path: Path
    source_rows: int
    duplicate_prompts_removed: int

    def quality_columns(self) -> list[str]:
        return [f"{QUALITY_PREFIX}{arm}" for arm in self.arms]

    def cost_columns(self) -> list[str]:
        return [f"{COST_PREFIX}{arm}" for arm in self.arms]

    def output_columns(self) -> list[str]:
        return [f"{OUTPUT_PREFIX}{arm}" for arm in self.arms]

    def quality_matrix(self, subset: pd.DataFrame | None = None) -> np.ndarray:
        data = self.frame if subset is None else subset
        return data[self.quality_columns()].to_numpy(dtype=np.float64, copy=True)

    def cost_matrix(self, subset: pd.DataFrame | None = None) -> np.ndarray:
        data = self.frame if subset is None else subset
        return data[self.cost_columns()].to_numpy(dtype=np.float64, copy=True)

    def output_token_matrix(self, subset: pd.DataFrame | None = None) -> np.ndarray:
        data = self.frame if subset is None else subset
        return data[self.output_columns()].to_numpy(dtype=np.float64, copy=True)


def load_routerbench(path: str | Path) -> RouterBenchData:
    """Load the local pickle and immediately anonymise all model-facing data."""

    source_path = Path(path).expanduser().resolve()
    raw = pd.read_pickle(source_path)
    required = {"prompt", "eval_name"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"RouterBench data is missing required columns: {missing}")

    source_models = detect_model_columns(raw)
    arms = tuple(f"arm_{index:02d}" for index in range(len(source_models)))
    mapping = dict(zip(arms, source_models, strict=True))

    compact = pd.DataFrame(index=raw.index)
    compact["task_id"] = raw.get("sample_id", raw.index.astype(str)).astype(str)
    compact["prompt"] = raw["prompt"].fillna("").astype(str)
    compact["eval_name"] = raw["eval_name"].fillna("unknown").astype(str)
    compact["prompt_hash"] = compact["prompt"].map(_text_hash)
    compact["input_tokens"] = compact["prompt"].map(_rough_token_count).astype(np.int32)

    for arm, source_model in mapping.items():
        quality = pd.to_numeric(raw[source_model], errors="coerce")
        cost = pd.to_numeric(raw[f"{source_model}|total_cost"], errors="coerce")
        compact[f"{QUALITY_PREFIX}{arm}"] = quality.fillna(0.0).astype(np.float32)
        positive_cost = cost[cost >= 0]
        fallback_cost = float(positive_cost.median()) if not positive_cost.empty else 0.0
        compact[f"{COST_PREFIX}{arm}"] = cost.fillna(fallback_cost).clip(lower=0).astype(np.float64)
        compact[f"{OUTPUT_PREFIX}{arm}"] = (
            raw[f"{source_model}|model_response"].map(_rough_token_count).astype(np.int32)
        )

    # A stable row order makes the split invariant to the pickle's current index.
    source_rows = len(compact)
    compact = compact.sort_values(["prompt_hash", "task_id"], kind="stable")
    compact = compact.drop_duplicates("prompt_hash", keep="first").reset_index(drop=True)
    return RouterBenchData(
        compact,
        arms,
        mapping,
        source_path,
        source_rows,
        source_rows - len(compact),
    )


@dataclass(frozen=True)
class SplitSpec:
    historical: int = 26_481
    calibration: int = 500
    streaming: int = 9_500
    seed: int = 2025


@dataclass(frozen=True)
class DatasetSplits:
    historical: pd.DataFrame
    calibration: pd.DataFrame
    streaming: pd.DataFrame
    requested: SplitSpec
    deduplicated_size: int

    @property
    def actual_sizes(self) -> dict[str, int]:
        return {
            "historical": len(self.historical),
            "calibration": len(self.calibration),
            "streaming": len(self.streaming),
        }


def _apportion_strata(
    counts: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Integer proportional allocation with exact row and column sums."""

    total = int(counts.sum())
    if int(targets.sum()) != total:
        raise ValueError("targets must include every row (use an unused target if needed)")
    expected = counts[:, None] * (targets[None, :] / max(total, 1))
    allocation = np.floor(expected).astype(np.int64)
    row_remaining = counts.astype(np.int64) - allocation.sum(axis=1)
    column_remaining = targets.astype(np.int64) - allocation.sum(axis=0)

    # Only a few rounding units remain per stratum.  Repeatedly allocate the
    # cell furthest below its proportional ideal while respecting exact totals.
    while int(row_remaining.sum()) > 0:
        best: tuple[float, int, int] | None = None
        for row in np.flatnonzero(row_remaining > 0):
            for column in np.flatnonzero(column_remaining > 0):
                score = float(expected[row, column] - allocation[row, column])
                candidate = (score, -int(row), -int(column))
                if best is None or candidate > best:
                    best = candidate
        if best is None:  # pragma: no cover - guards against arithmetic bugs
            raise RuntimeError("Unable to complete stratified allocation")
        _, neg_row, neg_column = best
        row, column = -neg_row, -neg_column
        allocation[row, column] += 1
        row_remaining[row] -= 1
        column_remaining[column] -= 1

    if np.any(allocation.sum(axis=1) != counts) or np.any(allocation.sum(axis=0) != targets):
        raise RuntimeError("Internal stratification invariant failed")
    return allocation


def stratified_split(frame: pd.DataFrame, spec: SplitSpec = SplitSpec()) -> DatasetSplits:
    """Split by ``eval_name`` after prompt-hash deduplication.

    The public pickle has 16 duplicate prompts.  The defaults therefore use
    all 36,481 unique prompts: 26,481 historical, 500 calibration and 9,500
    streaming.  No duplicated prompt is ever reintroduced to satisfy a
    requested count.
    """

    if any(value < 0 for value in (spec.historical, spec.calibration, spec.streaming)):
        raise ValueError("split sizes must be non-negative")
    if "prompt_hash" not in frame or "eval_name" not in frame:
        raise ValueError("frame must contain prompt_hash and eval_name")
    deduplicated = (
        frame.sort_values(["prompt_hash", "task_id"], kind="stable")
        .drop_duplicates("prompt_hash", keep="first")
        .reset_index(drop=True)
    )
    size = len(deduplicated)
    streaming_size = min(spec.streaming, size)
    calibration_size = min(spec.calibration, size - streaming_size)
    historical_size = min(spec.historical, size - streaming_size - calibration_size)
    unused_size = size - historical_size - calibration_size - streaming_size
    targets = np.asarray(
        [historical_size, calibration_size, streaming_size, unused_size], dtype=np.int64
    )

    groups = list(deduplicated.groupby("eval_name", sort=True, observed=True))
    counts = np.asarray([len(group) for _, group in groups], dtype=np.int64)
    allocation = _apportion_strata(counts, targets)
    buckets: list[list[pd.DataFrame]] = [[], [], [], []]
    for group_index, (name, group) in enumerate(groups):
        group_seed = int(_text_hash(f"{spec.seed}:{name}")[:16], 16) % (2**32)
        order = np.random.default_rng(group_seed).permutation(len(group))
        shuffled = group.iloc[order]
        cursor = 0
        for split_index, amount in enumerate(allocation[group_index]):
            next_cursor = cursor + int(amount)
            if amount:
                buckets[split_index].append(shuffled.iloc[cursor:next_cursor])
            cursor = next_cursor

    def combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return deduplicated.iloc[:0].copy()
        result = pd.concat(parts, axis=0)
        # Stable pseudo-random interleaving prevents eval_name blocks in streams.
        keys = result["prompt_hash"].map(lambda value: _text_hash(f"{spec.seed}:{value}"))
        return result.assign(_split_order=keys).sort_values("_split_order").drop(
            columns="_split_order"
        ).reset_index(drop=True)

    return DatasetSplits(
        historical=combine(buckets[0]),
        calibration=combine(buckets[1]),
        streaming=combine(buckets[2]),
        requested=spec,
        deduplicated_size=size,
    )


@dataclass(frozen=True)
class KNNPrediction:
    quality: np.ndarray
    cost: np.ndarray
    output_tokens: np.ndarray
    latency_samples: np.ndarray
    neighbor_weights: np.ndarray

    @property
    def latency_mean(self) -> np.ndarray:
        return np.einsum("qka,qk->qa", self.latency_samples, self.neighbor_weights)

    def latency_quantile(self, probability: float) -> np.ndarray:
        if not 0 <= probability <= 1:
            raise ValueError("probability must lie in [0, 1]")
        order = np.argsort(self.latency_samples, axis=1)
        sorted_values = np.take_along_axis(self.latency_samples, order, axis=1)
        expanded_weights = np.broadcast_to(
            self.neighbor_weights[:, :, None], self.latency_samples.shape
        )
        sorted_weights = np.take_along_axis(expanded_weights, order, axis=1)
        cumulative = np.cumsum(sorted_weights, axis=1)
        positions = np.argmax(cumulative >= probability, axis=1)
        return np.take_along_axis(sorted_values, positions[:, None, :], axis=1)[:, 0, :]

    def latency_cdf(self, thresholds: np.ndarray) -> np.ndarray:
        """Evaluate per-query/per-arm empirical CDFs at matching thresholds."""

        values = np.asarray(thresholds, dtype=np.float64)
        expected_shape = (self.latency_samples.shape[0], self.latency_samples.shape[2])
        if values.shape != expected_shape:
            raise ValueError(f"thresholds must have shape {expected_shape}, got {values.shape}")
        hits = self.latency_samples <= values[:, None, :]
        return np.einsum("qka,qk->qa", hits, self.neighbor_weights)


class TfidfKNNProfiles:
    """Training-free quality/cost/output-demand/latency profiles."""

    def __init__(
        self,
        *,
        neighbors: int = 16,
        max_features: int = 4_096,
        query_batch_size: int = 256,
        n_jobs: int = 1,
    ) -> None:
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        self.neighbors = neighbors
        self.query_batch_size = query_batch_size
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.index = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=n_jobs)
        self._quality: np.ndarray | None = None
        self._cost: np.ndarray | None = None
        self._output_tokens: np.ndarray | None = None
        self._latency: np.ndarray | None = None

    def fit(
        self,
        prompts: Sequence[str],
        quality: np.ndarray,
        cost: np.ndarray,
        static_latency: np.ndarray,
        output_tokens: np.ndarray | None = None,
    ) -> "TfidfKNNProfiles":
        documents = [str(prompt) for prompt in prompts]
        if not documents:
            raise ValueError("historical prompts may not be empty")
        matrices = [np.asarray(value, dtype=np.float64) for value in (quality, cost, static_latency)]
        if output_tokens is None:
            # Compatibility for callers that only need quality/cost/latency.
            # The experiment runner always supplies measured historical output
            # lengths, because TPM admission must use a prediction made before
            # dispatch rather than the streaming task's realised output.
            output_matrix = np.zeros_like(matrices[0])
        else:
            output_matrix = np.asarray(output_tokens, dtype=np.float64)
            matrices.append(output_matrix)
        if any(matrix.ndim != 2 or matrix.shape[0] != len(documents) for matrix in matrices):
            raise ValueError("all label matrices must be [historical_task, arm]")
        if len({matrix.shape for matrix in matrices}) != 1:
            raise ValueError(
                "quality, cost, static_latency and output_tokens shapes must match"
            )
        vectors = self.vectorizer.fit_transform(documents)
        self.index.fit(vectors)
        self._quality, self._cost, self._latency = matrices[:3]
        self._output_tokens = output_matrix
        return self

    def predict(self, prompts: Sequence[str]) -> KNNPrediction:
        if (
            self._quality is None
            or self._cost is None
            or self._output_tokens is None
            or self._latency is None
        ):
            raise RuntimeError("fit must be called before predict")
        documents = [str(prompt) for prompt in prompts]
        if not documents:
            arm_count = self._quality.shape[1]
            return KNNPrediction(
                np.empty((0, arm_count)),
                np.empty((0, arm_count)),
                np.empty((0, arm_count)),
                np.empty((0, 0, arm_count)),
                np.empty((0, 0)),
            )
        vectors = self.vectorizer.transform(documents)
        neighbor_count = min(self.neighbors, self._quality.shape[0])
        all_distances: list[np.ndarray] = []
        all_indices: list[np.ndarray] = []
        for start in range(0, len(documents), self.query_batch_size):
            distances, indices = self.index.kneighbors(
                vectors[start : start + self.query_batch_size],
                n_neighbors=neighbor_count,
                return_distance=True,
            )
            all_distances.append(distances)
            all_indices.append(indices)
        distance = np.concatenate(all_distances, axis=0)
        index = np.concatenate(all_indices, axis=0)
        # Exact matches dominate but never create infinite/nan weights.
        raw_weights = 1.0 / np.maximum(distance, 1e-6)
        weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)
        quality_neighbors = self._quality[index]
        cost_neighbors = self._cost[index]
        output_neighbors = self._output_tokens[index]
        latency_neighbors = self._latency[index]
        quality_prediction = np.einsum("qka,qk->qa", quality_neighbors, weights)
        cost_prediction = np.einsum("qka,qk->qa", cost_neighbors, weights)
        output_prediction = np.einsum("qka,qk->qa", output_neighbors, weights)
        return KNNPrediction(
            quality=quality_prediction,
            cost=cost_prediction,
            output_tokens=np.maximum(output_prediction, 0.0),
            latency_samples=latency_neighbors,
            neighbor_weights=weights,
        )


def calibration_cost_scale(cost_matrix: np.ndarray, quantile: float = 0.95) -> float:
    """A frozen robust scale for the monetary-cost term."""

    values = np.asarray(cost_matrix, dtype=np.float64)
    finite = values[np.isfinite(values) & (values >= 0)]
    if not finite.size:
        return 1.0
    scale = float(np.quantile(finite, quantile))
    return scale if scale > 0 else 1.0
