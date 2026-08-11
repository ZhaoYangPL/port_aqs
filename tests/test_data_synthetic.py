from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from port_aqs.data import SplitSpec, TfidfKNNProfiles, stratified_split
from port_aqs.synthetic import (
    LatencyNoiseConfig,
    generate_burst_arrivals,
    generate_poisson_arrivals,
    make_speed_profiles,
    materialize_latency_table,
)


class DataTests(unittest.TestCase):
    def test_stratified_split_is_disjoint_and_exact(self) -> None:
        frame = pd.DataFrame(
            {
                "task_id": [f"q{index}" for index in range(100)],
                "prompt_hash": [f"h{index}" for index in range(100)],
                "eval_name": [f"family_{index % 5}" for index in range(100)],
            }
        )
        split = stratified_split(frame, SplitSpec(60, 20, 20, seed=7))
        self.assertEqual(split.actual_sizes, {"historical": 60, "calibration": 20, "streaming": 20})
        sets = [set(part.prompt_hash) for part in (split.historical, split.calibration, split.streaming)]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])

    def test_tfidf_knn_prediction_shapes_and_ranges(self) -> None:
        prompts = ["python sort list", "java sort array", "translate chinese"]
        quality = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
        cost = np.asarray([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]])
        latency = np.asarray([[3.0, 1.0], [4.0, 1.5], [2.0, 2.5]])
        output_tokens = np.asarray([[30.0, 10.0], [40.0, 15.0], [20.0, 25.0]])
        model = TfidfKNNProfiles(neighbors=2, max_features=64).fit(
            prompts, quality, cost, latency, output_tokens
        )
        prediction = model.predict(["python list sorting"])
        self.assertEqual(prediction.quality.shape, (1, 2))
        self.assertEqual(prediction.cost.shape, (1, 2))
        self.assertEqual(prediction.output_tokens.shape, (1, 2))
        self.assertEqual(prediction.latency_samples.shape, (1, 2, 2))
        self.assertTrue(np.all((prediction.quality >= 0) & (prediction.quality <= 1)))
        self.assertTrue(np.all(prediction.output_tokens > 0))


class SyntheticTests(unittest.TestCase):
    def test_latency_table_is_deterministic_and_positive(self) -> None:
        frame = pd.DataFrame(
            {"task_id": ["q0", "q1"], "input_tokens": [10, 20]}
        )
        arms = ("arm_00", "arm_01")
        output = np.asarray([[20, 30], [40, 50]], dtype=float)
        speeds = make_speed_profiles(arms)
        noise = LatencyNoiseConfig(tail_probability=0.0)
        first = materialize_latency_table(frame, arms, output, speeds, seed=9, noise=noise)
        second = materialize_latency_table(frame, arms, output, speeds, seed=9, noise=noise)
        np.testing.assert_allclose(first.static_response_seconds, second.static_response_seconds)
        self.assertTrue(np.all(first.static_response_seconds > 0))
        # Stage 1 has no health process: dispatch time cannot change the outcome.
        self.assertEqual(first.response_seconds(0, "arm_00", 0.0), first.response_seconds(0, "arm_00", 999.0))

    def test_arrival_generators_are_monotone_and_sized(self) -> None:
        poisson = generate_poisson_arrivals(
            50, aggregate_refill_rate=5.0, quota_load=0.7, seed=1
        )
        burst = generate_burst_arrivals(50, aggregate_refill_rate=5.0, seed=1)
        self.assertEqual(len(poisson), 50)
        self.assertEqual(len(burst), 50)
        self.assertTrue(np.all(np.diff(poisson) >= 0))
        self.assertTrue(np.all(np.diff(burst) >= 0))


if __name__ == "__main__":
    unittest.main()
