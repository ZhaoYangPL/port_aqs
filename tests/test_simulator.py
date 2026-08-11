from __future__ import annotations

import json
import math
import unittest

import numpy as np
import pandas as pd

from port_aqs.data import KNNPrediction
from port_aqs.metrics import compute_metrics
from port_aqs.simulator import (
    QuotaProxyPrices,
    SimulationConfig,
    quota_proxy_gamma,
    simulate_policy,
)
from port_aqs.synthetic import SyntheticLatencyTable
from port_aqs.types import EndpointSpec


def _profiles(
    task_count: int,
    arm_count: int,
    *,
    latency: float = 1.0,
    output_tokens: float = 8.0,
) -> KNNPrediction:
    return KNNPrediction(
        quality=np.full((task_count, arm_count), 0.5),
        cost=np.full((task_count, arm_count), 0.1),
        output_tokens=np.full((task_count, arm_count), output_tokens),
        latency_samples=np.full((task_count, 3, arm_count), latency),
        neighbor_weights=np.full((task_count, 3), 1.0 / 3.0),
    )


def _endpoint(
    arm: str = "arm",
    *,
    rpm: float = 60.0,
    rpm_capacity: float = 1.0,
    tpm: float = 60.0,
    tpm_capacity: float = 10.0,
) -> EndpointSpec:
    return EndpointSpec(arm, rpm, rpm_capacity, tpm, tpm_capacity)


class SimulatorTests(unittest.TestCase):
    def test_joint_quota_wait_uses_predicted_not_actual_tokens_and_drains(self) -> None:
        frame = pd.DataFrame(
            {
                "task_id": ["q0", "q1"],
                "eval_name": ["x", "x"],
                "input_tokens": [2, 2],
            }
        )
        world = SyntheticLatencyTable(
            task_ids=("q0", "q1"),
            arms=("arm",),
            base_seconds=np.asarray([[0.5], [0.5]]),
            random_multiplier=np.ones((2, 1)),
        )
        trace = simulate_policy(
            policy="quota_risk",
            frame=frame,
            arms=("arm",),
            arrivals=[0.0, 0.0],
            actual_quality=np.asarray([[1.0], [0.0]]),
            actual_cost=np.asarray([[0.2], [0.3]]),
            actual_output_tokens=np.asarray([[7.0], [50.0]]),
            profiles=_profiles(2, 1, latency=0.5, output_tokens=8.0),
            world=world,
            endpoints=(_endpoint(),),
            cost_scale=1.0,
            config=SimulationConfig(deadline_seconds=20.0, lambda_penalty=1.0),
        )
        self.assertEqual(trace["status"].tolist(), ["completed", "completed"])
        np.testing.assert_allclose(trace["dispatch_time"], [0.0, 10.0])
        np.testing.assert_allclose(trace["rpm_wait"], [0.0, 1.0])
        np.testing.assert_allclose(trace["tpm_wait"], [0.0, 10.0])
        np.testing.assert_allclose(trace["admission_wait"], [0.0, 10.0])
        np.testing.assert_allclose(trace["predicted_token_demand"], [10.0, 10.0])
        np.testing.assert_allclose(trace["reserved_token_demand"], [10.0, 10.0])
        np.testing.assert_allclose(trace["actual_token_demand"], [9.0, 52.0])
        # The large actual second output does not retroactively change dispatch.
        self.assertEqual(float(trace.iloc[1]["token_reservation_error"]), 42.0)
        self.assertTrue(bool(trace.iloc[1]["tpm_binding"]))
        self.assertEqual(len(json.loads(trace.iloc[0]["candidate_diagnostics"])), 1)
        self.assertEqual(
            json.loads(trace.iloc[0]["tpm_quota_snapshot"])["tpm_refill_rate"],
            1.0,
        )
        metrics = compute_metrics(trace)
        self.assertEqual(metrics["completed"], 2)
        self.assertAlmostEqual(float(metrics["quota_wait_mean"]), 5.0)

    def test_rpm_only_ablation_has_no_tpm_wait(self) -> None:
        frame = pd.DataFrame(
            {
                "task_id": ["q0", "q1", "q2"],
                "eval_name": ["x", "x", "x"],
                "input_tokens": [2, 2, 2],
            }
        )
        world = SyntheticLatencyTable(
            task_ids=("q0", "q1", "q2"),
            arms=("arm",),
            base_seconds=np.ones((3, 1)),
            random_multiplier=np.ones((3, 1)),
        )
        trace = simulate_policy(
            policy="admission",  # compatibility alias; trace is canonical.
            frame=frame,
            arms=("arm",),
            arrivals=[0.0, 0.0, 0.0],
            actual_quality=np.ones((3, 1)),
            actual_cost=np.zeros((3, 1)),
            actual_output_tokens=np.full((3, 1), 8.0),
            profiles=_profiles(3, 1),
            world=world,
            endpoints=(
                _endpoint(tpm=math.inf, tpm_capacity=math.inf),
            ),
            cost_scale=1.0,
        )
        self.assertTrue((trace["policy"] == "rpm_aware").all())
        np.testing.assert_allclose(trace["admission_wait"], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(trace["rpm_wait"], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(trace["tpm_wait"], [0.0, 0.0, 0.0])
        self.assertFalse(trace["tpm_binding"].any())
        snapshot = json.loads(trace.iloc[0]["tpm_quota_snapshot"])
        self.assertIsNone(snapshot["tpm_capacity"])
        self.assertIsNone(snapshot["tpm_refill_rate"])

    def test_tpm_only_ablation_has_no_rpm_wait(self) -> None:
        frame = pd.DataFrame(
            {
                "task_id": ["q0", "q1", "q2"],
                "eval_name": ["x", "x", "x"],
                "input_tokens": [2, 2, 2],
            }
        )
        world = SyntheticLatencyTable(
            task_ids=("q0", "q1", "q2"),
            arms=("arm",),
            base_seconds=np.ones((3, 1)),
            random_multiplier=np.ones((3, 1)),
        )
        trace = simulate_policy(
            policy="quota_risk",
            frame=frame,
            arms=("arm",),
            arrivals=[0.0, 0.0, 0.0],
            actual_quality=np.ones((3, 1)),
            actual_cost=np.zeros((3, 1)),
            actual_output_tokens=np.full((3, 1), 8.0),
            profiles=_profiles(3, 1),
            world=world,
            endpoints=(
                _endpoint(rpm=math.inf, rpm_capacity=math.inf),
            ),
            cost_scale=1.0,
        )
        np.testing.assert_allclose(trace["admission_wait"], [0.0, 10.0, 20.0])
        np.testing.assert_allclose(trace["tpm_wait"], [0.0, 10.0, 20.0])
        np.testing.assert_allclose(trace["rpm_wait"], [0.0, 0.0, 0.0])
        self.assertFalse(trace["rpm_binding"].any())
        snapshot = json.loads(trace.iloc[0]["rpm_quota_snapshot"])
        self.assertIsNone(snapshot["rpm_capacity"])
        self.assertIsNone(snapshot["rpm_refill_rate"])

    def test_no_completion_feedback_or_health_fields(self) -> None:
        frame = pd.DataFrame(
            {
                "task_id": ["q0", "q1"],
                "eval_name": ["x", "x"],
                "input_tokens": [2, 2],
            }
        )
        world = SyntheticLatencyTable(
            task_ids=("q0", "q1"),
            arms=("arm",),
            base_seconds=np.ones((2, 1)),
            random_multiplier=np.asarray([[100.0], [1.0]]),
        )
        trace = simulate_policy(
            policy="static_risk",
            frame=frame,
            arms=("arm",),
            arrivals=[0.0, 200.0],
            actual_quality=np.ones((2, 1)),
            actual_cost=np.zeros((2, 1)),
            actual_output_tokens=np.full((2, 1), 8.0),
            profiles=_profiles(2, 1),
            world=world,
            endpoints=(_endpoint(rpm=math.inf, rpm_capacity=math.inf, tpm=math.inf, tpm_capacity=math.inf),),
            cost_scale=1.0,
            config=SimulationConfig(deadline_seconds=5.0, lambda_penalty=1.0),
        )
        self.assertNotIn("health_estimate", trace)
        self.assertNotIn("actual_health_multiplier", trace)
        np.testing.assert_allclose(trace["predicted_violation"], [0.0, 0.0])

    def test_routes_are_invariant_to_every_streaming_outcome_matrix(self) -> None:
        frame = pd.DataFrame(
            {
                "task_id": ["q0", "q1", "q2"],
                "eval_name": ["x", "x", "x"],
                "input_tokens": [2, 2, 2],
            }
        )
        arms = ("arm_a", "arm_b")
        profiles = KNNPrediction(
            quality=np.asarray([[0.8, 0.7], [0.6, 0.9], [0.8, 0.7]]),
            cost=np.zeros((3, 2)),
            output_tokens=np.asarray([[8.0, 4.0], [8.0, 4.0], [8.0, 4.0]]),
            latency_samples=np.ones((3, 2, 2)),
            neighbor_weights=np.full((3, 2), 0.5),
        )
        endpoints = tuple(
            _endpoint(arm, rpm=60.0, rpm_capacity=1.0, tpm=600.0, tpm_capacity=20.0)
            for arm in arms
        )
        common = dict(
            policy="quota_risk",
            frame=frame,
            arms=arms,
            arrivals=[0.0, 5.0, 10.0],
            profiles=profiles,
            endpoints=endpoints,
            cost_scale=1.0,
            config=SimulationConfig(deadline_seconds=5.0, lambda_penalty=1.0),
        )
        first = simulate_policy(
            actual_quality=np.zeros((3, 2)),
            actual_cost=np.zeros((3, 2)),
            actual_output_tokens=np.ones((3, 2)),
            world=SyntheticLatencyTable(
                tuple(frame.task_id), arms, np.ones((3, 2)), np.ones((3, 2))
            ),
            **common,
        )
        second = simulate_policy(
            actual_quality=np.full((3, 2), -999.0),
            actual_cost=np.full((3, 2), 1e9),
            actual_output_tokens=np.full((3, 2), 1e6),
            world=SyntheticLatencyTable(
                tuple(frame.task_id), arms, np.full((3, 2), 1000.0), np.ones((3, 2))
            ),
            **common,
        )
        for column in (
            "model_id",
            "dispatch_time",
            "admission_wait",
            "reserved_token_demand",
            "predicted_violation",
        ):
            with self.subTest(column=column):
                np.testing.assert_array_equal(first[column], second[column])
        self.assertFalse(np.array_equal(first["quality"], second["quality"]))
        self.assertFalse(np.array_equal(first["api_latency"], second["api_latency"]))

    def test_weighted_static_risk_and_min_risk(self) -> None:
        frame = pd.DataFrame(
            {"task_id": ["q0"], "eval_name": ["x"], "input_tokens": [1]}
        )
        arms = ("weighted_fast", "weighted_slow")
        profiles = KNNPrediction(
            quality=np.full((1, 2), 0.5),
            cost=np.zeros((1, 2)),
            output_tokens=np.ones((1, 2)),
            latency_samples=np.asarray([[[1.0, 10.0], [10.0, 1.0]]]),
            neighbor_weights=np.asarray([[0.9, 0.1]]),
        )
        world = SyntheticLatencyTable(
            task_ids=("q0",),
            arms=arms,
            base_seconds=np.ones((1, 2)),
            random_multiplier=np.ones((1, 2)),
        )
        common = dict(
            frame=frame,
            arms=arms,
            arrivals=[0.0],
            actual_quality=np.ones((1, 2)),
            actual_cost=np.zeros((1, 2)),
            actual_output_tokens=np.ones((1, 2)),
            profiles=profiles,
            world=world,
            endpoints=tuple(
                _endpoint(
                    arm,
                    rpm=math.inf,
                    rpm_capacity=math.inf,
                    tpm=math.inf,
                    tpm_capacity=math.inf,
                )
                for arm in arms
            ),
            cost_scale=1.0,
            config=SimulationConfig(deadline_seconds=5.0, lambda_penalty=1.0),
        )
        for policy in ("static_risk", "min_risk"):
            with self.subTest(policy=policy):
                trace = simulate_policy(policy=policy, **common)
                self.assertEqual(trace.iloc[0]["model_id"], "weighted_fast")
                diagnostics = {
                    item["model_id"]: item
                    for item in json.loads(trace.iloc[0]["candidate_diagnostics"])
                }
                self.assertAlmostEqual(
                    diagnostics["weighted_fast"]["predicted_violation"], 0.1
                )
                self.assertAlmostEqual(
                    diagnostics["weighted_slow"]["predicted_violation"], 0.9
                )

    def test_lambda_zero_strictly_matches_quality_cost_tie_break(self) -> None:
        frame = pd.DataFrame(
            {"task_id": ["q0"], "eval_name": ["x"], "input_tokens": [1]}
        )
        arms = ("arm_a", "arm_b")
        profiles = KNNPrediction(
            quality=np.ones((1, 2)),
            cost=np.zeros((1, 2)),
            output_tokens=np.ones((1, 2)),
            # Both violate a 0.5s deadline, but arm_b has smaller p95.
            latency_samples=np.asarray([[[10.0, 1.0], [10.0, 1.0]]]),
            neighbor_weights=np.asarray([[0.5, 0.5]]),
        )
        world = SyntheticLatencyTable(
            task_ids=("q0",),
            arms=arms,
            base_seconds=np.ones((1, 2)),
            random_multiplier=np.ones((1, 2)),
        )
        common = dict(
            frame=frame,
            arms=arms,
            arrivals=[0.0],
            actual_quality=np.ones((1, 2)),
            actual_cost=np.zeros((1, 2)),
            actual_output_tokens=np.ones((1, 2)),
            profiles=profiles,
            world=world,
            endpoints=tuple(
                _endpoint(
                    arm,
                    rpm=math.inf,
                    rpm_capacity=math.inf,
                    tpm=math.inf,
                    tpm_capacity=math.inf,
                )
                for arm in arms
            ),
            cost_scale=1.0,
            config=SimulationConfig(deadline_seconds=0.5, lambda_penalty=0.0),
        )
        quality_cost = simulate_policy(policy="quality_cost", **common)
        quota_risk = simulate_policy(policy="quota_risk", **common)
        self.assertEqual(quality_cost.iloc[0]["model_id"], "arm_a")
        self.assertEqual(quota_risk.iloc[0]["model_id"], "arm_a")

    def test_joint_quota_proxy_has_separate_nonnegative_prices(self) -> None:
        utility = np.asarray([[10.0, 0.0], [10.0, 0.0]])
        demand = np.ones((2, 2))
        endpoints = (
            _endpoint("scarce", rpm=60.0, rpm_capacity=1.0, tpm=math.inf, tpm_capacity=math.inf),
            _endpoint("other", rpm=60.0, rpm_capacity=1.0, tpm=math.inf, tpm_capacity=math.inf),
        )
        prices = quota_proxy_gamma(
            utility, demand, endpoints, horizon_seconds=0.0
        )
        self.assertGreater(prices.gamma_rpm["scarce"], prices.gamma_rpm["other"])
        self.assertTrue(all(value >= 0 for value in prices.gamma_rpm.values()))
        self.assertTrue(all(value == 0 for value in prices.gamma_tpm.values()))
        self.assertIsInstance(prices, QuotaProxyPrices)

        tpm_endpoints = (
            _endpoint(
                "scarce",
                rpm=math.inf,
                rpm_capacity=math.inf,
                tpm=60.0,
                tpm_capacity=1.0,
            ),
            _endpoint(
                "other",
                rpm=math.inf,
                rpm_capacity=math.inf,
                tpm=60.0,
                tpm_capacity=1.0,
            ),
        )
        tpm_prices = quota_proxy_gamma(
            utility, demand, tpm_endpoints, horizon_seconds=0.0
        )
        self.assertGreater(
            tpm_prices.gamma_tpm["scarce"], tpm_prices.gamma_tpm["other"]
        )
        self.assertTrue(all(value == 0 for value in tpm_prices.gamma_rpm.values()))

    def test_oversized_predicted_demand_fails_without_actual_fallback(self) -> None:
        frame = pd.DataFrame(
            {"task_id": ["q0"], "eval_name": ["x"], "input_tokens": [5]}
        )
        world = SyntheticLatencyTable(
            task_ids=("q0",),
            arms=("arm",),
            base_seconds=np.ones((1, 1)),
            random_multiplier=np.ones((1, 1)),
        )
        with self.assertRaisesRegex(ValueError, "predicted_token_demand=15"):
            simulate_policy(
                policy="quota_risk",
                frame=frame,
                arms=("arm",),
                arrivals=[0.0],
                actual_quality=np.ones((1, 1)),
                actual_cost=np.zeros((1, 1)),
                actual_output_tokens=np.zeros((1, 1)),  # actual would fit, but is forbidden.
                profiles=_profiles(1, 1, output_tokens=10.0),
                world=world,
                endpoints=(_endpoint(tpm_capacity=12.0),),
                cost_scale=1.0,
            )


if __name__ == "__main__":
    unittest.main()
