from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from port_aqs.metrics import aggregate_runs, compute_metrics, paired_confidence_interval
from port_aqs.types import EndpointSpec


class MetricsTests(unittest.TestCase):
    def test_aggregate_runs_keeps_conditions_separate(self) -> None:
        runs = pd.DataFrame(
            [
                {"condition": condition, "policy": policy, "seed": seed, "quality_mean": value}
                for condition, offset in (("c1", 0.0), ("c2", 10.0))
                for policy, policy_offset in (("quota_risk", 1.0), ("baseline", 0.0))
                for seed, value in enumerate((offset + policy_offset, offset + policy_offset + 2.0))
            ]
        )
        runs["route_share__arm"] = 1.0
        result = aggregate_runs(runs, metrics=("quality_mean",))
        self.assertEqual(len(result), 4)
        means = {
            (row.condition, row.policy): row.quality_mean__mean
            for row in result.itertuples(index=False)
        }
        self.assertEqual(means[("c1", "quota_risk")], 2.0)
        self.assertEqual(means[("c2", "quota_risk")], 12.0)
        self.assertTrue((result["quality_mean__n"] == 2).all())
        self.assertTrue((result["route_share__arm__mean"] == 1.0).all())

    def test_paired_interval_uses_common_seed_differences(self) -> None:
        runs = pd.DataFrame(
            [
                {"policy": policy, "seed": seed, "metric": seed + offset}
                for seed in range(3)
                for policy, offset in (("quota_risk", 2.0), ("baseline", 0.0))
            ]
        )
        result = paired_confidence_interval(
            runs, metric="metric", treatment="quota_risk", baseline="baseline"
        )
        self.assertEqual(result["pairs"], 3)
        self.assertAlmostEqual(result["mean_difference"], 2.0)

    @staticmethod
    def _trace() -> pd.DataFrame:
        arrival = np.asarray([0.0, 0.0, 2.0])
        return pd.DataFrame(
            {
                "request_id": ["q0", "q1", "q2"],
                "model_id": ["arm", "arm", "arm"],
                "arrival_time": arrival,
                "dispatch_time": [0.0, 1.0, 2.0],
                "completion_time": [1.0, 2.0, 3.0],
                "admission_wait": [0.0, 1.0, 0.0],
                "rpm_wait": [0.0, 1.0, 0.0],
                "tpm_wait": [0.0, 0.5, 0.0],
                "rpm_binding": [False, True, False],
                "tpm_binding": [False, False, False],
                "api_latency": [1.0, 1.0, 1.0],
                "quality": [1.0, 1.0, 1.0],
                "monetary_cost": [0.0, 0.0, 0.0],
                "predicted_violation": [0.0, 0.0, 0.0],
                "predicted_token_demand": [10.0, 12.0, 8.0],
                "reserved_token_demand": [10.0, 12.0, 8.0],
                "actual_token_demand": [9.0, 14.0, 8.0],
                "deadline": [20.0, 20.0, 20.0],
                "e2e_latency": [1.0, 2.0, 1.0],
                "slo_violated": [False, False, False],
                "status": ["completed", "completed", "completed"],
            }
        )

    def test_dual_utilization_and_token_prediction_metrics(self) -> None:
        endpoints = {
            "arm": EndpointSpec(
                "arm",
                rpm=60.0,
                rpm_bucket_capacity=1.0,
                tpm=600.0,
                tpm_bucket_capacity=10.0,
            )
        }
        result = compute_metrics(self._trace(), endpoints=endpoints)
        self.assertAlmostEqual(result["rpm_wait_mean"], 1.0 / 3.0)
        self.assertAlmostEqual(result["rpm_wait_p99"], 0.98)
        self.assertAlmostEqual(result["tpm_wait_mean"], 1.0 / 6.0)
        self.assertAlmostEqual(result["rpm_binding_rate"], 1.0 / 3.0)
        # RPM available = initial 1 + 1/s * 2s = 3; all three dispatch.
        self.assertAlmostEqual(result["rpm_utilization__arm"], 1.0)
        # Reserved 30 tokens / (initial 10 + 10/s * 2s) = 1.
        self.assertAlmostEqual(result["tpm_utilization__arm"], 1.0)
        self.assertAlmostEqual(result["token_prediction_bias"], 1.0 / 3.0)
        self.assertAlmostEqual(result["token_prediction_mae"], 1.0)
        self.assertAlmostEqual(
            result["token_reservation_coverage_rate"], 2.0 / 3.0
        )
        self.assertAlmostEqual(result["token_underreservation_rate"], 1.0 / 3.0)
        self.assertEqual(result["actual_token_demand_total"], 31.0)
        self.assertAlmostEqual(result["actual_tpm_utilization__arm"], 31.0 / 30.0)
        self.assertEqual(result["routing_hhi"], 1.0)

    def test_utilization_uses_common_arrival_window_not_drain_time(self) -> None:
        trace = self._trace()
        trace["dispatch_time"] = [0.0, 1.0, 10.0]
        endpoints = {
            "arm": EndpointSpec(
                "arm",
                rpm=60.0,
                rpm_bucket_capacity=1.0,
                tpm=np.inf,
                tpm_bucket_capacity=np.inf,
            )
        }
        result = compute_metrics(trace, endpoints=endpoints)
        # Observation window is [first arrival=0, last arrival=2], so only two
        # dispatches count against 1 initial + 2 refilled permits.
        self.assertAlmostEqual(result["rpm_utilization__arm"], 2.0 / 3.0)

    def test_route_share_includes_unselected_endpoint_as_zero(self) -> None:
        endpoints = {
            arm: EndpointSpec(arm, 60.0, 1.0, 600.0, 10.0)
            for arm in ("arm", "never")
        }
        result = compute_metrics(self._trace(), endpoints=endpoints)
        self.assertEqual(result["route_share__arm"], 1.0)
        self.assertEqual(result["route_share__never"], 0.0)
        self.assertEqual(result["routing_hhi"], 1.0)

    def test_disabled_resource_utilization_is_nan(self) -> None:
        endpoint = EndpointSpec(
            "arm",
            rpm=60.0,
            rpm_bucket_capacity=1.0,
            tpm=np.inf,
            tpm_bucket_capacity=np.inf,
        )
        result = compute_metrics(self._trace(), endpoints={"arm": endpoint})
        self.assertTrue(np.isnan(result["tpm_utilization__arm"]))

    def test_paired_interval_all_nan_metric_returns_zero_pairs(self) -> None:
        runs = pd.DataFrame(
            [
                {"policy": policy, "seed": seed, "metric": np.nan}
                for seed in range(2)
                for policy in ("quota_risk", "baseline")
            ]
        )
        result = paired_confidence_interval(
            runs, metric="metric", treatment="quota_risk", baseline="baseline"
        )
        self.assertEqual(result["pairs"], 0)
        self.assertTrue(np.isnan(result["mean_difference"]))


if __name__ == "__main__":
    unittest.main()
