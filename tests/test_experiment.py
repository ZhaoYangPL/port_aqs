from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from port_aqs.data import KNNPrediction
from port_aqs.experiment import (
    _aggregate_request_capacity,
    _demand_statistics,
    _expanded_conditions,
    _make_endpoints,
    _preflight_demands,
    _reorder_profiles,
    _select_conditions,
)


class ExperimentHelpersTests(unittest.TestCase):
    def test_condition_selection_is_ordered_validated_and_disables_reexpansion(self) -> None:
        selected = _select_conditions(
            {
                "experiment": {
                    "conditions": [
                        {"name": "rho_030", "quota_mode": "rpm_only"},
                        {"name": "rho_050", "quota_mode": "rpm_only"},
                    ],
                    "robustness": {
                        "enabled": True,
                        "quota_permutations": [0],
                    },
                }
            },
            ["heterogeneous_joint_p0", "rho_050"],
        )
        self.assertEqual(
            [item["name"] for item in selected["experiment"]["conditions"]],
            ["heterogeneous_joint_p0", "rho_050"],
        )
        self.assertEqual(selected["experiment"]["robustness"], {"enabled": False})
        with self.assertRaisesRegex(ValueError, "unknown conditions"):
            _select_conditions(selected, ["missing"])

    def test_heterogeneous_robustness_adds_five_stable_joint_permutations(self) -> None:
        conditions = _expanded_conditions(
            {
                "conditions": [{"name": "main", "quota_mode": "joint"}],
                "robustness": {
                    "enabled": True,
                    "quota_permutations": [0, 1, 2, 3, 4],
                },
            }
        )
        names = {condition["name"] for condition in conditions}
        self.assertEqual(len(conditions), 6)
        self.assertIn("heterogeneous_joint_p0", names)
        self.assertIn("heterogeneous_joint_p4", names)
        self.assertTrue(all("slowdown" not in name for name in names))

    @staticmethod
    def _profiles() -> KNNPrediction:
        return KNNPrediction(
            quality=np.asarray([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]]),
            cost=np.zeros((3, 2)),
            output_tokens=np.asarray([[9.0, 19.0], [19.0, 9.0], [29.0, 29.0]]),
            latency_samples=np.arange(12, dtype=float).reshape(3, 2, 2) + 1.0,
            neighbor_weights=np.asarray([[0.7, 0.3]] * 3),
        )

    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "task_id": ["q0", "q1", "q2"],
                "eval_name": ["x", "x", "x"],
                "input_tokens": [1.0, 1.0, 1.0],
            }
        )

    def test_profile_reorder_is_common_across_all_arrays(self) -> None:
        profiles = self._profiles()
        order = np.asarray([2, 0, 1])
        reordered = _reorder_profiles(profiles, order)
        np.testing.assert_array_equal(reordered.quality, profiles.quality[order])
        np.testing.assert_array_equal(
            reordered.output_tokens, profiles.output_tokens[order]
        )
        np.testing.assert_array_equal(
            reordered.latency_samples, profiles.latency_samples[order]
        )
        np.testing.assert_array_equal(
            reordered.neighbor_weights, profiles.neighbor_weights[order]
        )

    def test_joint_and_single_resource_endpoint_modes(self) -> None:
        arms = ("arm_00", "arm_01")
        stats = _demand_statistics(self._frame(), self._profiles(), arms)
        quota = {
            "homogeneous_rpm": 60.0,
            "rpm_bucket_capacity": 6.0,
            "tpm_bucket_request_equivalents": 6.0,
            "tpm_calibration_max_headroom": 1.10,
        }
        joint = _make_endpoints(
            arms, quota, {"quota_mode": "joint"}, stats
        )
        rpm_only = _make_endpoints(
            arms, quota, {"quota_mode": "rpm_only"}, stats
        )
        tpm_only = _make_endpoints(
            arms, quota, {"quota_mode": "tpm_only"}, stats
        )
        self.assertEqual(joint[0].rpm, 60.0)
        self.assertAlmostEqual(joint[0].tpm, 60.0 * stats.mean_by_arm["arm_00"])
        self.assertGreaterEqual(
            joint[0].tpm_bucket_capacity,
            1.10 * stats.max_by_arm["arm_00"],
        )
        self.assertTrue(math.isinf(rpm_only[0].tpm))
        self.assertTrue(math.isinf(rpm_only[0].tpm_bucket_capacity))
        self.assertTrue(math.isinf(tpm_only[0].rpm))
        self.assertTrue(math.isinf(tpm_only[0].rpm_bucket_capacity))
        self.assertAlmostEqual(
            _aggregate_request_capacity(joint, stats),
            _aggregate_request_capacity(rpm_only, stats),
        )
        self.assertAlmostEqual(
            _aggregate_request_capacity(joint, stats),
            _aggregate_request_capacity(tpm_only, stats),
        )

    def test_preflight_uses_prediction_and_fails_explicitly(self) -> None:
        arms = ("arm_00", "arm_01")
        profiles = self._profiles()
        stats = _demand_statistics(self._frame(), profiles, arms)
        endpoints = _make_endpoints(
            arms,
            {
                "homogeneous_rpm": 60.0,
                "rpm_bucket_capacity": 6.0,
                "tpm_bucket_request_equivalents": 6.0,
                "tpm_calibration_max_headroom": 1.0,
            },
            {"quota_mode": "joint"},
            stats,
        )
        oversized = KNNPrediction(
            quality=profiles.quality,
            cost=profiles.cost,
            output_tokens=profiles.output_tokens + 1000.0,
            latency_samples=profiles.latency_samples,
            neighbor_weights=profiles.neighbor_weights,
        )
        with self.assertRaisesRegex(ValueError, "actual token outcomes were not consulted"):
            _preflight_demands(
                self._frame(), oversized, endpoints, label="test stream"
            )


if __name__ == "__main__":
    unittest.main()
