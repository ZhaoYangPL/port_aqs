from __future__ import annotations

import math
import unittest

import numpy as np

from port_aqs import (
    CandidateEstimate,
    EndpointSpec,
    LatencyEstimator,
    PortAQSRouter,
    QuotaLimiter,
    QuotaPrice,
    RoutingMode,
)


def endpoint(
    model_id: str,
    *,
    rpm: float = math.inf,
    rpm_capacity: float = math.inf,
    tpm: float = math.inf,
    tpm_capacity: float = math.inf,
) -> EndpointSpec:
    return EndpointSpec(model_id, rpm, rpm_capacity, tpm, tpm_capacity)


class QuotaLimiterTests(unittest.TestCase):
    def test_variable_tpm_demand_changes_wait(self) -> None:
        limiter = QuotaLimiter(
            endpoint("arm", tpm=60.0, tpm_capacity=10.0),
            initial_tpm_tokens=0.0,
        )
        small = limiter.preview(0.0, 2.0)
        large = limiter.preview(0.0, 5.0)
        self.assertEqual(small.admission_wait, 2.0)
        self.assertEqual(large.admission_wait, 5.0)
        self.assertEqual(small.binding_resource, "tpm")
        # Preview is non-mutating.
        self.assertEqual(large, limiter.preview(0.0, 5.0))

    def test_joint_bucket_reports_binding_resource(self) -> None:
        tpm_bound = QuotaLimiter(
            endpoint(
                "tpm-bound", rpm=60.0, rpm_capacity=1.0, tpm=120.0, tpm_capacity=4.0
            ),
            initial_rpm_tokens=0.0,
            initial_tpm_tokens=0.0,
        ).preview(0.0, 4.0)
        self.assertEqual(tpm_bound.rpm_ready_wait, 1.0)
        self.assertEqual(tpm_bound.tpm_ready_wait, 2.0)
        self.assertEqual(tpm_bound.admission_wait, 2.0)
        self.assertEqual(tpm_bound.binding_resource, "tpm")

        rpm_bound = QuotaLimiter(
            endpoint(
                "rpm-bound", rpm=60.0, rpm_capacity=1.0, tpm=120.0, tpm_capacity=4.0
            ),
            initial_rpm_tokens=0.0,
            initial_tpm_tokens=0.0,
        ).preview(0.0, 1.0)
        self.assertEqual(rpm_bound.binding_resource, "rpm")

        both = QuotaLimiter(
            endpoint(
                "both", rpm=60.0, rpm_capacity=1.0, tpm=60.0, tpm_capacity=1.0
            ),
            initial_rpm_tokens=0.0,
            initial_tpm_tokens=0.0,
        ).preview(0.0, 1.0)
        self.assertEqual(both.rpm_ready_wait, 1.0)
        self.assertEqual(both.tpm_ready_wait, 1.0)
        self.assertEqual(both.binding_resource, "both")

    def test_fifo_and_dual_reservation_conservation(self) -> None:
        limiter = QuotaLimiter(
            endpoint(
                "arm", rpm=120.0, rpm_capacity=2.0, tpm=600.0, tpm_capacity=10.0
            ),
            initial_rpm_tokens=0.0,
            initial_tpm_tokens=0.0,
        )
        reservations = [
            limiter.commit(0.0, 5.0, "q1"),
            limiter.commit(0.0, 2.0, "q2"),
            limiter.commit(0.0, 8.0, "q3"),
        ]
        self.assertEqual(
            [item.dispatch_time for item in reservations], [0.5, 1.0, 1.5]
        )
        self.assertEqual([item.sequence for item in reservations], [1, 2, 3])

        at_zero = limiter.snapshot(0.0)
        self.assertEqual(at_zero.total_committed, 3)
        self.assertEqual(at_zero.total_dispatched, 0)
        self.assertEqual(at_zero.queued, 3)
        self.assertEqual(at_zero.total_tpm_committed, 15.0)
        self.assertEqual(at_zero.total_tpm_dispatched, 0.0)
        self.assertEqual(at_zero.queued_tpm_demand, 15.0)
        self.assertGreaterEqual(at_zero.rpm_tokens, 0.0)
        self.assertGreaterEqual(at_zero.tpm_tokens, 0.0)

        at_one = limiter.snapshot(1.0)
        self.assertEqual(at_one.total_dispatched, 2)
        self.assertEqual(at_one.queued, 1)
        self.assertEqual(at_one.total_tpm_dispatched, 7.0)
        self.assertEqual(at_one.queued_tpm_demand, 8.0)

        drained = limiter.snapshot(2.0)
        self.assertEqual(drained.total_dispatched, 3)
        self.assertEqual(drained.queued, 0)
        self.assertEqual(drained.total_tpm_dispatched, 15.0)
        self.assertGreaterEqual(drained.rpm_tokens, 0.0)
        self.assertGreaterEqual(drained.tpm_tokens, 0.0)

    def test_completion_has_no_quota_or_feedback_effect(self) -> None:
        limiter = QuotaLimiter(
            endpoint(
                "arm", rpm=60.0, rpm_capacity=1.0, tpm=60.0, tpm_capacity=5.0
            )
        )
        limiter.commit(0.0, 4.0, "q0")
        before = limiter.snapshot(0.0)
        limiter.on_completion(100.0)
        after = limiter.snapshot(0.0)
        self.assertEqual(before, after)

    def test_each_resource_can_be_disabled_with_infinity(self) -> None:
        no_tpm = QuotaLimiter(
            endpoint(
                "no-tpm",
                rpm=60.0,
                rpm_capacity=1.0,
                tpm=math.inf,
                tpm_capacity=math.inf,
            )
        )
        first = no_tpm.commit(0.0, 1_000_000.0, "q0")
        second = no_tpm.commit(0.0, 1_000_000.0, "q1")
        third = no_tpm.commit(0.0, 1_000_000.0, "q2")
        self.assertEqual(first.admission_wait, 0.0)
        self.assertEqual(second.admission_wait, 1.0)
        self.assertEqual(third.admission_wait, 2.0)
        self.assertEqual(third.tpm_ready_wait, 0.0)
        self.assertEqual(second.binding_resource, "rpm")
        self.assertTrue(math.isinf(second.snapshot_before.tpm_tokens))
        # RPM-only uses the virtual-balance fast path; no FIFO replay queue is
        # populated, while public conservation snapshots remain exact.
        self.assertTrue(no_tpm._rpm_only_fast)
        self.assertEqual(len(no_tpm._pending), 0)
        at_one = no_tpm.snapshot(1.0)
        self.assertEqual(at_one.total_dispatched, 2)
        self.assertEqual(at_one.queued, 1)
        self.assertEqual(at_one.queued_tpm_demand, 1_000_000.0)

        no_rpm = QuotaLimiter(
            endpoint(
                "no-rpm",
                rpm=math.inf,
                rpm_capacity=math.inf,
                tpm=60.0,
                tpm_capacity=2.0,
            ),
            initial_tpm_tokens=0.0,
        )
        reservation = no_rpm.commit(0.0, 2.0, "q")
        self.assertEqual(reservation.rpm_ready_wait, 0.0)
        self.assertEqual(reservation.tpm_ready_wait, 2.0)
        self.assertEqual(reservation.binding_resource, "tpm")

    def test_rpm_only_fast_path_matches_virtual_balance_formula(self) -> None:
        limiter = QuotaLimiter(
            endpoint(
                "rpm-only",
                rpm=120.0,
                rpm_capacity=3.0,
                tpm=math.inf,
                tpm_capacity=math.inf,
            )
        )
        refill_rate = 2.0
        virtual = 3.0
        previous = 0.0
        arrivals = [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 1.5, 4.0]
        for index, timestamp in enumerate(arrivals):
            virtual = min(3.0, virtual + refill_rate * (timestamp - previous))
            expected = max(0.0, (1.0 - virtual) / refill_rate)
            preview = limiter.preview(timestamp, 10.0 + index)
            reservation = limiter.commit(timestamp, 10.0 + index, str(index))
            self.assertAlmostEqual(preview.admission_wait, expected)
            self.assertAlmostEqual(reservation.admission_wait, expected)
            virtual -= 1.0
            previous = timestamp

        drained = limiter.snapshot(max(item for item in limiter._rpm_dispatch_times))
        self.assertEqual(drained.total_committed, len(arrivals))
        self.assertEqual(drained.total_dispatched, len(arrivals))
        self.assertEqual(drained.queued, 0)

    def test_finite_tpm_bucket_rejects_oversized_reservation(self) -> None:
        limiter = QuotaLimiter(
            endpoint("arm", tpm=60.0, tpm_capacity=10.0)
        )
        with self.assertRaisesRegex(ValueError, "exceeds finite TPM bucket capacity"):
            limiter.preview(0.0, 11.0)


class LatencyEstimatorTests(unittest.TestCase):
    def test_weighted_static_cdf_quantiles_and_median(self) -> None:
        estimator = LatencyEstimator({"arm": [2.0, 4.0]})
        samples = (1.0, 10.0)
        weights = (0.9, 0.1)
        self.assertAlmostEqual(
            estimator.predict_cdf(
                "arm", 5.0, static_samples=samples, sample_weights=weights
            ),
            0.9,
        )
        self.assertEqual(
            estimator.predict_quantile(
                "arm", 0.5, static_samples=samples, sample_weights=weights
            ),
            1.0,
        )
        self.assertEqual(
            estimator.predict_quantile(
                "arm", 0.95, static_samples=samples, sample_weights=weights
            ),
            10.0,
        )
        self.assertEqual(estimator.static_median("arm", samples, weights), 1.0)
        self.assertAlmostEqual(estimator.predict_cdf("arm", 2.0), 0.5)

    def test_stage1_estimator_has_no_completion_feedback_api(self) -> None:
        estimator = LatencyEstimator({"arm": [1.0]})
        self.assertFalse(hasattr(estimator, "update_on_completion"))
        self.assertFalse(hasattr(estimator, "health_multiplier"))
        np.testing.assert_allclose(estimator.predict_samples("arm"), [1.0])

    def test_deadline_consumed_by_admission_has_unit_risk(self) -> None:
        estimator = LatencyEstimator({"arm": [1.0, 2.0]})
        self.assertEqual(
            estimator.slo_violation_probability("arm", 3.0, 3.0), 1.0
        )

    def test_candidate_normalizes_weights_and_requires_token_demand(self) -> None:
        candidate = CandidateEstimate(
            "arm",
            0.5,
            0.0,
            predicted_token_demand=20.0,
            static_latency_samples=(1.0, 2.0),
            neighbor_weights=(9.0, 1.0),
        )
        np.testing.assert_allclose(candidate.neighbor_weights, [0.9, 0.1])
        with self.assertRaises(ValueError):
            CandidateEstimate("arm", 0.5, 0.0, predicted_token_demand=0.0)


class RouterTests(unittest.TestCase):
    @staticmethod
    def _candidates() -> list[CandidateEstimate]:
        return [
            CandidateEstimate(
                "slow", 0.9, 0.0, 10.0, static_latency_samples=(10.0,)
            ),
            CandidateEstimate(
                "fast", 0.8, 0.0, 10.0, static_latency_samples=(1.0,)
            ),
        ]

    @staticmethod
    def _router(mode: RoutingMode, penalty: float) -> PortAQSRouter:
        endpoints = [endpoint("slow"), endpoint("fast")]
        return PortAQSRouter(
            endpoints,
            LatencyEstimator({"slow": [10.0], "fast": [1.0]}),
            lambda_penalty=penalty,
            mode=mode,
        )

    def test_zero_lambda_strictly_reduces_to_baseline_including_tie(self) -> None:
        baseline = self._router(RoutingMode.NO_LATENCY, 0.0)
        admission = self._router(RoutingMode.ADMISSION, 0.0)
        self.assertEqual(
            baseline.route("b", 0.0, 5.0, self._candidates()).selected_model_id,
            admission.route("a", 0.0, 5.0, self._candidates()).selected_model_id,
        )
        self.assertEqual(
            admission.route("a2", 0.0, 5.0, self._candidates()).selected_model_id,
            "slow",
        )

        tied_candidates = [
            CandidateEstimate("a_slow", 0.5, 0.0, 1.0, static_latency_samples=(10.0,)),
            CandidateEstimate("z_fast", 0.5, 0.0, 1.0, static_latency_samples=(1.0,)),
        ]
        tied = PortAQSRouter(
            [endpoint("a_slow"), endpoint("z_fast")],
            LatencyEstimator({"a_slow": [10.0], "z_fast": [1.0]}),
            lambda_penalty=0.0,
            mode=RoutingMode.ADMISSION,
        )
        self.assertEqual(
            tied.route("tie", 0.0, 0.5, tied_candidates).selected_model_id,
            "a_slow",
        )

    def test_infinite_quota_admission_reduces_to_static_risk(self) -> None:
        static = self._router(RoutingMode.STATIC, 1.0)
        admission = self._router(RoutingMode.ADMISSION, 1.0)
        static_decision = static.route("s", 0.0, 5.0, self._candidates())
        admission_decision = admission.route("a", 0.0, 5.0, self._candidates())
        self.assertEqual(static_decision.selected_model_id, "fast")
        self.assertEqual(static_decision.selected_model_id, admission_decision.selected_model_id)
        self.assertEqual(
            [item.predicted_risk for item in static_decision.candidates],
            [item.predicted_risk for item in admission_decision.candidates],
        )

    def test_admission_risk_uses_predicted_tpm_demand(self) -> None:
        endpoints = [
            endpoint(
                "quality", tpm=60.0, tpm_capacity=20.0
            ),
            endpoint("available"),
        ]
        candidates = [
            CandidateEstimate(
                "quality", 0.9, 0.0, 10.0, static_latency_samples=(1.0,)
            ),
            CandidateEstimate(
                "available", 0.8, 0.0, 10.0, static_latency_samples=(1.0,)
            ),
        ]
        static = PortAQSRouter(
            endpoints,
            LatencyEstimator({"quality": [1.0], "available": [1.0]}),
            lambda_penalty=1.0,
            mode=RoutingMode.STATIC,
            initial_quota_balances={"quality": (math.inf, 0.0)},
        )
        admission = PortAQSRouter(
            endpoints,
            LatencyEstimator({"quality": [1.0], "available": [1.0]}),
            lambda_penalty=1.0,
            mode=RoutingMode.ADMISSION,
            initial_quota_balances={"quality": (math.inf, 0.0)},
        )
        self.assertEqual(static.route("s", 0.0, 5.0, candidates).selected_model_id, "quality")
        decision = admission.route("a", 0.0, 5.0, candidates)
        self.assertEqual(decision.selected_model_id, "available")
        risks = {item.model_id: item.predicted_risk for item in decision.candidates}
        self.assertEqual(risks, {"quality": 1.0, "available": 0.0})

    def test_scarcity_price_scales_with_predicted_token_demand(self) -> None:
        candidates = [
            CandidateEstimate("large", 0.9, 0.0, 100.0, static_latency_samples=(1.0,)),
            CandidateEstimate("small", 0.9, 0.0, 10.0, static_latency_samples=(1.0,)),
        ]
        router = PortAQSRouter(
            [endpoint("large"), endpoint("small")],
            LatencyEstimator({"large": [1.0], "small": [1.0]}),
            quota_prices={
                "large": QuotaPrice(gamma_rpm=0.1, gamma_tpm=0.01),
                "small": QuotaPrice(gamma_rpm=0.1, gamma_tpm=0.01),
            },
            mode=RoutingMode.NO_LATENCY,
        )
        decision = router.route("q", 0.0, 5.0, candidates)
        penalties = {item.model_id: item.scarcity_penalty for item in decision.candidates}
        self.assertEqual(penalties, {"large": 1.1, "small": 0.2})
        self.assertEqual(decision.selected_model_id, "small")

    def test_saturated_risk_tie_uses_e2e_p95_before_model_id(self) -> None:
        candidates = [
            CandidateEstimate("a_slow", 0.5, 0.0, 1.0, static_latency_samples=(10.0,)),
            CandidateEstimate("z_fast", 0.5, 0.0, 1.0, static_latency_samples=(2.0,)),
        ]
        router = PortAQSRouter(
            [endpoint("a_slow"), endpoint("z_fast")],
            LatencyEstimator({"a_slow": [10.0], "z_fast": [2.0]}),
            lambda_penalty=1.0,
            mode=RoutingMode.STATIC,
        )
        self.assertEqual(
            router.route("q", 0.0, 1.0, candidates).selected_model_id,
            "z_fast",
        )

    def test_static_saturated_tie_does_not_read_live_admission_wait(self) -> None:
        endpoints = [
            endpoint("a_congested", rpm=60.0, rpm_capacity=1.0),
            endpoint("z_clear", rpm=60.0, rpm_capacity=1.0),
        ]
        candidates = [
            CandidateEstimate(
                model_id,
                0.5,
                0.0,
                1.0,
                static_latency_samples=(10.0,),
            )
            for model_id in ("a_congested", "z_clear")
        ]
        static = PortAQSRouter(
            endpoints,
            LatencyEstimator({model_id: [10.0] for model_id in ("a_congested", "z_clear")}),
            lambda_penalty=1.0,
            mode=RoutingMode.STATIC,
        )
        admission = PortAQSRouter(
            endpoints,
            LatencyEstimator({model_id: [10.0] for model_id in ("a_congested", "z_clear")}),
            lambda_penalty=1.0,
            mode=RoutingMode.ADMISSION,
        )
        for router in (static, admission):
            router.quota_limiter("a_congested").commit(0.0, 1.0, "warmup")

        # STATIC sees identical frozen profiles and therefore uses model_id;
        # ADMISSION is allowed to break the saturated tie with live waiting.
        self.assertEqual(
            static.route("static", 0.0, 1.0, candidates).selected_model_id,
            "a_congested",
        )
        self.assertEqual(
            admission.route("admission", 0.0, 1.0, candidates).selected_model_id,
            "z_clear",
        )

    def test_stage1_router_exposes_no_completion_feedback(self) -> None:
        router = self._router(RoutingMode.ADMISSION, 1.0)
        self.assertFalse(hasattr(router, "record_completion"))
        with self.assertRaises(ValueError):
            RoutingMode.parse("dynamic_health")

    def test_completion_order_cannot_change_a_route(self) -> None:
        untouched = self._router(RoutingMode.ADMISSION, 1.0)
        completed = self._router(RoutingMode.ADMISSION, 1.0)
        completed.quota_limiter("slow").on_completion(100.0)
        completed.quota_limiter("fast").on_completion(2.0)
        completed.quota_limiter("slow").on_completion(1.0)
        left = untouched.route("left", 0.0, 5.0, self._candidates())
        right = completed.route("right", 0.0, 5.0, self._candidates())
        self.assertEqual(left.selected_model_id, right.selected_model_id)
        self.assertEqual(
            [item.predicted_risk for item in left.candidates],
            [item.predicted_risk for item in right.candidates],
        )


if __name__ == "__main__":
    unittest.main()
