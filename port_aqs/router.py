"""Feedback-free PORT-inspired RPM/TPM quota and SLO router."""

from __future__ import annotations

from dataclasses import replace
import math
from threading import RLock
from typing import Iterable, Mapping

from .latency import LatencyEstimator
from .quota import QuotaLimiter
from .types import (
    CandidateEstimate,
    EndpointSpec,
    QuotaPrice,
    RoutingDecision,
    RoutingMode,
)


class PortAQSRouter:
    """Score candidates and reserve joint quota exactly once.

    The frozen Stage 1 score is

    ``quality - beta*cost - (gamma_rpm + gamma_tpm*token_demand) - lambda*risk``.

    ``quota_prices`` and all scalar parameters come from calibration.  The
    streaming router changes only observable RPM/TPM admission state; it has no
    completion-feedback method and never updates latency or quality profiles.
    """

    def __init__(
        self,
        endpoints: Iterable[EndpointSpec],
        latency_estimator: LatencyEstimator,
        *,
        beta: float = 0.0,
        lambda_penalty: float = 0.0,
        quota_prices: Mapping[str, QuotaPrice] | None = None,
        mode: RoutingMode | str = RoutingMode.ADMISSION,
        start_time: float = 0.0,
        initial_quota_balances: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        specs = tuple(endpoints)
        if not specs:
            raise ValueError("at least one endpoint is required")
        by_id = {spec.model_id: spec for spec in specs}
        if len(by_id) != len(specs):
            raise ValueError("endpoint model_id values must be unique")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be non-negative and finite")
        if not math.isfinite(lambda_penalty) or lambda_penalty < 0:
            raise ValueError("lambda_penalty must be non-negative and finite")

        raw_prices = dict(quota_prices or {})
        unknown_prices = set(raw_prices).difference(by_id)
        if unknown_prices:
            raise ValueError(
                f"quota_prices contains unknown endpoints: {sorted(unknown_prices)}"
            )
        for model_id, price in raw_prices.items():
            if not isinstance(price, QuotaPrice):
                raise TypeError(f"quota_prices[{model_id!r}] must be a QuotaPrice")

        raw_balances = dict(initial_quota_balances or {})
        unknown_balances = set(raw_balances).difference(by_id)
        if unknown_balances:
            raise ValueError(
                "initial_quota_balances contains unknown endpoints: "
                f"{sorted(unknown_balances)}"
            )
        for model_id, balances in raw_balances.items():
            if len(balances) != 2:
                raise ValueError(
                    f"initial_quota_balances[{model_id!r}] must be (rpm, tpm)"
                )

        self.endpoints = by_id
        self.latency_estimator = latency_estimator
        self.beta = float(beta)
        self.lambda_penalty = float(lambda_penalty)
        self.quota_prices = {
            model_id: raw_prices.get(model_id, QuotaPrice()) for model_id in by_id
        }
        self.mode = RoutingMode.parse(mode)
        self.limiters = {}
        for model_id, spec in by_id.items():
            balances = raw_balances.get(model_id)
            self.limiters[model_id] = QuotaLimiter(
                spec,
                start_time=start_time,
                initial_rpm_tokens=None if balances is None else balances[0],
                initial_tpm_tokens=None if balances is None else balances[1],
            )
        self._lock = RLock()

    def quota_limiter(self, model_id: str) -> QuotaLimiter:
        try:
            return self.limiters[model_id]
        except KeyError as error:
            raise KeyError(f"unknown endpoint {model_id!r}") from error

    def _score_candidate(
        self,
        candidate: CandidateEstimate,
        arrival_time: float,
        deadline: float,
    ) -> CandidateEstimate:
        if candidate.model_id not in self.endpoints:
            raise ValueError(f"candidate refers to unknown endpoint {candidate.model_id!r}")
        preview = self.limiters[candidate.model_id].preview(
            arrival_time, candidate.predicted_token_demand
        )
        sample_override = candidate.static_latency_samples or None
        weight_override = candidate.neighbor_weights or None
        effective_wait = (
            preview.admission_wait if self.mode is RoutingMode.ADMISSION else 0.0
        )

        if self.mode is RoutingMode.NO_LATENCY:
            risk = 0.0
            try:
                response_p95 = self.latency_estimator.predict_quantile(
                    candidate.model_id,
                    0.95,
                    static_samples=sample_override,
                    sample_weights=weight_override,
                )
            except KeyError:
                response_p95 = math.nan
        else:
            risk = self.latency_estimator.slo_violation_probability(
                candidate.model_id,
                deadline,
                effective_wait,
                static_samples=sample_override,
                sample_weights=weight_override,
            )
            response_p95 = self.latency_estimator.predict_quantile(
                candidate.model_id,
                0.95,
                static_samples=sample_override,
                sample_weights=weight_override,
            )

        # Always report an admission-aware E2E diagnostic, even when an
        # ablation intentionally omits admission wait from its risk score.
        e2e_p95 = response_p95 + preview.admission_wait
        scarcity = self.quota_prices[candidate.model_id].penalty(
            candidate.predicted_token_demand
        )
        score = (
            candidate.predicted_quality
            - self.beta * candidate.normalized_cost
            - scarcity
            - self.lambda_penalty * risk
        )
        return replace(
            candidate,
            quota_preview=preview,
            predicted_risk=risk,
            predicted_response_p95=response_p95,
            predicted_e2e_p95=e2e_p95,
            scarcity_penalty=scarcity,
            score=score,
        )

    def _choose(self, scored: tuple[CandidateEstimate, ...]) -> CandidateEstimate:
        best_score = max(candidate.score for candidate in scored)
        tied = tuple(
            candidate
            for candidate in scored
            if math.isclose(candidate.score, best_score, rel_tol=0.0, abs_tol=1e-12)
        )
        if len(tied) == 1:
            return tied[0]
        # lambda=0 must strictly reduce to the quality/cost/scarcity baseline,
        # including tie behavior.  The p95 exception is used only for a
        # genuinely active, fully saturated risk term.
        if self.lambda_penalty > 0.0 and all(
            math.isclose(candidate.predicted_risk, 1.0, rel_tol=0.0, abs_tol=1e-12)
            for candidate in tied
        ):
            # STATIC may use only the frozen response profile.  Admission-aware
            # E2E p95 contains live quota wait and would contaminate the ablation.
            if self.mode is RoutingMode.ADMISSION:
                return min(
                    tied,
                    key=lambda item: (item.predicted_e2e_p95, item.model_id),
                )
            return min(
                tied,
                key=lambda item: (item.predicted_response_p95, item.model_id),
            )
        return min(tied, key=lambda item: item.model_id)

    def route(
        self,
        request_id: str,
        arrival_time: float,
        deadline: float,
        candidates: Iterable[CandidateEstimate],
    ) -> RoutingDecision:
        """Choose once, commit to that endpoint FIFO, and never reroute."""

        if not math.isfinite(arrival_time):
            raise ValueError("arrival_time must be finite")
        if not math.isfinite(deadline) or deadline <= 0:
            raise ValueError("deadline must be positive and finite")
        raw = tuple(candidates)
        if not raw:
            raise ValueError("at least one candidate is required")
        ids = [candidate.model_id for candidate in raw]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate model_id values must be unique")

        # One lock makes all previews and the winning commit a single atomic
        # routing decision across endpoints.
        with self._lock:
            scored = tuple(
                self._score_candidate(candidate, arrival_time, deadline)
                for candidate in raw
            )
            selected = self._choose(scored)
            reservation = self.limiters[selected.model_id].commit(
                arrival_time,
                selected.predicted_token_demand,
                request_id=request_id,
            )
            preview = selected.quota_preview
            if preview is None:  # pragma: no cover - construction invariant
                raise RuntimeError("selected candidate has no quota preview")
            if not math.isclose(
                preview.dispatch_time,
                reservation.dispatch_time,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError("selected quota preview changed before commit")
            return RoutingDecision(
                request_id=request_id,
                arrival_time=arrival_time,
                deadline=deadline,
                mode=self.mode,
                selected_model_id=selected.model_id,
                selected_candidate=selected,
                reservation=reservation,
                candidates=scored,
            )
