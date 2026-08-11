from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from port_aqs.visualize import (
    build_paired_effects,
    generate_analysis,
    load_run_metrics,
    main,
)


class VisualizationTests(unittest.TestCase):
    @staticmethod
    def _write_runs(directory: Path, *, include_available: bool = True) -> None:
        policies = ["rpm_aware", "quality_cost", "static_latency"]
        if include_available:
            policies.append("available")
        base = {
            "rpm_aware": (0.700, 0.00080, 0.01, 2.0, 5.0, 7.0, 0.20, 0.30),
            "quality_cost": (0.710, 0.00075, 0.30, 20.0, 25.0, 30.0, 0.24, 0.50),
            "static_latency": (0.715, 0.00085, 0.35, 30.0, 35.0, 42.0, 0.27, 0.60),
            "available": (0.690, 0.00090, 0.08, 4.0, 8.0, 10.0, 0.18, 0.25),
        }
        records = []
        for seed in range(3):
            for policy in policies:
                quality, cost, slo, wait, e2e, cvar, hhi, binding = base[policy]
                records.append(
                    {
                        "condition": "stable_rho_030",
                        "seed": seed,
                        "policy": policy,
                        "arrival": "poisson",
                        "quota_mode": "rpm_only",
                        "quota_load": 0.3,
                        "requests": 20,
                        "completed": 20,
                        "quality_mean": quality + seed * 0.001,
                        "monetary_cost_mean": cost + seed * 0.000001,
                        "slo_violation_rate": slo + seed * 0.001,
                        "quota_wait_p95": wait + seed * 0.1,
                        "e2e_p95": e2e + seed * 0.1,
                        "cvar95_e2e": cvar + seed * 0.1,
                        "routing_hhi": hhi,
                        "rpm_binding_rate": binding + seed * 0.001,
                        "route_share__arm_00": 0.6,
                        "route_share__arm_01": 0.4,
                        "rpm_utilization__arm_00": 0.8,
                        "rpm_utilization__arm_01": 0.5,
                    }
                )
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(records).to_csv(
            directory / "run_metrics.csv", index=False
        )

    def test_load_filters_available_and_rejects_missing_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary) / "result"
            self._write_runs(result_dir)
            runs = load_run_metrics([result_dir])
            self.assertNotIn("available", set(runs["policy"]))
            self.assertEqual(len(runs), 9)

            broken_dir = Path(temporary) / "broken"
            broken_dir.mkdir()
            pd.DataFrame({"condition": ["x"]}).to_csv(
                broken_dir / "run_metrics.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "missing columns"):
                load_run_metrics([broken_dir])

    def test_paired_effects_have_one_consistent_better_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary) / "result"
            self._write_runs(result_dir, include_available=False)
            effects = build_paired_effects(load_run_metrics([result_dir]))
            slo = effects[
                (effects["metric"] == "slo_violation_rate")
                & (effects["baseline"] == "quality_cost")
            ].iloc[0]
            quality = effects[
                (effects["metric"] == "quality_mean")
                & (effects["baseline"] == "quality_cost")
            ].iloc[0]
            self.assertAlmostEqual(float(slo["raw_difference"]), -0.29)
            self.assertAlmostEqual(float(slo["benefit_display"]), 29.0)
            self.assertEqual(slo["display_unit"], "percentage points")
            self.assertAlmostEqual(float(quality["benefit_display"]), -1.0)

    def test_cli_generates_compact_idempotent_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_dir = root / "result"
            output_dir = root / "analysis"
            self._write_runs(result_dir)
            arguments = [
                str(result_dir),
                "--output",
                str(output_dir),
                "--dpi",
                "72",
            ]
            self.assertEqual(main(arguments), 0)
            self.assertEqual(main(arguments), 0)

            expected = {
                "analysis_summary.csv",
                "paired_effects.csv",
                "00_scorecard.png",
                "01_tradeoffs.png",
                "02_paired_effects.png",
                "03_seed_stability.png",
                "04_routing_mechanism.png",
                "analysis_report.pdf",
                "analysis_report.html",
                "analysis_manifest.json",
            }
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected)
            for name in expected:
                self.assertGreater((output_dir / name).stat().st_size, 0)
            self.assertEqual(
                (output_dir / "00_scorecard.png").read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )
            self.assertEqual(
                (output_dir / "analysis_report.pdf").read_bytes()[:4], b"%PDF"
            )
            summary = pd.read_csv(output_dir / "analysis_summary.csv")
            self.assertNotIn("available", set(summary["policy"]))
            self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
