"""Real-photograph validation utilities for ATHER-ACP.

This module validates the integrity and repeatability of ACP analysis on the
actual image currently loaded in ATHER. It does not claim perceptual quality,
photographic intent, or ground-truth accuracy.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from core.adaptive_engine import build_processing_plan


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def validate_real_image_plan(
    image: np.ndarray,
    plan: dict[str, Any],
    registered_algorithms: Iterable[str],
) -> dict[str, Any]:
    """Run integrity checks and a deterministic repeat analysis for one real image.

    `image` must be the actual original image loaded in the ATHER workspace.
    The function does not modify it or execute any enhancement algorithm.
    """
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    if not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "passed": False,
            "checks": [{"name": "Input image", "passed": False,
                        "detail": "No valid current image array was supplied."}],
            "summary": "Validation could not run because the current image is invalid.",
        }

    try:
        analysis = plan["analysis"]
        image_info = analysis["image"]
        luminance = analysis["luminance"]
        noise = analysis["noise"]
        classification = plan["classification"]
        recommendations = plan.get("recommendations", [])

        expected_height, expected_width = image.shape[:2]
        dimensions_ok = (
            image_info.get("width") == expected_width
            and image_info.get("height") == expected_height
            and image_info.get("total_pixels") == expected_width * expected_height
        )
        record(
            "Image identity and dimensions",
            dimensions_ok,
            f"Current image: {expected_width} × {expected_height}; "
            f"ACP reports: {image_info.get('width')} × {image_info.get('height')}.",
        )

        engine_ok = (
            plan.get("engine") == "ATHER-ACP"
            and plan.get("execution_mode") == "recommendation_only"
            and plan.get("automatic_execution") is False
        )
        record(
            "Classical recommendation-only mode",
            engine_ok,
            "ACP remains an analysis/recommendation engine; no automatic enhancement is executed."
            if engine_ok else "Unexpected engine identity or execution mode.",
        )

        required_metrics = {
            "mean": (0.0, 255.0),
            "median": (0.0, 255.0),
            "standard_deviation": (0.0, 127.5),
            "dynamic_range": (0.0, 255.0),
            "dark_pixel_fraction_le_51": (0.0, 1.0),
            "black_clipping_fraction_le_5": (0.0, 1.0),
            "highlight_clipping_fraction_ge_250": (0.0, 1.0),
        }
        metric_errors = []
        for key, (low, high) in required_metrics.items():
            value = luminance.get(key)
            if not _is_finite_number(value) or not low <= float(value) <= high:
                metric_errors.append(f"{key}={value!r}")
        noise_value = noise.get("noise_estimate_proxy")
        if not _is_finite_number(noise_value) or not 0.0 <= float(noise_value) <= 255.0:
            metric_errors.append(f"noise_estimate_proxy={noise_value!r}")
        metrics_ok = not metric_errors
        record(
            "Measurement validity and ranges",
            metrics_ok,
            "Required luminance and noise-proxy measurements are finite and within expected 8-bit ranges."
            if metrics_ok else "Invalid measurement(s): " + ", ".join(metric_errors),
        )

        allowed = set(registered_algorithms)
        seen: set[str] = set()
        recommendation_errors: list[str] = []
        for item in recommendations:
            key = item.get("algorithm_key")
            score = item.get("priority_score")
            if key not in allowed:
                recommendation_errors.append(f"unregistered algorithm {key!r}")
            if key in seen:
                recommendation_errors.append(f"duplicate algorithm {key!r}")
            seen.add(key)
            if not _is_finite_number(score) or not 0 <= float(score) <= 100:
                recommendation_errors.append(f"invalid priority score for {key!r}: {score!r}")
        recs_ok = not recommendation_errors
        record(
            "Recommendation integrity",
            recs_ok,
            f"{len(recommendations)} recommendation(s); algorithm keys are registered, unique, "
            "and priority scores are within 0–100."
            if recs_ok else "; ".join(recommendation_errors),
        )

        classification_ok = (
            isinstance(classification.get("active_conditions"), list)
            and isinstance(classification.get("explanations"), dict)
            and isinstance(classification.get("exposure_interpretation"), str)
        )
        record(
            "Condition classification structure",
            classification_ok,
            "Condition list, explanations and exposure interpretation are present."
            if classification_ok else "One or more classification fields are missing or malformed.",
        )

        # Re-run the same image through ACP to test deterministic repeatability.
        repeated = build_processing_plan(image.copy())
        deterministic = (
            repeated.get("analysis") == analysis
            and repeated.get("classification") == classification
            and repeated.get("recommendations") == recommendations
        )
        record(
            "Deterministic repeatability",
            deterministic,
            "A second analysis of the same unchanged photograph produced identical "
            "measurements, classifications and recommendations."
            if deterministic else "Repeated analysis differed; investigate nondeterministic inputs or state.",
        )

    except Exception as exc:
        record("Validation execution", False, f"Validation error: {type(exc).__name__}: {exc}")

    passed = all(item["passed"] for item in checks)
    passed_count = sum(item["passed"] for item in checks)
    summary = (
        f"{passed_count}/{len(checks)} checks passed. "
        + ("ACP analysis passed the real-image integrity checks." if passed
           else "Review the failed checks before relying on this ACP analysis.")
    )
    return {"passed": passed, "checks": checks, "summary": summary}
