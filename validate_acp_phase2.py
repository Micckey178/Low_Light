"""Regression checks for ATHER-ACP Phase 2 (v0.3.0).
Run from the ATHER project root with its virtual environment activated:
    python validate_acp_phase2.py
"""
from __future__ import annotations

import numpy as np

from core.adaptive_engine import ENGINE_VERSION, build_processing_plan
from core.acp_validation import validate_real_image_plan

ALGORITHMS = {
    "gamma", "adaptive_gamma", "contrast_stretching",
    "histogram_equalization", "clahe", "ssr", "msr", "lime",
    "homomorphic",
}


def make_profiles() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2026)
    dark_flat = np.full((256, 256, 3), 28, dtype=np.uint8)
    bright_flat = np.full((256, 256, 3), 220, dtype=np.uint8)
    dark_uneven = np.broadcast_to(
        np.linspace(0, 120, 512, dtype=np.uint8)[None, :, None],
        (512, 512, 3),
    ).copy()
    dark_noisy = np.clip(
        35 + rng.normal(0, 20, (256, 256, 3)), 0, 255
    ).astype(np.uint8)
    low_contrast = np.full((256, 256, 3), 110, dtype=np.uint8)
    mixed_uneven = np.broadcast_to(
        np.where(np.arange(512)[None, :, None] < 256, 20, 245).astype(np.uint8),
        (512, 512, 3),
    ).copy()
    return {
        "dark flat": dark_flat,
        "bright flat": bright_flat,
        "dark + uneven illumination": dark_uneven,
        "dark + elevated noise proxy": dark_noisy,
        "low contrast": low_contrast,
        "mixed + uneven illumination": mixed_uneven,
    }


def main() -> int:
    if ENGINE_VERSION != "0.3.0":
        raise AssertionError(f"Expected ACP 0.3.0, found {ENGINE_VERSION}")

    profiles = make_profiles()
    plans = {}
    for name, image in profiles.items():
        plan = build_processing_plan(image)
        plans[name] = plan
        validation = validate_real_image_plan(image, plan, ALGORITHMS)
        if not validation["passed"]:
            raise AssertionError(f"{name}: {validation['summary']} {validation['checks']}")
        if plan.get("automatic_execution") is not False:
            raise AssertionError(f"{name}: ACP must remain recommendation-only")
        if plan != build_processing_plan(image.copy()):
            raise AssertionError(f"{name}: repeated analysis was not deterministic")
        for item in plan.get("recommendations", []):
            if not 0 <= item["priority_score"] <= 100:
                raise AssertionError(f"{name}: priority score outside 0–100")
            if not item.get("score_breakdown"):
                raise AssertionError(f"{name}: missing score breakdown")
            if not item.get("parameter_guidance"):
                raise AssertionError(f"{name}: missing parameter guidance")
            if item.get("parameter_guidance_status") != "manual_starting_point_not_automatically_applied":
                raise AssertionError(f"{name}: parameter guidance must remain manual")
        print(f"PASS — {name}: {validation['summary']}")

    assert not plans["bright flat"]["recommendations"], (
        "Bright flat scene should not receive forced routine recommendations."
    )
    assert "dark_with_uneven_illumination" in plans["dark + uneven illumination"]["classification"]["compound_conditions"]
    assert "dark_with_elevated_noise" in plans["dark + elevated noise proxy"]["classification"]["compound_conditions"]
    assert "uneven_mixed_lighting" in plans["mixed + uneven illumination"]["classification"]["compound_conditions"]

    print("\nAll ATHER-ACP Phase 2 regression checks passed.")
    print("ACP engine version:", ENGINE_VERSION)
    print("No enhancement algorithm was executed by these tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
