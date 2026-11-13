"""
ATHER - Unified Validation Suite
================================
Runs the four existing validation phases using generated synthetic images.
No base photograph, camera, GUI, or external image path is required.

Run from the ATHER project root:
    python validate_ather.py

The script validates ACP planning, ACP Phase 2 regression safeguards,
classical processing components, and universal image evaluation.
It does not claim that an enhancement is objectively better, and it does
not calculate reference-based PSNR/SSIM without ground-truth images.
"""

from __future__ import annotations

import traceback
import unittest

import cv2
import numpy as np

from core.adaptive_engine import (
    ENGINE_VERSION as ACP_ENGINE_VERSION,
    build_processing_plan,
    validate_image,
)
from core.acp_validation import validate_real_image_plan
from core.advanced_processing import (
    MODULE_VERSION,
    classical_denoise,
    edge_preserving_sharpen,
    illumination_correction,
    preserve_original_chroma,
    apply_selected_components,
    get_component_descriptions,
)
from core.processing_evaluation import (
    ENGINE_VERSION as EVALUATION_ENGINE_VERSION,
    AVAILABLE_COMPONENTS,
    calculate_image_metrics,
    evaluate_components,
    evaluate_single_component,
    get_evaluation_engine_info,
)


ALLOWED_ALGORITHMS = {
    "gamma",
    "contrast_stretching",
    "histogram_equalization",
    "clahe",
    "adaptive_gamma",
    "ssr",
    "msr",
    "lime",
    "homomorphic",
}


def make_phase1_images() -> dict[str, np.ndarray]:
    """Synthetic image patterns used by the original Phase 1 validator."""
    rng = np.random.default_rng(20261109)
    height, width = 192, 256

    dark_texture = np.clip(
        24 + rng.normal(0, 7, (height, width)), 0, 255
    ).astype(np.uint8)
    bright = np.clip(
        218 + rng.normal(0, 9, (height, width)), 0, 255
    ).astype(np.uint8)
    low_contrast = np.clip(
        112 + rng.normal(0, 4, (height, width)), 0, 255
    ).astype(np.uint8)

    horizontal = np.linspace(12, 185, width, dtype=np.float32)
    gradient = np.tile(horizontal, (height, 1))
    uneven_illumination = np.clip(
        gradient + rng.normal(0, 5, (height, width)), 0, 255
    ).astype(np.uint8)
    noisy_dark = np.clip(
        38 + rng.normal(0, 17, (height, width)), 0, 255
    ).astype(np.uint8)
    normal = rng.integers(35, 221, (height, width), dtype=np.uint8)
    grayscale = np.tile(
        np.linspace(0, 255, width, dtype=np.uint8), (height, 1)
    )

    def to_bgr(gray: np.ndarray) -> np.ndarray:
        return np.repeat(gray[:, :, None], 3, axis=2)

    bgr_dark = to_bgr(dark_texture)
    bgra = np.dstack(
        [bgr_dark, np.full((height, width), 255, dtype=np.uint8)]
    )

    return {
        "dark_textured_bgr": bgr_dark,
        "bright_grayscale": bright,
        "low_contrast_grayscale": low_contrast,
        "uneven_illumination_grayscale": uneven_illumination,
        "noisy_dark_grayscale": noisy_dark,
        "normal_bgr": to_bgr(normal),
        "grayscale_gradient": grayscale,
        "bgra_with_alpha": bgra,
    }


def make_phase2_profiles() -> dict[str, np.ndarray]:
    """Synthetic profiles used by the original Phase 2 regression validator."""
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
        np.where(
            np.arange(512)[None, :, None] < 256, 20, 245
        ).astype(np.uint8),
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


def make_processing_test_image() -> np.ndarray:
    """Synthetic colour scene used by the original Phase 3.1 validator."""
    height, width = 240, 320
    y, x = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.clip(35 + x * 0.35, 0, 255).astype(np.uint8)
    image[:, :, 1] = np.clip(25 + y * 0.45, 0, 255).astype(np.uint8)
    image[:, :, 2] = np.clip(
        45 + (x + y) * 0.25, 0, 255
    ).astype(np.uint8)
    cv2.rectangle(image, (70, 55), (245, 185), (180, 210, 230), -1)
    cv2.circle(image, (155, 120), 38, (30, 80, 220), -1)
    return image


def make_evaluation_test_image() -> np.ndarray:
    """Synthetic textured BGR image used for Phase 3.2 evaluation."""
    rng = np.random.default_rng(3202)
    height, width = 240, 320
    y, x = np.mgrid[0:height, 0:width]
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:, :, 0] = 24 + x * 0.28
    base[:, :, 1] = 18 + y * 0.32
    base[:, :, 2] = 30 + (x + y) * 0.20
    noise = rng.normal(0, 3, base.shape)
    image = np.clip(base + noise, 0, 255).astype(np.uint8)
    cv2.rectangle(image, (55, 45), (265, 195), (125, 150, 175), -1)
    cv2.circle(image, (160, 120), 42, (35, 75, 180), -1)
    return image


def run_phase1() -> int:
    print("\n" + "=" * 72)
    print("PHASE 1 - ATHER-ACP BASE VALIDATION")
    print("=" * 72)

    images = make_phase1_images()
    checks = 0

    for name, image in images.items():
        original = image.copy()
        plan = build_processing_plan(image)

        assert plan["engine"] == "ATHER-ACP", f"{name}: wrong engine"
        assert plan["automatic_execution"] is False, f"{name}: auto execution enabled"
        assert plan["execution_mode"] == "recommendation_only", f"{name}: wrong mode"
        assert plan["analysis"]["image"]["width"] == image.shape[1]
        assert plan["analysis"]["image"]["height"] == image.shape[0]
        assert plan["analysis"]["image"]["color_mode"] in {"BGR", "BGRA", "grayscale"}
        assert isinstance(plan["classification"]["active_conditions"], list)
        assert len(plan["recommendations"]) <= 5
        assert plan["explanation"]
        assert plan["cautions"]
        np.testing.assert_array_equal(image, original)

        for item in plan["recommendations"]:
            assert item["algorithm_key"] in ALLOWED_ALGORITHMS
            assert 0 <= item["priority_score"] <= 100
            assert item["score_type"] == "heuristic_rule_priority_not_probability"
        checks += 1
        print(f"PASS: valid plan for {name}")

    plan = build_processing_plan(images["dark_textured_bgr"])
    assert "very_dark" in plan["classification"]["active_conditions"]
    print("PASS: dark image detected as very dark")
    checks += 1

    plan = build_processing_plan(images["bright_grayscale"])
    assert plan["analysis"]["luminance"]["mean"] > 180
    assert "very_dark" not in plan["classification"]["active_conditions"]
    print("PASS: bright image measured as bright")
    checks += 1

    gray_plan = build_processing_plan(images["grayscale_gradient"])
    bgra_plan = build_processing_plan(images["bgra_with_alpha"])
    assert gray_plan["analysis"]["image"]["color_mode"] == "grayscale"
    assert bgra_plan["analysis"]["image"]["color_mode"] == "BGRA"
    assert bgra_plan["analysis"]["image"]["channels"] == 4
    print("PASS: grayscale and BGRA inputs supported")
    checks += 1

    image = images["uneven_illumination_grayscale"]
    first, second = build_processing_plan(image), build_processing_plan(image)
    assert first["analysis"] == second["analysis"]
    assert first["classification"] == second["classification"]
    assert first["recommendations"] == second["recommendations"]
    print("PASS: deterministic analysis")
    checks += 1

    assert len(build_processing_plan(
        images["dark_textured_bgr"], max_recommendations=3
    )["recommendations"]) <= 3
    print("PASS: recommendation limit respected")
    checks += 1

    invalid_cases = [
        (np.zeros((32, 32), dtype=np.float32), TypeError),
        (np.zeros((32, 32, 2), dtype=np.uint8), ValueError),
        (np.zeros((0, 32), dtype=np.uint8), ValueError),
        (None, ValueError),
    ]
    for invalid, expected in invalid_cases:
        try:
            validate_image(invalid)
        except expected:
            pass
        else:
            raise AssertionError(f"validate_image did not reject {type(invalid)}")
    print("PASS: unsupported image data rejected")
    checks += 1

    rng = np.random.default_rng(41)
    height, width = 192, 256
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    bright_low_contrast = np.clip(
        222 + 12 * x + rng.normal(0, 4, (height, width)), 0, 255
    ).astype(np.uint8)
    plan = build_processing_plan(bright_low_contrast)
    assert plan["classification"]["exposure_interpretation"] == "bright_dominant"
    assert plan["recommendations"] == []
    assert any("withheld" in item.lower() for item in plan["cautions"])
    print("PASS: bright low-contrast scene is not forced to enhance")
    checks += 1

    rng = np.random.default_rng(42)
    dark_scene = np.clip(
        22 + rng.normal(0, 5, (192, 256)), 0, 255
    ).astype(np.uint8)
    plan = build_processing_plan(dark_scene)
    assert "very_dark" in plan["classification"]["active_conditions"]
    assert plan["classification"]["exposure_interpretation"] == "dark_dominant_intent_unknown"
    assert any("intentional" in item.lower() for item in plan["cautions"])
    print("PASS: dark-scene photographic intent remains unknown")
    checks += 1

    mixed = np.full((192, 256), 105, dtype=np.uint8)
    mixed[35:165, 15:105] = 12
    mixed[45:110, 175:245] = 252
    plan = build_processing_plan(mixed)
    assert "mixed_lighting" in plan["classification"]["active_conditions"]
    assert plan["classification"]["exposure_interpretation"] == "mixed_dark_and_bright_regions"
    assert any("both dark and bright" in item.lower() for item in plan["cautions"])
    print("PASS: mixed dark and bright regions disclosed")
    checks += 1

    try:
        build_processing_plan(images["normal_bgr"], max_recommendations=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid recommendation limit was not rejected")
    print("PASS: invalid recommendation limit rejected")
    checks += 1

    print(f"PHASE 1 PASSED ({checks} grouped checks)")
    return checks


def run_phase2() -> int:
    print("\n" + "=" * 72)
    print("PHASE 2 - ACP v0.3.0 REGRESSION VALIDATION")
    print("=" * 72)

    if ACP_ENGINE_VERSION != "0.3.0":
        raise AssertionError(
            f"Expected ACP 0.3.0, found {ACP_ENGINE_VERSION}"
        )

    profiles = make_phase2_profiles()
    plans = {}
    checks = 0

    for name, image in profiles.items():
        plan = build_processing_plan(image)
        plans[name] = plan
        validation = validate_real_image_plan(image, plan, ALLOWED_ALGORITHMS)
        if not validation["passed"]:
            raise AssertionError(
                f"{name}: {validation['summary']} {validation['checks']}"
            )
        assert plan.get("automatic_execution") is False, name
        assert plan == build_processing_plan(image.copy()), (
            f"{name}: repeated analysis was not deterministic"
        )

        for item in plan.get("recommendations", []):
            assert 0 <= item["priority_score"] <= 100
            assert item.get("score_breakdown"), f"{name}: missing score breakdown"
            assert item.get("parameter_guidance"), f"{name}: missing parameter guidance"
            assert item.get("parameter_guidance_status") == (
                "manual_starting_point_not_automatically_applied"
            )
        print(f"PASS: {name}: {validation['summary']}")
        checks += 1

    assert not plans["bright flat"]["recommendations"], (
        "Bright flat scene should not receive forced routine recommendations."
    )
    assert "dark_with_uneven_illumination" in plans[
        "dark + uneven illumination"
    ]["classification"]["compound_conditions"]
    assert "dark_with_elevated_noise" in plans[
        "dark + elevated noise proxy"
    ]["classification"]["compound_conditions"]
    assert "uneven_mixed_lighting" in plans[
        "mixed + uneven illumination"
    ]["classification"]["compound_conditions"]
    print("PASS: bright-scene safeguard and three compound-condition safeguards")
    checks += 1

    print("ACP engine version:", ACP_ENGINE_VERSION)
    print("No enhancement algorithm was executed by Phase 2.")
    print(f"PHASE 2 PASSED ({checks} grouped checks)")
    return checks


def run_phase3_1() -> int:
    print("\n" + "=" * 72)
    print("PHASE 3.1 - CLASSICAL PROCESSING COMPONENTS")
    print("=" * 72)

    image = make_processing_test_image()
    rng = np.random.default_rng(42)
    noisy = np.clip(
        image.astype(np.int16) + rng.normal(0, 12, image.shape),
        0, 255
    ).astype(np.uint8)
    checks = 0

    def check(name: str, condition: bool) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(name)
        print(f"PASS: {name}")
        checks += 1

    denoised = classical_denoise(noisy)
    check("Denoising output dimensions", denoised.shape == image.shape)
    check("Denoising output data type", denoised.dtype == np.uint8)
    check("Zero-strength denoising identity", np.array_equal(
        classical_denoise(image, 0), image
    ))

    sharpened = edge_preserving_sharpen(image)
    check("Sharpening output dimensions", sharpened.shape == image.shape)
    check("Zero-strength sharpening identity", np.array_equal(
        edge_preserving_sharpen(image, 0), image
    ))

    corrected = illumination_correction(image)
    check("Illumination output dimensions", corrected.shape == image.shape)
    check("Zero-strength illumination identity", np.array_equal(
        illumination_correction(image, 0), image
    ))

    chroma = preserve_original_chroma(image, corrected)
    check("Chroma output dimensions", chroma.shape == image.shape)
    check("Zero-strength chroma identity", np.array_equal(
        preserve_original_chroma(image, corrected, 0), corrected
    ))

    original_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    chroma_lab = cv2.cvtColor(chroma, cv2.COLOR_BGR2LAB)
    chroma_difference = np.abs(
        chroma_lab[:, :, 1:].astype(np.int16)
        - original_lab[:, :, 1:].astype(np.int16)
    )
    check(
        "Original chroma retained within rounding tolerance",
        int(chroma_difference.max()) <= 1,
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    check("Grayscale denoising", classical_denoise(gray).shape == gray.shape)
    check("Grayscale sharpening", edge_preserving_sharpen(gray).shape == gray.shape)
    check(
        "Grayscale illumination correction",
        illumination_correction(gray).shape == gray.shape,
    )

    pipeline, applied = apply_selected_components(
        image, denoise=True, illumination=True, sharpen=True
    )
    check("Explicit processing pipeline", pipeline.shape == image.shape)
    check(
        "Selected component reporting",
        applied == ["denoising", "illumination", "sharpening"],
    )
    check("Component descriptions", len(get_component_descriptions()) == 4)

    invalid_tests = [
        ("Invalid denoising strength", lambda: classical_denoise(image, 2)),
        ("Invalid sharpening amount", lambda: edge_preserving_sharpen(image, -1)),
        ("Invalid illumination gain", lambda: illumination_correction(image, max_gain=4)),
        ("Empty image", lambda: classical_denoise(np.array([], dtype=np.uint8))),
        ("Incorrect image data type", lambda: classical_denoise(image.astype(np.float32))),
        ("Mismatched chroma dimensions", lambda: preserve_original_chroma(image, image[:100, :100])),
    ]
    for name, function in invalid_tests:
        try:
            function()
        except (TypeError, ValueError):
            check(name, True)
        else:
            raise AssertionError(f"Expected rejection: {name}")

    print(f"Module version: {MODULE_VERSION}")
    print(f"PHASE 3.1 PASSED ({checks} checks)")
    return checks


def run_phase3_2() -> int:
    print("\n" + "=" * 72)
    print("PHASE 3.2 - UNIVERSAL IMAGE EVALUATION")
    print("=" * 72)

    image = make_evaluation_test_image()
    original_copy = image.copy()
    checks = 0

    def check(name: str, condition: bool) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(name)
        print(f"PASS: {name}")
        checks += 1

    metrics = calculate_image_metrics(image)
    check(
        "Original image metrics calculated",
        "brightness" in metrics and "contrast" in metrics and "entropy" in metrics,
    )

    for component in AVAILABLE_COMPONENTS:
        result = evaluate_single_component(image, component)
        check(
            f"{component} evaluation completed",
            result["final_image"].shape == image.shape,
        )
        check(
            f"{component} processing time recorded",
            result["total_processing_time_ms"] >= 0,
        )

    selected = ["denoising", "illumination", "sharpening"]
    combined = evaluate_components(image, selected)
    check("Multiple-component evaluation completed", len(combined["steps"]) == 3)
    check("Final image dimensions preserved", combined["final_image"].shape == image.shape)
    check("Final metrics available", "brightness" in combined["final_metrics"])
    check("Metric changes available", "contrast" in combined["final_changes"])
    check(
        "Total processing time available",
        combined["total_processing_time_ms"] >= 0,
    )

    # Confirm evaluation did not mutate the in-memory source image.
    unchanged = np.array_equal(image, original_copy)
    check("Original synthetic image remains unchanged", unchanged)

    info = get_evaluation_engine_info()
    check(
        "Engine reports no fixed image dependency",
        info["fixed_image_dependency"] is False,
    )

    print("\nSynthetic original-image metrics:")
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {float(value):.4f}")
        else:
            print(f"{key}: {value}")

    print("\nCombined processing results:")
    for step in combined["steps"]:
        print(
            f"{step['component']}: "
            f"{step['processing_time_ms']:.3f} ms"
        )
    print(
        "Total processing time: "
        f"{combined['total_processing_time_ms']:.3f} ms"
    )
    print(f"Evaluation engine version: {EVALUATION_ENGINE_VERSION}")
    print(f"PHASE 3.2 PASSED ({checks} checks)")
    return checks


def main() -> int:
    print("=" * 72)
    print("ATHER - UNIFIED AUTOMATED VALIDATION")
    print("Four phases | Synthetic images | No base photo required")
    print("=" * 72)

    phases = [
        ("Phase 1 - ACP base validation", run_phase1),
        ("Phase 2 - ACP regression validation", run_phase2),
        ("Phase 3.1 - Classical processing", run_phase3_1),
        ("Phase 3.2 - Universal image evaluation", run_phase3_2),
    ]

    passed = []
    failed = []

    for name, function in phases:
        try:
            count = function()
            passed.append((name, count))
        except Exception as exc:
            failed.append((name, exc))
            print(f"\nFAILED: {name}")
            print(f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            print("\nStopping after the first failed phase.")
            break

    print("\n" + "=" * 72)
    print("ATHER VALIDATION SUMMARY")
    print("=" * 72)
    for name, count in passed:
        print(f"PASS: {name} ({count} grouped checks)")
    for name, exc in failed:
        print(f"FAIL: {name} - {type(exc).__name__}: {exc}")

    if failed:
        print("\nVALIDATION FAILED. Keep the original validation scripts.")
        return 1

    print("\nALL FOUR VALIDATION PHASES PASSED.")
    print("No user-supplied base photograph was required.")
    print("Keep the original scripts until you have reviewed this result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
