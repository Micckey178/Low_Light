"""
ATHER-ACP automated validation suite
------------------------------------
Run from the ATHER project root:

    python validate_acp.py

This script creates its own synthetic images. It does not require a fixed
photograph, image file, camera, or GUI. It tests the ACP analysis and
recommendation engine only; it does not run enhancement algorithms.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.adaptive_engine import (
    build_processing_plan,
    validate_image,
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


def make_test_images() -> dict[str, np.ndarray]:
    """Create repeatable test patterns with different image characteristics."""
    rng = np.random.default_rng(20261109)
    height, width = 192, 256

    # Very dark, but with visible texture and small intensity variations.
    dark_texture = np.clip(
        24 + rng.normal(0, 7, (height, width)), 0, 255
    ).astype(np.uint8)

    # Bright image with a small amount of texture.
    bright = np.clip(
        218 + rng.normal(0, 9, (height, width)), 0, 255
    ).astype(np.uint8)

    # Narrow intensity distribution: intentionally low global contrast.
    low_contrast = np.clip(
        112 + rng.normal(0, 4, (height, width)), 0, 255
    ).astype(np.uint8)

    # Spatially varying illumination with texture overlaid.
    horizontal = np.linspace(12, 185, width, dtype=np.float32)
    gradient = np.tile(horizontal, (height, 1))
    uneven_illumination = np.clip(
        gradient + rng.normal(0, 5, (height, width)), 0, 255
    ).astype(np.uint8)

    # Dark base with stronger random noise.
    noisy_dark = np.clip(
        38 + rng.normal(0, 17, (height, width)), 0, 255
    ).astype(np.uint8)

    # Broad tonal range with deterministic texture.
    normal = rng.integers(35, 221, (height, width), dtype=np.uint8)

    # Grayscale gradient.
    grayscale = np.tile(
        np.linspace(0, 255, width, dtype=np.uint8), (height, 1)
    )

    # Convert selected patterns to BGR and add an alpha channel for BGRA.
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


class TestAetherACP(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.images = make_test_images()

    def test_all_synthetic_image_types_produce_valid_plans(self) -> None:
        for name, image in self.images.items():
            with self.subTest(image=name):
                original = image.copy()
                plan = build_processing_plan(image)

                self.assertEqual(plan["engine"], "ATHER-ACP")
                self.assertFalse(plan["automatic_execution"])
                self.assertEqual(
                    plan["execution_mode"], "recommendation_only"
                )
                self.assertEqual(
                    plan["analysis"]["image"]["width"], image.shape[1]
                )
                self.assertEqual(
                    plan["analysis"]["image"]["height"], image.shape[0]
                )
                self.assertIn(
                    plan["analysis"]["image"]["color_mode"],
                    {"BGR", "BGRA", "grayscale"},
                )
                self.assertIsInstance(
                    plan["classification"]["active_conditions"], list
                )
                # An empty shortlist is valid: ACP must not invent a need to
                # enhance an image whose measurements do not justify it.
                self.assertLessEqual(len(plan["recommendations"]), 5)
                self.assertTrue(plan["explanation"])
                self.assertTrue(plan["cautions"])
                np.testing.assert_array_equal(image, original)

                for item in plan["recommendations"]:
                    self.assertIn(item["algorithm_key"], ALLOWED_ALGORITHMS)
                    self.assertGreaterEqual(item["priority_score"], 0)
                    self.assertLessEqual(item["priority_score"], 100)
                    self.assertEqual(
                        item["score_type"],
                        "heuristic_rule_priority_not_probability",
                    )

    def test_dark_image_is_detected_as_very_dark(self) -> None:
        plan = build_processing_plan(self.images["dark_textured_bgr"])
        self.assertIn(
            "very_dark", plan["classification"]["active_conditions"]
        )

    def test_bright_image_is_measured_as_bright(self) -> None:
        plan = build_processing_plan(self.images["bright_grayscale"])
        mean_luminance = plan["analysis"]["luminance"]["mean"]
        self.assertGreater(mean_luminance, 180)
        self.assertNotIn(
            "very_dark", plan["classification"]["active_conditions"]
        )

    def test_grayscale_and_bgra_inputs_are_supported(self) -> None:
        gray_plan = build_processing_plan(self.images["grayscale_gradient"])
        bgra_plan = build_processing_plan(self.images["bgra_with_alpha"])

        self.assertEqual(
            gray_plan["analysis"]["image"]["color_mode"], "grayscale"
        )
        self.assertEqual(
            bgra_plan["analysis"]["image"]["color_mode"], "BGRA"
        )
        self.assertEqual(
            bgra_plan["analysis"]["image"]["channels"], 4
        )

    def test_analysis_is_deterministic_for_same_input(self) -> None:
        image = self.images["uneven_illumination_grayscale"]
        first = build_processing_plan(image)
        second = build_processing_plan(image)

        self.assertEqual(first["analysis"], second["analysis"])
        self.assertEqual(first["classification"], second["classification"])
        self.assertEqual(first["recommendations"], second["recommendations"])

    def test_recommendation_limit_is_respected(self) -> None:
        image = self.images["dark_textured_bgr"]
        plan = build_processing_plan(image, max_recommendations=3)
        self.assertLessEqual(len(plan["recommendations"]), 3)

    def test_input_validation_rejects_unsupported_data(self) -> None:
        with self.assertRaises(TypeError):
            validate_image(np.zeros((32, 32), dtype=np.float32))

        with self.assertRaises(ValueError):
            validate_image(np.zeros((32, 32, 2), dtype=np.uint8))

        with self.assertRaises(ValueError):
            validate_image(np.zeros((0, 32), dtype=np.uint8))

        with self.assertRaises(ValueError):
            validate_image(None)


    def test_bright_low_contrast_scene_is_not_forced_to_enhance(self) -> None:
        rng = np.random.default_rng(41)
        height, width = 192, 256
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
        image = np.clip(
            222 + 12 * x + rng.normal(0, 4, (height, width)), 0, 255
        ).astype(np.uint8)
        plan = build_processing_plan(image)
        self.assertEqual(
            plan["classification"]["exposure_interpretation"],
            "bright_dominant",
        )
        self.assertEqual(plan["recommendations"], [])
        self.assertTrue(any("withheld" in item.lower() for item in plan["cautions"]))

    def test_dark_scene_reports_unknown_photographic_intent(self) -> None:
        rng = np.random.default_rng(42)
        image = np.clip(
            22 + rng.normal(0, 5, (192, 256)), 0, 255
        ).astype(np.uint8)
        plan = build_processing_plan(image)
        self.assertIn("very_dark", plan["classification"]["active_conditions"])
        self.assertEqual(
            plan["classification"]["exposure_interpretation"],
            "dark_dominant_intent_unknown",
        )
        self.assertTrue(any("intentional" in item.lower() for item in plan["cautions"]))

    def test_mixed_dark_and_bright_regions_are_disclosed(self) -> None:
        image = np.full((192, 256), 105, dtype=np.uint8)
        image[35:165, 15:105] = 12
        image[45:110, 175:245] = 252
        plan = build_processing_plan(image)
        self.assertIn("mixed_lighting", plan["classification"]["active_conditions"])
        self.assertEqual(
            plan["classification"]["exposure_interpretation"],
            "mixed_dark_and_bright_regions",
        )
        self.assertTrue(any("both dark and bright" in item.lower() for item in plan["cautions"]))

    def test_invalid_recommendation_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_processing_plan(
                self.images["normal_bgr"], max_recommendations=0
            )


if __name__ == "__main__":
    print("=" * 68)
    print("ATHER-ACP AUTOMATED VALIDATION")
    print("Synthetic tests are generated in memory; no fixed photo is required.")
    print("Includes bright-scene, dark-intent ambiguity, and mixed-lighting safeguards.")
    print("=" * 68)
    unittest.main(verbosity=2)
