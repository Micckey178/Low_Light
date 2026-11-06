
import unittest
import numpy as np

from core.enhancement import (
    ALGORITHMS,
    apply_enhancement
)


class TestATHEREnhancement(unittest.TestCase):

    def setUp(self):

        # Create a simulated low-light color image.
        self.image = np.full(
            (120, 160, 3),
            45,
            dtype=np.uint8
        )

        # Add a few brighter regions.
        self.image[30:70, 40:90] = 65
        self.image[75:100, 100:140] = 85

        self.original_copy = self.image.copy()

    def test_all_algorithms_are_available(self):

        expected = {
            "gamma",
            "contrast_stretching",
            "histogram_equalization",
            "clahe"
        }

        self.assertEqual(
            set(ALGORITHMS.keys()),
            expected
        )

    def test_all_algorithms_return_valid_images(self):

        for key in ALGORITHMS:

            with self.subTest(algorithm=key):

                result = apply_enhancement(
                    key,
                    self.image
                )

                self.assertEqual(
                    result.shape,
                    self.image.shape
                )

                self.assertEqual(
                    result.dtype,
                    np.uint8
                )

                self.assertTrue(
                    np.isfinite(result).all()
                )

    def test_gamma_correction_brightens_image(self):

        result = apply_enhancement(
            "gamma",
            self.image,
            0.6
        )

        self.assertGreater(
            np.mean(result),
            np.mean(self.image)
        )

    def test_original_image_is_not_modified(self):

        for key in ALGORITHMS:

            apply_enhancement(
                key,
                self.image
            )

            np.testing.assert_array_equal(
                self.image,
                self.original_copy
            )

    def test_invalid_algorithm_is_rejected(self):

        with self.assertRaises(ValueError):

            apply_enhancement(
                "unknown_algorithm",
                self.image
            )

    def test_invalid_gamma_is_rejected(self):

        with self.assertRaises(ValueError):

            apply_enhancement(
                "gamma",
                self.image,
                0
            )


if __name__ == "__main__":

    unittest.main(verbosity=2)
