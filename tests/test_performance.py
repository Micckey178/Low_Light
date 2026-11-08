
import unittest
import numpy as np

from core.performance import (
    calculate_image_metrics,
    benchmark_algorithm
)


class TestATHERPerformance(unittest.TestCase):

    def setUp(self):

        self.image = np.full(
            (120, 160, 3),
            45,
            dtype=np.uint8
        )

        self.image[30:70, 40:90] = 65
        self.image[75:100, 100:140] = 85

        self.original_copy = self.image.copy()

    def test_image_metrics(self):

        metrics = calculate_image_metrics(
            self.image
        )

        self.assertEqual(
            metrics["width"], 160
        )

        self.assertEqual(
            metrics["height"], 120
        )

        self.assertEqual(
            metrics["total_pixels"], 19200
        )

        self.assertGreaterEqual(
            metrics["entropy"], 0
        )

    def test_benchmark_gamma(self):

        result = benchmark_algorithm(
            "gamma",
            self.image,
            0.6
        )

        self.assertGreaterEqual(
            result["processing_time_ms"], 0
        )

        self.assertGreater(
            result["enhanced_metrics"]["brightness"],
            result["original_metrics"]["brightness"]
        )

    def test_benchmark_all_algorithms(self):

        algorithms = [
            "gamma",
            "contrast_stretching",
            "histogram_equalization",
            "clahe"
        ]

        for algorithm in algorithms:

            with self.subTest(algorithm=algorithm):

                result = benchmark_algorithm(
                    algorithm,
                    self.image
                )

                self.assertEqual(
                    result["algorithm"],
                    algorithm
                )

                self.assertIn(
                    "processing_time_ms",
                    result
                )

                self.assertIn(
                    "entropy",
                    result["enhanced_metrics"]
                )

    def test_original_image_unchanged(self):

        benchmark_algorithm(
            "gamma",
            self.image
        )

        np.testing.assert_array_equal(
            self.image,
            self.original_copy
        )


if __name__ == "__main__":

    unittest.main(verbosity=2)
