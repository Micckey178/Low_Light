
"""
ATHER - Image Quality and Performance Evaluation
Phase 2: Repeatable Benchmarking and Improved Metrics

Classical image processing only.
No AI, ML, neural networks or pretrained models.
"""

import time

import cv2
import numpy as np

from core.enhancement import (
    ALGORITHMS,
    apply_enhancement
)


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image):
    """Validate uint8 grayscale, BGR or BGRA images."""

    if image is None:
        raise ValueError("Input image is empty or None.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.dtype != np.uint8:
        raise TypeError("Only uint8 images are supported.")

    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[2] in (1, 3, 4):
        return image

    raise ValueError(
        "Unsupported image format. Use grayscale, BGR or BGRA."
    )


def _processing_image(image):
    """
    Prepare image for enhancement.

    BGRA images are converted to BGR for processing.
    Alpha is preserved separately.
    """

    validate_image(image)

    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0], None

    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3].copy(), image[:, :, 3].copy()

    return image.copy(), None


def _restore_alpha(enhanced, alpha):
    """Reattach original alpha channel when necessary."""

    if alpha is None:
        return enhanced

    return np.dstack((enhanced, alpha))


# ============================================================
# GRAYSCALE CONVERSION
# ============================================================

def to_grayscale(image):
    """Convert supported images to grayscale."""

    validate_image(image)

    if image.ndim == 2:
        return image.copy()

    if image.shape[2] == 1:
        return image[:, :, 0].copy()

    if image.shape[2] == 4:
        image = image[:, :, :3]

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


# ============================================================
# IMAGE METRICS
# ============================================================

def calculate_entropy(image):
    """
    Calculate grayscale Shannon entropy.

    Higher entropy indicates a more distributed intensity
    histogram, not necessarily better perceptual quality.
    """

    gray = to_grayscale(image)

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    ).flatten()

    total = float(np.sum(histogram))

    if total <= 0:
        return 0.0

    probabilities = histogram / total

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy)


def estimate_noise(image):
    """
    Approximate noise estimation using a median-filter residual.

    This is a rough estimate. Fine texture and edges may also
    contribute to the residual, so it must not be interpreted
    as a laboratory-grade noise measurement.
    """

    gray = to_grayscale(image)

    median = cv2.medianBlur(gray, 3)

    residual = (
        gray.astype(np.float32)
        - median.astype(np.float32)
    )

    median_residual = float(
        np.median(residual)
    )

    mad = float(
        np.median(
            np.abs(residual - median_residual)
        )
    )

    estimate = 1.4826 * mad

    return float(max(0.0, estimate))


def _calculate_clipping(image):
    """Calculate approximate black and highlight clipping."""

    gray = to_grayscale(image)

    total = float(gray.size)

    if total <= 0:
        return 0.0, 0.0

    black = float(
        np.count_nonzero(gray <= 5)
    ) / total * 100.0

    highlights = float(
        np.count_nonzero(gray >= 250)
    ) / total * 100.0

    return black, highlights


def calculate_image_metrics(image):
    """
    Calculate image properties and no-reference metrics.

    Existing metric keys are retained for report compatibility.
    """

    validate_image(image)

    gray = to_grayscale(image)

    height, width = gray.shape

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    minimum = int(np.min(gray))
    maximum = int(np.max(gray))

    dynamic_range = int(maximum - minimum)

    entropy = calculate_entropy(gray)

    estimated_noise = estimate_noise(gray)

    black_percentage, highlight_percentage = (
        _calculate_clipping(gray)
    )

    memory_kb = float(
        image.nbytes / 1024.0
    )

    return {
        "width": int(width),
        "height": int(height),
        "total_pixels": int(width * height),

        "brightness": brightness,
        "contrast": contrast,
        "dynamic_range": dynamic_range,
        "entropy": entropy,

        "estimated_noise": estimated_noise,

        "black_pixel_percentage": black_percentage,
        "highlight_pixel_percentage": highlight_percentage,

        "image_memory_kb": memory_kb
    }


# ============================================================
# METRIC COMPARISON
# ============================================================

def _calculate_changes(original_metrics, enhanced_metrics):
    """Calculate absolute and percentage metric changes."""

    changes = {}

    keys = (
        "brightness",
        "contrast",
        "dynamic_range",
        "entropy",
        "estimated_noise",
        "black_pixel_percentage",
        "highlight_pixel_percentage"
    )

    for key in keys:
        original_value = float(
            original_metrics.get(key, 0.0)
        )

        enhanced_value = float(
            enhanced_metrics.get(key, 0.0)
        )

        difference = enhanced_value - original_value

        if abs(original_value) > 1e-10:
            percentage = (
                difference / abs(original_value)
            ) * 100.0
        else:
            percentage = 0.0

        changes[key] = {
            "original": original_value,
            "enhanced": enhanced_value,
            "difference": float(difference),
            "percentage_change": float(percentage)
        }

    return changes


# ============================================================
# BENCHMARK ONE ALGORITHM
# ============================================================

def benchmark_algorithm(
    algorithm_key,
    original_image,
    gamma=0.6,
    repeats=3
):
    """
    Benchmark one enhancement algorithm.

    Uses repeated measurements and reports median processing
    time as the primary result.

    The original and enhanced metric calculations are outside
    the timed section.
    """

    validate_image(original_image)

    if algorithm_key not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm_key}"
        )

    repeats = int(repeats)

    if repeats < 1:
        raise ValueError(
            "Benchmark repeats must be at least 1."
        )

    working_image, alpha = _processing_image(
        original_image
    )

    original_metrics = calculate_image_metrics(
        original_image
    )

    # Warm-up run: not included in reported timing.
    apply_enhancement(
        algorithm_key,
        working_image,
        gamma=gamma
    )

    processing_times = []

    enhanced = None

    for _ in range(repeats):
        start = time.perf_counter()

        current_result = apply_enhancement(
            algorithm_key,
            working_image,
            gamma=gamma
        )

        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000.0

        processing_times.append(
            float(elapsed_ms)
        )

        enhanced = current_result

    if enhanced is None:
        raise RuntimeError(
            "Benchmark did not produce an enhanced image."
        )

    enhanced = _restore_alpha(
        enhanced,
        alpha
    )

    enhanced_metrics = calculate_image_metrics(
        enhanced
    )

    median_time = float(
        np.median(processing_times)
    )

    mean_time = float(
        np.mean(processing_times)
    )

    standard_deviation = float(
        np.std(processing_times)
    )

    minimum_time = float(
        np.min(processing_times)
    )

    maximum_time = float(
        np.max(processing_times)
    )

    changes = _calculate_changes(
        original_metrics,
        enhanced_metrics
    )

    return {
        "algorithm": algorithm_key,

        "algorithm_name": ALGORITHMS[
            algorithm_key
        ]["name"],

        # Existing report/API field:
        "processing_time_ms": median_time,

        # Additional repeatability information:
        "processing_time_mean_ms": mean_time,
        "processing_time_std_ms": standard_deviation,
        "processing_time_min_ms": minimum_time,
        "processing_time_max_ms": maximum_time,
        "benchmark_repeats": repeats,

        "original_metrics": original_metrics,
        "enhanced_metrics": enhanced_metrics,

        "changes": changes,

        "enhanced_image": enhanced,

        "image_memory_kb": float(
            enhanced.nbytes / 1024.0
        )
    }


# ============================================================
# BENCHMARK ALL ALGORITHMS
# ============================================================

def benchmark_all_algorithms(
    original_image,
    gamma=0.6,
    repeats=3
):
    """
    Benchmark every registered algorithm.

    Each algorithm receives an independent copy of the original
    processing image. The input image is never modified.
    """

    validate_image(original_image)

    original_copy = original_image.copy()

    results = {}

    for algorithm_key in ALGORITHMS:

        result = benchmark_algorithm(
            algorithm_key=algorithm_key,
            original_image=original_copy.copy(),
            gamma=gamma,
            repeats=repeats
        )

        results[algorithm_key] = result

    if not np.array_equal(
        original_image,
        original_copy
    ):
        raise RuntimeError(
            "An algorithm unexpectedly modified the input image."
        )

    return results


# ============================================================
# BENCHMARK SUMMARY
# ============================================================

def create_benchmark_summary(results):
    """
    Create compact benchmark rows for the GUI and report.

    Keeps the commonly used summary fields and adds timing
    consistency information.
    """

    if not isinstance(results, dict):
        raise TypeError(
            "Benchmark results must be a dictionary."
        )

    summary = []

    for algorithm_key, result in results.items():

        if not isinstance(result, dict):
            continue

        original_metrics = result.get(
            "original_metrics",
            {}
        )

        enhanced_metrics = result.get(
            "enhanced_metrics",
            {}
        )

        summary.append({
            "algorithm": algorithm_key,

            "algorithm_name": result.get(
                "algorithm_name",
                ALGORITHMS.get(
                    algorithm_key,
                    {}
                ).get("name", algorithm_key)
            ),

            "processing_time_ms": result.get(
                "processing_time_ms",
                0.0
            ),

            "processing_time_std_ms": result.get(
                "processing_time_std_ms",
                0.0
            ),

            "brightness": enhanced_metrics.get(
                "brightness",
                0.0
            ),

            "contrast": enhanced_metrics.get(
                "contrast",
                0.0
            ),

            "entropy": enhanced_metrics.get(
                "entropy",
                0.0
            ),

            "dynamic_range": enhanced_metrics.get(
                "dynamic_range",
                0
            ),

            "estimated_noise": enhanced_metrics.get(
                "estimated_noise",
                0.0
            ),

            "black_pixel_percentage": enhanced_metrics.get(
                "black_pixel_percentage",
                0.0
            ),

            "highlight_pixel_percentage": enhanced_metrics.get(
                "highlight_pixel_percentage",
                0.0
            ),

            "original_brightness": original_metrics.get(
                "brightness",
                0.0
            ),

            "original_contrast": original_metrics.get(
                "contrast",
                0.0
            )
        })

    return summary
