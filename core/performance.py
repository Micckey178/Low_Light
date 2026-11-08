
"""
ATHER - Performance Evaluation Engine
-------------------------------------

Benchmarks all registered classical image-enhancement
algorithms using consistent input images.

Measures:
- Processing time
- Brightness
- Contrast
- Entropy
- Dynamic range
- Estimated noise
- Image dimensions and memory

No AI, machine learning, neural networks or pretrained
models are used.
"""

import time

import cv2
import numpy as np

from core.enhancement import (
    ALGORITHMS,
    apply_enhancement
)


# ============================================================
# 1. IMAGE VALIDATION
# ============================================================

def validate_image(image):
    """Validate an 8-bit grayscale or colour image."""

    if image is None:
        raise ValueError("Image cannot be empty.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Image contains no pixel data.")

    if image.dtype != np.uint8:
        raise TypeError("Expected an 8-bit image.")

    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[2] in (1, 3, 4):
        return image

    raise ValueError("Unsupported image dimensions or channels.")


# ============================================================
# 2. GRAYSCALE CONVERSION
# ============================================================

def to_grayscale(image):
    """Convert grayscale, BGR or BGRA images to grayscale."""

    validate_image(image)

    if image.ndim == 2:
        return image

    if image.shape[2] == 1:
        return image[:, :, 0]

    if image.shape[2] == 3:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGRA2GRAY
    )


# ============================================================
# 3. ENTROPY
# ============================================================

def calculate_entropy(gray):
    """
    Calculate Shannon entropy.

    H = -sum(p(i) * log2(p(i)))

    Higher entropy indicates a more distributed
    grayscale intensity histogram. It does not
    automatically mean better visual quality.
    """

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    probabilities = histogram.ravel()

    total = probabilities.sum()

    if total == 0:
        return 0.0

    probabilities = probabilities / total

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy)


# ============================================================
# 4. APPROXIMATE NOISE ESTIMATION
# ============================================================

def estimate_noise(gray):
    """
    Estimate noise using a high-pass residual
    and median absolute deviation.

    This is an approximate statistical estimate.
    It is not a ground-truth sensor-noise measurement.
    """

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    residual = (
        gray.astype(np.float32)
        - blurred.astype(np.float32)
    )

    median = np.median(residual)

    mad = np.median(
        np.abs(residual - median)
    )

    return float(mad / 0.6745)


# ============================================================
# 5. IMAGE QUALITY AND PROPERTY METRICS
# ============================================================

def calculate_image_metrics(image):
    """
    Calculate objective image statistics.

    Returns:
        Dictionary containing image dimensions,
        brightness, contrast, entropy, dynamic range,
        estimated noise and memory consumption.
    """

    validate_image(image)

    gray = to_grayscale(image)

    minimum = int(np.min(gray))
    maximum = int(np.max(gray))

    return {
        "width": int(image.shape[1]),

        "height": int(image.shape[0]),

        "total_pixels": int(
            image.shape[0] * image.shape[1]
        ),

        "brightness": float(
            np.mean(gray)
        ),

        "contrast": float(
            np.std(gray)
        ),

        "minimum_intensity": minimum,

        "maximum_intensity": maximum,

        "dynamic_range": int(
            maximum - minimum
        ),

        "entropy": calculate_entropy(gray),

        "estimated_noise": estimate_noise(gray),

        "image_memory_kb": float(
            image.nbytes / 1024
        )
    }


# ============================================================
# 6. BENCHMARK ONE ALGORITHM
# ============================================================

def benchmark_algorithm(
    algorithm_key,
    original_image,
    gamma=0.6
):
    """
    Execute one enhancement algorithm.

    Measures enhancement execution time separately
    from image-metric calculation.

    The original input image is never modified.

    Returns:
        Dictionary containing:
        - Algorithm name and key
        - Processing time
        - Original metrics
        - Enhanced metrics
        - Metric changes
        - Enhanced image
    """

    validate_image(original_image)

    if algorithm_key not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm_key}. "
            f"Available algorithms: {list(ALGORITHMS.keys())}"
        )

    source = original_image.copy()

    start_time = time.perf_counter()

    enhanced_image = apply_enhancement(
        algorithm_key,
        source.copy(),
        gamma
    )

    end_time = time.perf_counter()

    processing_time_ms = (
        end_time - start_time
    ) * 1000.0

    # Validate the algorithm output.

    validate_image(enhanced_image)

    if enhanced_image.shape != source.shape:
        raise ValueError(
            f"{algorithm_key} changed the image dimensions "
            f"or channel count."
        )

    if enhanced_image.dtype != np.uint8:
        raise ValueError(
            f"{algorithm_key} did not return an 8-bit image."
        )

    if not np.all(np.isfinite(enhanced_image)):
        raise ValueError(
            f"{algorithm_key} returned invalid pixel values."
        )

    # Calculate metrics outside the timed section.

    original_metrics = calculate_image_metrics(
        source
    )

    enhanced_metrics = calculate_image_metrics(
        enhanced_image
    )

    brightness_change = (
        enhanced_metrics["brightness"]
        - original_metrics["brightness"]
    )

    contrast_change = (
        enhanced_metrics["contrast"]
        - original_metrics["contrast"]
    )

    entropy_change = (
        enhanced_metrics["entropy"]
        - original_metrics["entropy"]
    )

    noise_change = (
        enhanced_metrics["estimated_noise"]
        - original_metrics["estimated_noise"]
    )

    return {
        "algorithm": algorithm_key,

        "algorithm_name": ALGORITHMS[
            algorithm_key
        ]["name"],

        "processing_time_ms": float(
            processing_time_ms
        ),

        "original_metrics": original_metrics,

        "enhanced_metrics": enhanced_metrics,

        "brightness_change": float(
            brightness_change
        ),

        "contrast_change": float(
            contrast_change
        ),

        "entropy_change": float(
            entropy_change
        ),

        "noise_change": float(
            noise_change
        ),

        "enhanced_image_memory_kb": float(
            enhanced_image.nbytes / 1024
        ),

        "enhanced_image": enhanced_image
    }


# ============================================================
# 7. BENCHMARK ALL REGISTERED ALGORITHMS
# ============================================================

def benchmark_all_algorithms(
    original_image,
    gamma=0.6
):
    """
    Benchmark every algorithm registered in ALGORITHMS.

    All algorithms receive the same original image.

    Returns:
        Dictionary indexed by algorithm key.

    Example:
        results = benchmark_all_algorithms(image)

        results["gamma"]["processing_time_ms"]
        results["clahe"]["enhanced_metrics"]
    """

    validate_image(original_image)

    results = {}

    original_copy = original_image.copy()

    for algorithm_key in ALGORITHMS:

        result = benchmark_algorithm(
            algorithm_key,
            original_copy,
            gamma
        )

        results[algorithm_key] = result

    # Confirm that benchmarking did not modify the input.

    if not np.array_equal(
        original_image,
        original_copy
    ):
        raise RuntimeError(
            "Original image was modified during benchmarking."
        )

    return results


# ============================================================
# 8. CREATE A COMPARISON TABLE DATASET
# ============================================================

def create_benchmark_summary(results):
    """
    Convert benchmark results into a compact
    dataset suitable for GUI tables and graphs.
    """

    if not isinstance(results, dict):
        raise TypeError(
            "Benchmark results must be a dictionary."
        )

    summary = []

    for algorithm_key, result in results.items():

        original = result["original_metrics"]

        enhanced = result["enhanced_metrics"]

        summary.append({
            "algorithm": algorithm_key,

            "algorithm_name": result["algorithm_name"],

            "processing_time_ms": result[
                "processing_time_ms"
            ],

            "original_brightness": original[
                "brightness"
            ],

            "enhanced_brightness": enhanced[
                "brightness"
            ],

            "original_contrast": original[
                "contrast"
            ],

            "enhanced_contrast": enhanced[
                "contrast"
            ],

            "original_entropy": original[
                "entropy"
            ],

            "enhanced_entropy": enhanced[
                "entropy"
            ],

            "original_dynamic_range": original[
                "dynamic_range"
            ],

            "enhanced_dynamic_range": enhanced[
                "dynamic_range"
            ],

            "original_estimated_noise": original[
                "estimated_noise"
            ],

            "enhanced_estimated_noise": enhanced[
                "estimated_noise"
            ]
        })

    return summary
