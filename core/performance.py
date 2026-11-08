
"""
ATHER - Performance Evaluation Engine

Measures image quality characteristics and algorithm
execution time using classical image-processing methods.
"""

import time
import cv2
import numpy as np

from core.enhancement import apply_enhancement


def validate_image(image):
    """Validate an input image."""

    if image is None:
        raise ValueError("Image cannot be empty.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Image contains no pixel data.")

    if image.dtype != np.uint8:
        raise TypeError("Expected an 8-bit image.")

    if len(image.shape) not in (2, 3):
        raise ValueError("Unsupported image dimensions.")

    return image


def to_grayscale(image):
    """Convert a color image to grayscale."""

    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def calculate_entropy(gray):
    """
    Calculate Shannon entropy.

    H = -sum(p(i) * log2(p(i)))
    """

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    probabilities = histogram.ravel()
    probabilities = probabilities / probabilities.sum()

    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy)


def estimate_noise(gray):
    """
    Estimate noise using a high-pass residual and
    median absolute deviation.

    This is an approximate estimate, not a ground-truth
    measurement of sensor noise.
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


def calculate_image_metrics(image):
    """
    Calculate objective image statistics.

    Returns a dictionary of measurable characteristics.
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
        "brightness": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
        "minimum_intensity": minimum,
        "maximum_intensity": maximum,
        "dynamic_range": int(maximum - minimum),
        "entropy": calculate_entropy(gray),
        "estimated_noise": estimate_noise(gray),
        "image_memory_kb": float(
            image.nbytes / 1024
        )
    }


def benchmark_algorithm(
    algorithm_key,
    original_image,
    gamma=0.6
):
    """
    Execute one enhancement algorithm and measure
    its processing time and image statistics.

    The original image is never modified.
    """

    validate_image(original_image)

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
    ) * 1000

    original_metrics = calculate_image_metrics(
        source
    )

    enhanced_metrics = calculate_image_metrics(
        enhanced_image
    )

    return {
        "algorithm": algorithm_key,
        "processing_time_ms": float(
            processing_time_ms
        ),
        "original_metrics": original_metrics,
        "enhanced_metrics": enhanced_metrics,
        "brightness_change": float(
            enhanced_metrics["brightness"]
            - original_metrics["brightness"]
        ),
        "contrast_change": float(
            enhanced_metrics["contrast"]
            - original_metrics["contrast"]
        ),
        "entropy_change": float(
            enhanced_metrics["entropy"]
            - original_metrics["entropy"]
        ),
        "enhanced_image_memory_kb": float(
            enhanced_image.nbytes / 1024
        )
    }
