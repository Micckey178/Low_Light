
"""
ATHER — Phase 3.2
Universal Classical Processing Evaluation Engine

Accepts any valid user-provided image.

Features:
- Individual or multiple component evaluation
- Per-component processing time
- Original and enhanced image metrics
- Before/after metric differences
- No fixed image paths
- No fixed image categories
- No AI, ML, neural networks, or pretrained models

Image convention:
OpenCV BGR uint8 or grayscale uint8.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from core.advanced_processing import (
    validate_image,
    classical_denoise,
    edge_preserving_sharpen,
    illumination_correction,
    preserve_original_chroma,
)


ENGINE_VERSION = "3.2.0"

AVAILABLE_COMPONENTS = (
    "denoising",
    "illumination",
    "sharpening",
    "chroma_preservation",
)


# ============================================================
# IMAGE METRICS
# ============================================================

def calculate_image_metrics(image):
    """
    Calculate descriptive image-quality measurements.

    Noise estimate is an approximate high-frequency residual
    proxy, not a ground-truth noise measurement.

    Entropy is measured in bits per pixel.
    """

    image = validate_image(image)

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image.copy()

    gray_float = gray.astype(np.float32)

    brightness = float(np.mean(gray_float))

    contrast = float(np.std(gray_float))

    p1, p99 = np.percentile(
        gray_float,
        [1, 99]
    )

    dynamic_range = float(p99 - p1)

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    ).flatten()

    probabilities = histogram / max(
        float(histogram.sum()),
        1.0
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = float(
        -np.sum(
            probabilities *
            np.log2(probabilities)
        )
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    sharpness = float(
        laplacian.var()
    )

    # Approximate high-frequency residual.
    # Texture and edges also contribute to this measurement.
    median_image = cv2.medianBlur(
        gray,
        3
    )

    residual = (
        gray.astype(np.float32) -
        median_image.astype(np.float32)
    )

    residual_median = float(
        np.median(residual)
    )

    noise_proxy = float(
        1.4826 *
        np.median(
            np.abs(residual - residual_median)
        )
    )

    metrics = {
        "brightness": brightness,
        "contrast": contrast,
        "dynamic_range": dynamic_range,
        "entropy": entropy,
        "sharpness": sharpness,
        "noise_proxy": noise_proxy,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "channels": (
            1 if image.ndim == 2 else 3
        ),
    }

    if image.ndim == 3:

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        metrics["mean_saturation"] = float(
            np.mean(hsv[:, :, 1])
        )

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        metrics["mean_chroma_a"] = float(
            np.mean(lab[:, :, 1])
        )

        metrics["mean_chroma_b"] = float(
            np.mean(lab[:, :, 2])
        )

    else:

        metrics["mean_saturation"] = None
        metrics["mean_chroma_a"] = None
        metrics["mean_chroma_b"] = None

    return metrics


# ============================================================
# METRIC COMPARISON
# ============================================================

def compare_metrics(original, processed):
    """
    Calculate absolute metric differences.

    Positive and negative changes are reported without
    automatically treating one direction as better.
    """

    comparison = {}

    for key, original_value in original.items():

        processed_value = processed.get(key)

        if (
            isinstance(original_value, (int, float))
            and isinstance(processed_value, (int, float))
        ):

            comparison[key] = float(
                processed_value - original_value
            )

        else:

            comparison[key] = None

    return comparison


# ============================================================
# COMPONENT PARAMETER HANDLING
# ============================================================

def _get_parameters(parameters, component):

    if parameters is None:
        return {}

    if not isinstance(parameters, dict):
        raise TypeError(
            "parameters must be a dictionary."
        )

    component_parameters = parameters.get(
        component,
        {}
    )

    if not isinstance(component_parameters, dict):
        raise TypeError(
            f"Parameters for {component} must be a dictionary."
        )

    return component_parameters.copy()


# ============================================================
# UNIVERSAL IMAGE EVALUATION
# ============================================================

def evaluate_components(
    original_image,
    selected_components,
    parameters=None,
):
    """
    Evaluate selected processing components on any supplied image.

    Parameters
    ----------
    original_image:
        User-provided OpenCV BGR or grayscale uint8 image.

    selected_components:
        A list or tuple containing one or more component names.

        Available:
            denoising
            illumination
            sharpening
            chroma_preservation

    parameters:
        Optional component-specific parameter dictionary.

    Returns
    -------
    Dictionary containing:
        original_metrics
        final_metrics
        final_image
        steps
        total_processing_time_ms
        warnings

    Images are returned as NumPy arrays.
    """

    original = validate_image(
        original_image
    ).copy()

    if not isinstance(
        selected_components,
        (list, tuple)
    ):
        raise TypeError(
            "selected_components must be a list or tuple."
        )

    if len(selected_components) == 0:
        raise ValueError(
            "Select at least one processing component."
        )

    if len(set(selected_components)) != len(
        selected_components
    ):
        raise ValueError(
            "Duplicate component selections are not allowed."
        )

    invalid = [
        component
        for component in selected_components
        if component not in AVAILABLE_COMPONENTS
    ]

    if invalid:
        raise ValueError(
            f"Unsupported components: {invalid}. "
            f"Available components: {AVAILABLE_COMPONENTS}"
        )

    if parameters is not None and not isinstance(
        parameters,
        dict
    ):
        raise TypeError(
            "parameters must be a dictionary."
        )

    original_metrics = calculate_image_metrics(
        original
    )

    current_image = original.copy()

    steps = []
    warnings = []

    total_start = time.perf_counter()

    for component in selected_components:

        component_parameters = _get_parameters(
            parameters,
            component
        )

        step_start = time.perf_counter()

        if component == "denoising":

            current_image = classical_denoise(
                current_image,
                **component_parameters
            )

        elif component == "illumination":

            current_image = illumination_correction(
                current_image,
                **component_parameters
            )

        elif component == "sharpening":

            current_image = edge_preserving_sharpen(
                current_image,
                **component_parameters
            )

        elif component == "chroma_preservation":

            if current_image.ndim == 2:

                warnings.append(
                    "Chroma preservation is unavailable "
                    "for grayscale images."
                )

            elif np.array_equal(
                current_image,
                original
            ):

                warnings.append(
                    "Chroma preservation was selected without "
                    "a preceding luminance-changing operation. "
                    "The result may be unchanged."
                )

            current_image = preserve_original_chroma(
                original,
                current_image,
                **component_parameters
            )

        elapsed_ms = (
            time.perf_counter() - step_start
        ) * 1000.0

        step_metrics = calculate_image_metrics(
            current_image
        )

        step_comparison = compare_metrics(
            original_metrics,
            step_metrics
        )

        steps.append({
            "component": component,
            "processing_time_ms": float(elapsed_ms),
            "metrics": step_metrics,
            "changes_from_original": step_comparison,
            "image": current_image.copy(),
        })

    total_time_ms = (
        time.perf_counter() - total_start
    ) * 1000.0

    final_metrics = calculate_image_metrics(
        current_image
    )

    final_comparison = compare_metrics(
        original_metrics,
        final_metrics
    )

    return {
        "engine_version": ENGINE_VERSION,

        "selected_components": list(
            selected_components
        ),

        "original_metrics": original_metrics,

        "final_metrics": final_metrics,

        "final_changes": final_comparison,

        "final_image": current_image.copy(),

        "steps": steps,

        "total_processing_time_ms": float(
            total_time_ms
        ),

        "warnings": warnings,

        "evaluation_notes": [
            "Noise proxy is an approximate high-frequency residual.",
            "Texture and edges can influence the noise proxy.",
            "Higher entropy does not automatically indicate better quality.",
            "Higher sharpness does not automatically indicate better detail.",
            "Metric changes are descriptive, not universal quality scores.",
            "PSNR and SSIM are not calculated without a valid reference image.",
        ],
    }


# ============================================================
# SINGLE COMPONENT EVALUATION
# ============================================================

def evaluate_single_component(
    original_image,
    component,
    parameters=None,
):
    """
    Convenience function for evaluating one component.
    """

    return evaluate_components(
        original_image,
        [component],
        parameters=parameters
    )


# ============================================================
# COMPONENT INFORMATION
# ============================================================

def get_evaluation_engine_info():

    return {
        "version": ENGINE_VERSION,
        "processing_type": "Classical image processing",
        "execution": "Local CPU",
        "fixed_image_dependency": False,
        "supports_user_images": True,
        "supports_grayscale": True,
        "supports_colour": True,
        "components": list(
            AVAILABLE_COMPONENTS
        ),
    }
