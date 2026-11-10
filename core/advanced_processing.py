
"""
ATHER — Phase 3.1
Advanced Classical Image Processing Components

Strictly classical image processing:
- No AI
- No machine learning
- No neural networks
- No pretrained models
- No cloud processing

Input/output:
- OpenCV BGR uint8 images
- Grayscale uint8 images

Components:
1. Classical noise reduction
2. Edge-preserving sharpening
3. Illumination correction
4. Original chroma preservation
"""

from __future__ import annotations

import cv2
import numpy as np


MODULE_VERSION = "3.1.0"


def validate_image(image, name="image"):
    """Validate a grayscale or BGR uint8 image."""

    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")

    if image.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if image.dtype != np.uint8:
        raise TypeError(f"{name} must use uint8 pixel data.")

    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[2] == 3:
        return image

    raise ValueError(
        f"{name} must be grayscale or a three-channel BGR image."
    )


def validate_parameter(value, name, minimum, maximum):

    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric.") from exc

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}."
        )

    return value


# ============================================================
# 1. CLASSICAL NOISE REDUCTION
# ============================================================

def classical_denoise(
    image,
    strength=0.35,
    diameter=7
):
    """
    Bilateral filtering for edge-aware noise reduction.

    strength:
        0.0 = no processing
        1.0 = strongest configured filtering

    diameter:
        Odd integer between 3 and 15.
    """

    image = validate_image(image)

    strength = validate_parameter(
        strength, "strength", 0.0, 1.0
    )

    if isinstance(diameter, bool) or not isinstance(diameter, int):
        raise TypeError("diameter must be an integer.")

    if diameter < 3 or diameter > 15 or diameter % 2 == 0:
        raise ValueError(
            "diameter must be an odd integer between 3 and 15."
        )

    if strength == 0:
        return image.copy()

    sigma_color = 12.0 + 58.0 * strength
    sigma_space = 12.0 + 38.0 * strength

    result = cv2.bilateralFilter(
        image,
        diameter,
        sigma_color,
        sigma_space,
        borderType=cv2.BORDER_REFLECT101
    )

    return result


# ============================================================
# 2. EDGE-PRESERVING SHARPENING
# ============================================================

def edge_preserving_sharpen(
    image,
    amount=0.65,
    radius=1.2,
    threshold=5
):
    """
    Thresholded unsharp masking.

    amount:
        Sharpening strength from 0.0 to 2.0.

    radius:
        Gaussian blur radius from 0.3 to 4.0.

    threshold:
        Minimum detail difference from 0 to 40.
    """

    image = validate_image(image)

    amount = validate_parameter(
        amount, "amount", 0.0, 2.0
    )

    radius = validate_parameter(
        radius, "radius", 0.3, 4.0
    )

    if isinstance(threshold, bool) or not isinstance(threshold, int):
        raise TypeError("threshold must be an integer.")

    if not 0 <= threshold <= 40:
        raise ValueError("threshold must be between 0 and 40.")

    if amount == 0:
        return image.copy()

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=radius,
        sigmaY=radius,
        borderType=cv2.BORDER_REFLECT101
    )

    original_float = image.astype(np.float32)
    detail = original_float - blurred.astype(np.float32)

    if image.ndim == 2:

        detail_strength = np.abs(detail)

    else:

        luminance = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

        blurred_luminance = cv2.GaussianBlur(
            luminance,
            (0, 0),
            sigmaX=radius,
            sigmaY=radius,
            borderType=cv2.BORDER_REFLECT101
        )

        detail_strength = np.abs(
            luminance - blurred_luminance
        )

    if threshold > 0:

        edge_weight = np.clip(
            (detail_strength - threshold) /
            (3.0 * threshold),
            0.0,
            1.0
        )

    else:

        edge_weight = np.ones_like(
            detail_strength,
            dtype=np.float32
        )

    if image.ndim == 3:
        edge_weight = edge_weight[:, :, None]

    sharpened = (
        original_float +
        amount * edge_weight * detail
    )

    return np.clip(
        np.rint(sharpened),
        0,
        255
    ).astype(np.uint8)


# ============================================================
# 3. ADAPTIVE ILLUMINATION CORRECTION
# ============================================================

def illumination_correction(
    image,
    strength=0.45,
    blur_scale=0.12,
    max_gain=1.8
):
    """
    Estimate broad illumination variation using Gaussian smoothing.

    Correction is applied to luminance rather than independently
    modifying colour channels.

    strength:
        0.0 to 1.0

    blur_scale:
        0.03 to 0.35

    max_gain:
        1.0 to 3.0
    """

    image = validate_image(image)

    strength = validate_parameter(
        strength, "strength", 0.0, 1.0
    )

    blur_scale = validate_parameter(
        blur_scale, "blur_scale", 0.03, 0.35
    )

    max_gain = validate_parameter(
        max_gain, "max_gain", 1.0, 3.0
    )

    if strength == 0:
        return image.copy()

    if image.ndim == 2:

        luminance = image.astype(np.float32)
        lab = None

    else:

        lab = cv2.cvtColor(
            image, cv2.COLOR_BGR2LAB
        )

        luminance = lab[:, :, 0].astype(np.float32)

    height, width = luminance.shape

    sigma = max(
        3.0,
        min(height, width) * blur_scale
    )

    illumination = cv2.GaussianBlur(
        luminance,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101
    )

    reference = max(
        8.0,
        float(np.median(illumination))
    )

    safe_illumination = np.maximum(
        illumination, 8.0
    )

    gain = np.clip(
        reference / safe_illumination,
        1.0 / max_gain,
        max_gain
    )

    # Protect highlights from unnecessary amplification.
    highlight_protection = np.clip(
        (235.0 - luminance) / 55.0,
        0.0,
        1.0
    )

    effective_gain = (
        1.0 +
        strength *
        (gain - 1.0) *
        highlight_protection
    )

    corrected_luminance = np.clip(
        luminance * effective_gain,
        0,
        255
    )

    if lab is None:

        return np.clip(
            np.rint(corrected_luminance),
            0,
            255
        ).astype(np.uint8)

    output_lab = lab.copy()

    output_lab[:, :, 0] = np.clip(
        np.rint(corrected_luminance),
        0,
        255
    ).astype(np.uint8)

    return cv2.cvtColor(
        output_lab,
        cv2.COLOR_LAB2BGR
    )


# ============================================================
# 4. ORIGINAL CHROMA PRESERVATION
# ============================================================

def preserve_original_chroma(
    original_image,
    enhanced_image,
    strength=1.0
):
    """
    Preserve original colour information while retaining
    enhanced luminance.

    strength:
        0.0 = use enhanced image chroma
        1.0 = use original image chroma

    Both images must have identical dimensions and colour format.
    """

    original = validate_image(
        original_image, "original_image"
    )

    enhanced = validate_image(
        enhanced_image, "enhanced_image"
    )

    strength = validate_parameter(
        strength, "strength", 0.0, 1.0
    )

    if original.shape != enhanced.shape:
        raise ValueError(
            "Both images must have identical dimensions."
        )

    if original.ndim != enhanced.ndim:
        raise ValueError(
            "Both images must use the same colour format."
        )

    if strength == 0:
        return enhanced.copy()

    if original.ndim == 2:
        return enhanced.copy()

    original_lab = cv2.cvtColor(
        original, cv2.COLOR_BGR2LAB
    )

    enhanced_lab = cv2.cvtColor(
        enhanced, cv2.COLOR_BGR2LAB
    )

    result = enhanced_lab.copy()

    original_chroma = (
        original_lab[:, :, 1:].astype(np.float32)
    )

    enhanced_chroma = (
        enhanced_lab[:, :, 1:].astype(np.float32)
    )

    blended_chroma = (
        enhanced_chroma * (1.0 - strength) +
        original_chroma * strength
    )

    result[:, :, 1:] = np.clip(
        np.rint(blended_chroma),
        0,
        255
    ).astype(np.uint8)

    return cv2.cvtColor(
        result, cv2.COLOR_LAB2BGR
    )


# ============================================================
# COMPONENT DESCRIPTIONS
# ============================================================

def get_component_descriptions():

    return {
        "denoising": {
            "name": "Classical Edge-Aware Denoising",
            "method": "Bilateral Filtering",
            "purpose": (
                "Reduce noise while retaining major image edges."
            )
        },

        "sharpening": {
            "name": "Edge-Preserving Sharpening",
            "method": "Thresholded Unsharp Masking",
            "purpose": (
                "Improve meaningful details without strongly "
                "sharpening flat regions."
            )
        },

        "illumination": {
            "name": "Adaptive Illumination Correction",
            "method": (
                "Gaussian Illumination-Field Estimation"
            ),
            "purpose": (
                "Correct broad lighting unevenness while "
                "protecting highlights."
            )
        },

        "chroma": {
            "name": "Original-Chroma Preservation",
            "method": "CIELAB Colour-Space Processing",
            "purpose": (
                "Retain source colour information while "
                "using enhanced luminance."
            )
        }
    }


# ============================================================
# EXPLICIT TEST PIPELINE
# ============================================================

def apply_selected_components(
    image,
    *,
    denoise=False,
    sharpen=False,
    illumination=False,
    denoise_strength=0.35,
    sharpen_amount=0.65,
    illumination_strength=0.45
):
    """
    Apply only the components explicitly selected by the caller.

    This function is for independent Phase 3.1 testing.
    It is not yet connected to main.py or ACP.
    """

    output = validate_image(image).copy()

    applied = []

    if denoise:

        output = classical_denoise(
            output,
            strength=denoise_strength
        )

        applied.append("denoising")

    if illumination:

        output = illumination_correction(
            output,
            strength=illumination_strength
        )

        applied.append("illumination")

    if sharpen:

        output = edge_preserving_sharpen(
            output,
            amount=sharpen_amount
        )

        applied.append("sharpening")

    return output, applied
