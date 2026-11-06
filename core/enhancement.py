
"""
ATHER - Classical Image Enhancement Engine

Algorithms:
1. Gamma Correction
2. Contrast Stretching
3. Histogram Equalization
4. CLAHE

All algorithms preserve the input image and return a new uint8 image.
"""

import cv2
import numpy as np


# ---------------------------------------------------------
# COMMON VALIDATION
# ---------------------------------------------------------

def validate_image(image):
    """Validate an image before enhancement."""

    if image is None:
        raise ValueError("Input image cannot be empty.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Input image contains no pixel data.")

    if image.dtype != np.uint8:
        raise TypeError(
            "Expected an 8-bit image with dtype uint8."
        )

    if len(image.shape) not in (2, 3):
        raise ValueError(
            "Image must be grayscale or a color image."
        )

    if len(image.shape) == 3 and image.shape[2] != 3:
        raise ValueError(
            "Color image must contain three channels."
        )

    return image.copy()


# ---------------------------------------------------------
# 1. GAMMA CORRECTION
# ---------------------------------------------------------

def gamma_correction(image, gamma=0.6):
    """
    Adjust image brightness using a nonlinear intensity
    transformation.

    Formula:
        s = 255 * (r / 255) ** gamma

    gamma < 1: Brightens the image.
    gamma > 1: Darkens the image.
    gamma = 1: Leaves intensity unchanged.
    """

    image = validate_image(image)

    if gamma <= 0:
        raise ValueError("Gamma must be greater than zero.")

    lookup_table = np.array([
        ((pixel / 255.0) ** gamma) * 255
        for pixel in range(256)
    ]).clip(0, 255).astype(np.uint8)

    return cv2.LUT(image, lookup_table)


# ---------------------------------------------------------
# 2. CONTRAST STRETCHING
# ---------------------------------------------------------

def contrast_stretching(image, gamma=0.6):
    """
    Expand the intensity range using percentile-based
    contrast stretching.

    Formula:
        s = 255 * (r - r_min) / (r_max - r_min)

    The 1st and 99th percentiles are used to reduce the
    influence of extreme outlier pixels.
    """

    image = validate_image(image)

    def stretch_channel(channel):

        lower = np.percentile(channel, 1)
        upper = np.percentile(channel, 99)

        if upper <= lower:
            return channel.copy()

        stretched = (
            (channel.astype(np.float32) - lower)
            * 255.0 / (upper - lower)
        )

        return np.clip(
            stretched, 0, 255
        ).astype(np.uint8)

    if len(image.shape) == 2:
        return stretch_channel(image)

    channels = cv2.split(image)

    stretched_channels = [
        stretch_channel(channel)
        for channel in channels
    ]

    return cv2.merge(stretched_channels)


# ---------------------------------------------------------
# 3. HISTOGRAM EQUALIZATION
# ---------------------------------------------------------

def histogram_equalization(image, gamma=0.6):
    """
    Improve global contrast using histogram equalization.

    For color images, equalization is applied to the
    luminance channel in LAB color space to reduce
    unnecessary color distortion.
    """

    image = validate_image(image)

    if len(image.shape) == 2:
        return cv2.equalizeHist(image)

    lab = cv2.cvtColor(
        image, cv2.COLOR_BGR2LAB
    )

    lightness, a, b = cv2.split(lab)

    equalized_lightness = cv2.equalizeHist(
        lightness
    )

    enhanced_lab = cv2.merge((
        equalized_lightness, a, b
    ))

    return cv2.cvtColor(
        enhanced_lab, cv2.COLOR_LAB2BGR
    )


# ---------------------------------------------------------
# 4. CLAHE
# ---------------------------------------------------------

def clahe_enhancement(image, gamma=0.6):
    """
    Contrast Limited Adaptive Histogram Equalization.

    CLAHE divides the image into small regions and
    improves local contrast while limiting histogram
    amplification using a clip limit.
    """

    image = validate_image(image)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    if len(image.shape) == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(
        image, cv2.COLOR_BGR2LAB
    )

    lightness, a, b = cv2.split(lab)

    enhanced_lightness = clahe.apply(
        lightness
    )

    enhanced_lab = cv2.merge((
        enhanced_lightness, a, b
    ))

    return cv2.cvtColor(
        enhanced_lab, cv2.COLOR_LAB2BGR
    )


# ---------------------------------------------------------
# ALGORITHM INFORMATION
# ---------------------------------------------------------

ALGORITHMS = {

    "gamma": {
        "name": "Gamma Correction",

        "purpose": (
            "Adjusts image brightness using a nonlinear "
            "intensity transformation. Gamma values below "
            "1 brighten darker regions."
        ),

        "formula": (
            "s = 255 × (r / 255)^γ"
        ),

        "variables": (
            "r = Original pixel intensity (0–255)\n"
            "s = Transformed pixel intensity (0–255)\n"
            "γ = Gamma correction parameter\n"
            "γ < 1 brightens; γ > 1 darkens."
        ),

        "function": gamma_correction
    },

    "contrast_stretching": {
        "name": "Contrast Stretching",

        "purpose": (
            "Expands the image intensity range to improve "
            "the visibility of details. Uses the 1st and "
            "99th percentiles to reduce outlier influence."
        ),

        "formula": (
            "s = 255 × (r - r_min) / (r_max - r_min)"
        ),

        "variables": (
            "r = Original pixel intensity\n"
            "s = Stretched pixel intensity\n"
            "r_min = 1st percentile intensity\n"
            "r_max = 99th percentile intensity\n"
            "Output values are clipped to 0–255."
        ),

        "function": contrast_stretching
    },

    "histogram_equalization": {
        "name": "Histogram Equalization",

        "purpose": (
            "Redistributes image intensity values using "
            "the cumulative histogram to improve global "
            "contrast. Color images are processed through "
            "their luminance channel."
        ),

        "formula": (
            "s_k = round((L - 1) × CDF(r_k))"
        ),

        "variables": (
            "s_k = Mapped output intensity\n"
            "L = Number of intensity levels (256)\n"
            "CDF = Cumulative distribution function\n"
            "r_k = Original intensity level."
        ),

        "function": histogram_equalization
    },

    "clahe": {
        "name": "CLAHE",

        "purpose": (
            "Improves local contrast by processing small "
            "image regions independently and limiting "
            "excessive histogram amplification."
        ),

        "formula": (
            "s_k = round((L - 1) × CDF_clip(r_k))"
        ),

        "variables": (
            "s_k = Mapped output intensity\n"
            "L = Number of intensity levels (256)\n"
            "CDF_clip = Clipped cumulative distribution\n"
            "r_k = Original intensity level\n"
            "Clip limit = 2.0; tile grid = 8 × 8."
        ),

        "function": clahe_enhancement
    }
}


# ---------------------------------------------------------
# COMMON ENHANCEMENT INTERFACE
# ---------------------------------------------------------

def apply_enhancement(
    algorithm_key,
    image,
    gamma=0.6
):
    """
    Apply the selected enhancement algorithm.

    This function is called by the ATHER desktop GUI.
    """

    if algorithm_key not in ALGORITHMS:
        raise ValueError(
            f"Unknown enhancement algorithm: {algorithm_key}"
        )

    image = validate_image(image)

    function = ALGORITHMS[algorithm_key]["function"]

    if algorithm_key == "gamma":
        result = function(image, gamma)
    else:
        result = function(image)

    if result.shape != image.shape:
        raise RuntimeError(
            "Enhancement changed the image dimensions."
        )

    if result.dtype != np.uint8:
        raise RuntimeError(
            "Enhancement returned an unexpected image format."
        )

    return result
