
"""
ATHER - Classical Image Enhancement Engine
Phase 2: Quality, Colour Preservation and Stability

Strictly classical digital image processing.
No AI, ML, neural networks or pretrained models.

Supported input:
    - uint8 grayscale images
    - uint8 BGR colour images

All algorithms return uint8 images with the original dimensions.
"""

import cv2
import numpy as np


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image):
    """Validate a grayscale or BGR uint8 image."""

    if image is None:
        raise ValueError("Input image is empty or None.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a NumPy image array.")

    if image.dtype != np.uint8:
        raise TypeError("Only uint8 images are supported.")

    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[2] == 3:
        return image

    raise ValueError(
        "Unsupported image format. Use grayscale or 3-channel BGR."
    )


def _get_luminance(image):
    """Return luminance and colour-space information."""

    validate_image(image)

    if image.ndim == 2:
        return image.copy(), None

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0].copy(), lab


def _restore_luminance(original, luminance):
    """Restore modified luminance while preserving colour information."""

    luminance = np.clip(
        np.nan_to_num(luminance, nan=0.0, posinf=255.0, neginf=0.0),
        0,
        255
    ).astype(np.uint8)

    if original.ndim == 2:
        return luminance

    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = luminance

    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return np.clip(result, 0, 255).astype(np.uint8)


def _safe_uint8(image):
    """Convert an array safely to uint8."""

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=255.0,
        neginf=0.0
    )

    return np.clip(image, 0, 255).astype(np.uint8)


def _percentile_normalize(
    image,
    low_percentile=1.0,
    high_percentile=99.0
):
    """Robust percentile normalization with constant-image protection."""

    image = np.asarray(image, dtype=np.float32)

    if not np.all(np.isfinite(image)):
        image = np.nan_to_num(image)

    low = float(np.percentile(image, low_percentile))
    high = float(np.percentile(image, high_percentile))

    if high - low < 1e-6:
        return np.clip(image, 0, 255)

    normalized = (image - low) * (255.0 / (high - low))

    return np.clip(normalized, 0, 255)


def _blend_luminance(original, enhanced, strength):
    """Blend enhanced luminance with original luminance."""

    strength = float(np.clip(strength, 0.0, 1.0))

    result = (
        original.astype(np.float32) * (1.0 - strength)
        + enhanced.astype(np.float32) * strength
    )

    return _safe_uint8(result)


# ============================================================
# 1. GAMMA CORRECTION
# ============================================================

def gamma_correction(image, gamma=0.6):
    """
    Gamma correction.

    Formula:
        s = 255 * (r / 255) ** gamma

    gamma < 1 brightens the image.
    """

    validate_image(image)

    gamma = float(gamma)

    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Gamma must be a positive finite number.")

    lookup = np.array(
        [
            np.clip(
                ((pixel / 255.0) ** gamma) * 255.0,
                0,
                255
            )
            for pixel in range(256)
        ],
        dtype=np.uint8
    )

    if image.ndim == 2:
        return cv2.LUT(image, lookup)

    return cv2.LUT(image, lookup)


# ============================================================
# 2. CONTRAST STRETCHING
# ============================================================

def contrast_stretching(image):
    """
    Percentile-based contrast stretching.

    Colour images are processed through LAB luminance instead
    of stretching B, G and R independently.
    """

    validate_image(image)

    luminance, _ = _get_luminance(image)

    low = float(np.percentile(luminance, 1.0))
    high = float(np.percentile(luminance, 99.0))

    if high - low < 2:
        return image.copy()

    stretched = (
        (luminance.astype(np.float32) - low)
        * (255.0 / (high - low))
    )

    stretched = _safe_uint8(stretched)

    return _restore_luminance(image, stretched)


# ============================================================
# 3. HISTOGRAM EQUALIZATION
# ============================================================

def histogram_equalization(image):
    """
    Histogram equalization on luminance.

    This avoids independently equalizing colour channels.
    """

    validate_image(image)

    if image.ndim == 2:
        return cv2.equalizeHist(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================
# 4. CLAHE
# ============================================================

def clahe_enhancement(
    image,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    Contrast Limited Adaptive Histogram Equalization.
    """

    validate_image(image)

    clip_limit = float(clip_limit)

    if clip_limit <= 0:
        raise ValueError("CLAHE clip limit must be positive.")

    if (
        not isinstance(tile_grid_size, tuple)
        or len(tile_grid_size) != 2
        or min(tile_grid_size) < 1
    ):
        raise ValueError("Invalid CLAHE tile grid size.")

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    if image.ndim == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================
# 5. ADAPTIVE GAMMA CORRECTION
# ============================================================

def adaptive_gamma_correction(image):
    """
    Adaptive gamma correction based on mean luminance.

    The estimated gamma is bounded to prevent extreme
    brightness changes.
    """

    validate_image(image)

    luminance, _ = _get_luminance(image)

    mean_value = float(np.mean(luminance)) / 255.0

    mean_value = float(np.clip(mean_value, 0.01, 0.99))

    gamma = np.log(0.5) / np.log(mean_value)

    gamma = float(np.clip(gamma, 0.35, 1.8))

    enhanced = gamma_correction(image, gamma)

    return enhanced


# ============================================================
# RETINEX SUPPORT
# ============================================================

def _retinex_channel(channel, sigma):
    """Calculate a single-scale Retinex response."""

    channel = channel.astype(np.float32) / 255.0

    illumination = cv2.GaussianBlur(
        channel,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma)
    )

    illumination = np.maximum(illumination, 1e-4)

    response = (
        np.log(channel + 1e-4)
        - np.log(illumination + 1e-4)
    )

    return response


def _retinex_enhancement(
    image,
    sigmas,
    strength=0.65
):
    """Shared controlled Retinex implementation."""

    validate_image(image)

    luminance, _ = _get_luminance(image)

    source = luminance.astype(np.float32) / 255.0

    responses = []

    for sigma in sigmas:
        response = _retinex_channel(
            luminance,
            sigma
        )

        responses.append(response)

    retinex = np.mean(
        np.stack(responses, axis=0),
        axis=0
    )

    normalized = _percentile_normalize(
        retinex,
        1.0,
        99.0
    )

    # Blend with original luminance to reduce harsh changes.
    blended = (
        source * 255.0 * (1.0 - strength)
        + normalized * strength
    )

    enhanced_luminance = _safe_uint8(blended)

    return _restore_luminance(
        image,
        enhanced_luminance
    )


# ============================================================
# 6. SINGLE-SCALE RETINEX
# ============================================================

def single_scale_retinex(image, sigma=80.0):
    """
    Single-Scale Retinex.

    Formula:
        R(x,y) = log(I(x,y)) - log(G(x,y) * I(x,y))
    """

    validate_image(image)

    sigma = float(sigma)

    if sigma <= 0:
        raise ValueError("Retinex sigma must be positive.")

    return _retinex_enhancement(
        image,
        sigmas=(sigma,),
        strength=0.65
    )


# ============================================================
# 7. MULTI-SCALE RETINEX
# ============================================================

def multi_scale_retinex(
    image,
    sigmas=(15, 80, 250)
):
    """
    Multi-Scale Retinex.

    Combines Retinex responses at multiple spatial scales.
    """

    validate_image(image)

    if not sigmas:
        raise ValueError("At least one Retinex scale is required.")

    checked_sigmas = tuple(float(s) for s in sigmas)

    if any(s <= 0 for s in checked_sigmas):
        raise ValueError("All Retinex scales must be positive.")

    return _retinex_enhancement(
        image,
        sigmas=checked_sigmas,
        strength=0.60
    )


# ============================================================
# 8. LIME-STYLE ILLUMINATION ENHANCEMENT
# ============================================================

def lime_enhancement(
    image,
    enhancement_strength=0.8
):
    """
    Simplified LIME-style illumination enhancement.

    Estimates illumination using the maximum colour channel,
    smooths the illumination map and applies a bounded gain.

    This is a classical implementation inspired by illumination
    map enhancement; it is not a claim of exact reproduction
    of the original LIME research algorithm.
    """

    validate_image(image)

    strength = float(enhancement_strength)

    if not 0.0 <= strength <= 1.0:
        raise ValueError(
            "Enhancement strength must be between 0 and 1."
        )

    if image.ndim == 2:
        channels = image.astype(np.float32)[:, :, None]
    else:
        channels = image.astype(np.float32)

    illumination = np.max(channels, axis=2)

    illumination = illumination / 255.0

    illumination = cv2.bilateralFilter(
        illumination.astype(np.float32),
        d=9,
        sigmaColor=0.1,
        sigmaSpace=9
    )

    illumination = np.maximum(illumination, 0.08)

    gain = 1.0 / illumination

    # Limit gain to avoid extreme amplification of dark noise.
    gain = np.clip(gain, 1.0, 4.0)

    if image.ndim == 2:
        enhanced = image.astype(np.float32) * gain
    else:
        enhanced = channels * gain[:, :, None]

    enhanced = _safe_uint8(enhanced)

    # Blend rather than applying the full estimated gain.
    result = cv2.addWeighted(
        image,
        1.0 - strength,
        enhanced,
        strength,
        0
    )

    return _safe_uint8(result)


# ============================================================
# 9. HOMOMORPHIC FILTERING
# ============================================================

def homomorphic_filtering(
    image,
    gamma_low=0.5,
    gamma_high=1.5,
    cutoff=30,
    sharpness=1.0
):
    """
    Homomorphic filtering on image luminance.

    Separates low-frequency illumination from high-frequency
    reflectance using logarithmic and frequency-domain processing.
    """

    validate_image(image)

    gamma_low = float(gamma_low)
    gamma_high = float(gamma_high)
    cutoff = float(cutoff)
    sharpness = float(sharpness)

    if gamma_low < 0 or gamma_high <= 0:
        raise ValueError("Invalid homomorphic filter gains.")

    if cutoff <= 0 or sharpness <= 0:
        raise ValueError("Cutoff and sharpness must be positive.")

    luminance, _ = _get_luminance(image)

    source = luminance.astype(np.float32) / 255.0

    log_image = np.log1p(source)

    rows, cols = log_image.shape

    padded = cv2.copyMakeBorder(
        log_image,
        0,
        rows % 2,
        0,
        cols % 2,
        cv2.BORDER_REFLECT
    )

    padded_rows, padded_cols = padded.shape

    spectrum = np.fft.fft2(padded)
    spectrum_shifted = np.fft.fftshift(spectrum)

    y, x = np.ogrid[
        :padded_rows,
        :padded_cols
    ]

    center_y = padded_rows // 2
    center_x = padded_cols // 2

    distance_squared = (
        (y - center_y) ** 2
        + (x - center_x) ** 2
    )

    distance = np.sqrt(distance_squared)

    # Smooth high-pass response.
    high_pass = 1.0 - np.exp(
        -sharpness * distance_squared
        / (2.0 * cutoff ** 2)
    )

    transfer = (
        gamma_low
        + (gamma_high - gamma_low) * high_pass
    )

    filtered_spectrum = spectrum_shifted * transfer

    inverse = np.fft.ifft2(
        np.fft.ifftshift(filtered_spectrum)
    )

    filtered = np.real(inverse)

    filtered = filtered[:rows, :cols]

    reconstructed = np.expm1(
        np.clip(filtered, -10, 10)
    )

    reconstructed = _percentile_normalize(
        reconstructed,
        1.0,
        99.0
    )

    # Avoid excessive changes to the original luminance.
    reconstructed = _blend_luminance(
        luminance,
        reconstructed,
        0.65
    )

    return _restore_luminance(
        image,
        reconstructed
    )


# ============================================================
# ALGORITHM REGISTRY
# ============================================================

ALGORITHMS = {
    "gamma": {
        "name": "Gamma Correction",
        "purpose": "Adjusts image brightness using a nonlinear intensity mapping.",
        "formula": "s = 255 * (r / 255) ^ gamma",
        "variables": {
            "gamma": "Controls brightness. Values below 1 brighten the image."
        },
        "function": gamma_correction
    },

    "contrast_stretching": {
        "name": "Contrast Stretching",
        "purpose": "Expands the useful luminance range using percentile limits.",
        "formula": "s = (r - r_low) * 255 / (r_high - r_low)",
        "variables": {
            "r_low": "1st percentile of image luminance.",
            "r_high": "99th percentile of image luminance."
        },
        "function": contrast_stretching
    },

    "histogram_equalization": {
        "name": "Histogram Equalization",
        "purpose": "Redistributes luminance values to improve global contrast.",
        "formula": "s = (L - 1) * CDF(r)",
        "variables": {
            "CDF": "Cumulative distribution function of image intensity.",
            "L": "Number of intensity levels."
        },
        "function": histogram_equalization
    },

    "clahe": {
        "name": "CLAHE",
        "purpose": "Enhances local contrast with histogram clipping.",
        "formula": "Local histogram equalization with contrast limiting.",
        "variables": {
            "clip_limit": "Limits local histogram amplification.",
            "tile_grid_size": "Number of contextual regions."
        },
        "function": clahe_enhancement
    },

    "adaptive_gamma": {
        "name": "Adaptive Gamma Correction",
        "purpose": "Selects gamma from average image luminance.",
        "formula": "gamma = log(0.5) / log(mean_intensity)",
        "variables": {
            "mean_intensity": "Normalized average luminance."
        },
        "function": adaptive_gamma_correction
    },

    "ssr": {
        "name": "Single-Scale Retinex",
        "purpose": "Estimates reflectance using one Gaussian illumination scale.",
        "formula": "R(x,y) = log(I(x,y)) - log(G_sigma * I)",
        "variables": {
            "sigma": "Gaussian illumination scale."
        },
        "function": single_scale_retinex
    },

    "msr": {
        "name": "Multi-Scale Retinex",
        "purpose": "Combines Retinex responses from multiple spatial scales.",
        "formula": "R = sum(R_sigma) / number_of_scales",
        "variables": {
            "sigmas": "Gaussian scales used for illumination estimation."
        },
        "function": multi_scale_retinex
    },

    "lime": {
        "name": "LIME-Style Enhancement",
        "purpose": "Estimates illumination and applies bounded brightness compensation.",
        "formula": "Enhanced = Image * illumination_gain",
        "variables": {
            "enhancement_strength": "Controls the blending of the enhanced result."
        },
        "function": lime_enhancement
    },

    "homomorphic": {
        "name": "Homomorphic Filtering",
        "purpose": "Uses frequency-domain filtering to separate illumination and reflectance.",
        "formula": "H(u,v) = gamma_low + (gamma_high - gamma_low) * HP(u,v)",
        "variables": {
            "gamma_low": "Low-frequency gain.",
            "gamma_high": "High-frequency gain.",
            "cutoff": "Frequency cutoff.",
            "sharpness": "Controls the filter transition."
        },
        "function": homomorphic_filtering
    }
}


# ============================================================
# COMMON ENHANCEMENT INTERFACE
# ============================================================

def apply_enhancement(
    algorithm_key,
    image,
    gamma=0.6
):
    """
    Apply an algorithm using the common ATHER interface.

    The interface intentionally remains compatible with the
    existing GUI and benchmark modules.
    """

    validate_image(image)

    if algorithm_key not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm_key}"
        )

    original_shape = image.shape

    algorithm = ALGORITHMS[algorithm_key]

    if algorithm_key == "gamma":
        enhanced = algorithm["function"](
            image,
            gamma=gamma
        )
    else:
        enhanced = algorithm["function"](image)

    if enhanced is None:
        raise RuntimeError(
            f"{algorithm['name']} returned no image."
        )

    if enhanced.shape != original_shape:
        raise RuntimeError(
            f"{algorithm['name']} changed image dimensions."
        )

    if enhanced.dtype != np.uint8:
        enhanced = _safe_uint8(enhanced)

    return enhanced
