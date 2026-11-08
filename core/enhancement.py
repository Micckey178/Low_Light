
"""
ATHER - Classical Digital Image Processing Engine
=================================================

Stage 1: Classical Low-Light Image Enhancement

Existing algorithms:
1. Gamma Correction
2. Contrast Stretching
3. Histogram Equalization
4. CLAHE

New algorithms:
5. Adaptive Gamma Correction
6. Single-Scale Retinex (SSR)
7. Multi-Scale Retinex (MSR)
8. LIME
9. Homomorphic Filtering

Requirements:
- No AI, ML, neural networks or pretrained models.
- Input images are preserved.
- All algorithms return uint8 images.
- Original image dimensions are preserved.
- Supports grayscale and three-channel BGR images.
"""

import cv2
import numpy as np


# =========================================================
# COMMON VALIDATION
# =========================================================

def validate_image(image):
    """Validate and copy an 8-bit grayscale or BGR image."""

    if image is None:
        raise ValueError("Input image cannot be empty.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Input image contains no pixel data.")

    if image.dtype != np.uint8:
        raise TypeError("Expected an 8-bit image with dtype uint8.")

    if image.ndim not in (2, 3):
        raise ValueError("Image must be grayscale or a color image.")

    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("Color image must contain three channels.")

    return image.copy()


def _get_luminance(image):
    """Extract luminance while preserving the original color channels."""

    if image.ndim == 2:
        return image.copy(), None

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, a, b = cv2.split(lab)

    return luminance, (a, b)


def _restore_luminance(luminance, color_channels):
    """Restore processed luminance to a grayscale or BGR image."""

    luminance = np.clip(
        luminance, 0, 255
    ).astype(np.uint8)

    if color_channels is None:
        return luminance

    a, b = color_channels

    lab = cv2.merge((luminance, a, b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _percentile_normalize(image, low=1.0, high=99.0):
    """Robustly normalize floating-point values to 0-255."""

    image = np.asarray(image, dtype=np.float32)

    lower = float(np.percentile(image, low))
    upper = float(np.percentile(image, high))

    if upper <= lower + 1e-8:
        return np.clip(image, 0, 255).astype(np.uint8)

    result = (image - lower) * (255.0 / (upper - lower))

    return np.clip(result, 0, 255).astype(np.uint8)


# =========================================================
# 1. GAMMA CORRECTION
# =========================================================

def gamma_correction(image, gamma=0.6):
    """
    Nonlinear intensity transformation.

    Formula:
        s = 255 * (r / 255)^gamma

    gamma < 1: Brightens.
    gamma > 1: Darkens.
    gamma = 1: Identity transformation.
    """

    image = validate_image(image)

    gamma = float(gamma)

    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Gamma must be a finite positive number.")

    lookup_table = np.array([
        ((pixel / 255.0) ** gamma) * 255.0
        for pixel in range(256)
    ]).clip(0, 255).astype(np.uint8)

    return cv2.LUT(image, lookup_table)


# =========================================================
# 2. CONTRAST STRETCHING
# =========================================================

def contrast_stretching(image):
    """
    Percentile-based contrast stretching.

    Formula:
        s = 255 * (r - r_min) / (r_max - r_min)

    Uses the 1st and 99th percentiles.
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

        return np.clip(stretched, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return stretch_channel(image)

    channels = cv2.split(image)

    return cv2.merge([
        stretch_channel(channel)
        for channel in channels
    ])


# =========================================================
# 3. HISTOGRAM EQUALIZATION
# =========================================================

def histogram_equalization(image):
    """
    Global histogram equalization.

    Color images are processed through LAB luminance.

    Formula:
        s_k = round((L - 1) * CDF(r_k))
    """

    image = validate_image(image)

    if image.ndim == 2:
        return cv2.equalizeHist(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lightness, a, b = cv2.split(lab)

    equalized = cv2.equalizeHist(lightness)

    enhanced_lab = cv2.merge((equalized, a, b))

    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# =========================================================
# 4. CLAHE
# =========================================================

def clahe_enhancement(
    image,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    Contrast Limited Adaptive Histogram Equalization.

    Parameters:
        clip_limit: Histogram clipping threshold.
        tile_grid_size: Number of local regions.
    """

    image = validate_image(image)

    if clip_limit <= 0:
        raise ValueError("CLAHE clip limit must be positive.")

    if (
        len(tile_grid_size) != 2
        or tile_grid_size[0] <= 0
        or tile_grid_size[1] <= 0
    ):
        raise ValueError("Invalid CLAHE tile grid size.")

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(tile_grid_size)
    )

    if image.ndim == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lightness, a, b = cv2.split(lab)

    enhanced_lightness = clahe.apply(lightness)

    enhanced_lab = cv2.merge((enhanced_lightness, a, b))

    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# =========================================================
# 5. ADAPTIVE GAMMA CORRECTION
# =========================================================

def adaptive_gamma_correction(image):
    """
    Automatically estimate gamma from average luminance.

    Formula:
        gamma = log(target) / log(mean_intensity)

        s = 255 * (r / 255)^gamma

    The target mean intensity is 0.5.

    A darker input generally receives a smaller gamma value.
    """

    image = validate_image(image)

    luminance, color_channels = _get_luminance(image)

    mean_intensity = float(
        np.mean(luminance.astype(np.float32)) / 255.0
    )

    if mean_intensity <= 0.0:
        return image.copy()

    if mean_intensity >= 1.0:
        return image.copy()

    target = 0.5

    gamma = np.log(target) / np.log(mean_intensity)

    gamma = float(np.clip(gamma, 0.25, 1.5))

    lookup_table = np.array([
        ((pixel / 255.0) ** gamma) * 255.0
        for pixel in range(256)
    ]).clip(0, 255).astype(np.uint8)

    enhanced_luminance = cv2.LUT(luminance, lookup_table)

    return _restore_luminance(
        enhanced_luminance,
        color_channels
    )


# =========================================================
# 6. SINGLE-SCALE RETINEX (SSR)
# =========================================================

def single_scale_retinex(image, sigma=80.0):
    """
    Single-Scale Retinex enhancement.

    Retinex model:
        R(x,y) = log(I(x,y)+1) - log(F(x,y)+1)

    F is estimated using Gaussian smoothing.

    sigma controls the illumination estimation scale.
    """

    image = validate_image(image)

    sigma = float(sigma)

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Retinex sigma must be positive.")

    luminance, color_channels = _get_luminance(image)

    source = luminance.astype(np.float32)

    illumination = cv2.GaussianBlur(
        source,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT
    )

    retinex = (
        np.log1p(source)
        - np.log1p(illumination)
    )

    enhanced = _percentile_normalize(retinex)

    return _restore_luminance(
        enhanced,
        color_channels
    )


# =========================================================
# 7. MULTI-SCALE RETINEX (MSR)
# =========================================================

def multi_scale_retinex(
    image,
    sigmas=(15.0, 80.0, 250.0)
):
    """
    Multi-Scale Retinex.

    Formula:
        MSR(x,y) = sum(w_i * SSR_i(x,y))

    Multiple Gaussian scales are combined to estimate
    illumination at different spatial resolutions.
    """

    image = validate_image(image)

    if not sigmas:
        raise ValueError("At least one Retinex scale is required.")

    sigmas = tuple(float(s) for s in sigmas)

    if any(not np.isfinite(s) or s <= 0 for s in sigmas):
        raise ValueError("All Retinex scales must be positive.")

    luminance, color_channels = _get_luminance(image)

    source = luminance.astype(np.float32)

    result = np.zeros_like(source, dtype=np.float32)

    weight = 1.0 / len(sigmas)

    for sigma in sigmas:

        illumination = cv2.GaussianBlur(
            source,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT
        )

        retinex = (
            np.log1p(source)
            - np.log1p(illumination)
        )

        result += weight * retinex

    enhanced = _percentile_normalize(result)

    return _restore_luminance(
        enhanced,
        color_channels
    )


# =========================================================
# 8. LIME
# =========================================================

def lime_enhancement(image, enhancement_strength=0.8):
    """
    Classical LIME-style illumination-map enhancement.

    Initial illumination estimate:
        T(x,y) = max(R(x,y), G(x,y), B(x,y))

    Enhancement:
        I_enhanced = I / T^alpha

    The illumination estimate is smoothed using bilateral
    filtering to reduce local noise in the map.

    This is a simplified LIME-style implementation,
    not a complete reproduction of the original paper's
    structure-aware illumination refinement.
    """

    image = validate_image(image)

    alpha = float(enhancement_strength)

    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "LIME enhancement strength must be between 0 and 1."
        )

    if image.ndim == 2:
        normalized = image.astype(np.float32) / 255.0

        illumination = cv2.bilateralFilter(
            normalized,
            d=9,
            sigmaColor=0.1,
            sigmaSpace=9
        )

        illumination = np.maximum(illumination, 0.05)

        enhanced = normalized / np.power(
            illumination, alpha
        )

        enhanced = np.clip(enhanced, 0.0, 1.0)

        return (enhanced * 255.0).astype(np.uint8)

    normalized = image.astype(np.float32) / 255.0

    illumination = np.max(normalized, axis=2)

    illumination = cv2.bilateralFilter(
        illumination,
        d=9,
        sigmaColor=0.1,
        sigmaSpace=9
    )

    illumination = np.maximum(illumination, 0.05)

    enhanced = normalized / np.power(
        illumination[:, :, np.newaxis],
        alpha
    )

    enhanced = np.clip(enhanced, 0.0, 1.0)

    return (enhanced * 255.0).astype(np.uint8)


# =========================================================
# 9. HOMOMORPHIC FILTERING
# =========================================================

def homomorphic_filtering(
    image,
    gamma_low=0.5,
    gamma_high=1.5,
    cutoff=30.0,
    sharpness=1.0
):
    """
    Homomorphic frequency-domain illumination correction.

    Processing:
        1. Convert luminance to logarithmic domain.
        2. Compute the 2D Fourier transform.
        3. Apply a high-frequency emphasis filter.
        4. Compute the inverse Fourier transform.
        5. Exponentiate and normalize the result.

    Filter:
        H(u,v) = (gamma_high - gamma_low)
                 * (1 - exp(-sharpness * D^2 / cutoff^2))
                 + gamma_low
    """

    image = validate_image(image)

    gamma_low = float(gamma_low)
    gamma_high = float(gamma_high)
    cutoff = float(cutoff)
    sharpness = float(sharpness)

    if not all(np.isfinite(v) for v in (
        gamma_low, gamma_high, cutoff, sharpness
    )):
        raise ValueError("Homomorphic parameters must be finite.")

    if gamma_low < 0 or gamma_high <= gamma_low:
        raise ValueError("Invalid homomorphic gamma values.")

    if cutoff <= 0 or sharpness <= 0:
        raise ValueError("Cutoff and sharpness must be positive.")

    luminance, color_channels = _get_luminance(image)

    source = luminance.astype(np.float32)

    rows, columns = source.shape

    logarithmic = np.log1p(source)

    frequency = np.fft.fftshift(
        np.fft.fft2(logarithmic)
    )

    u = np.arange(rows, dtype=np.float32) - rows / 2.0
    v = np.arange(columns, dtype=np.float32) - columns / 2.0

    V, U = np.meshgrid(v, u)

    distance_squared = U ** 2 + V ** 2

    filter_mask = (
        (gamma_high - gamma_low)
        * (
            1.0 - np.exp(
                -sharpness * distance_squared / (cutoff ** 2)
            )
        )
        + gamma_low
    )

    filtered_frequency = frequency * filter_mask

    spatial = np.fft.ifft2(
        np.fft.ifftshift(filtered_frequency)
    )

    spatial = np.real(spatial)

    restored = np.expm1(
        np.clip(spatial, -20.0, 20.0)
    )

    enhanced = _percentile_normalize(restored)

    return _restore_luminance(
        enhanced,
        color_channels
    )


# =========================================================
# ALGORITHM INFORMATION
# =========================================================

ALGORITHMS = {

    "gamma": {
        "name": "Gamma Correction",
        "purpose": "Adjusts image brightness using a nonlinear intensity transformation.",
        "formula": "s = 255 × (r / 255)^γ",
        "variables": (
            "r = Original pixel intensity (0–255)\n"
            "s = Transformed intensity\n"
            "γ = Gamma parameter\n"
            "γ < 1 brightens; γ > 1 darkens."
        ),
        "function": gamma_correction
    },

    "contrast_stretching": {
        "name": "Contrast Stretching",
        "purpose": "Expands the image intensity range using percentile-based normalization.",
        "formula": "s = 255 × (r - r_min) / (r_max - r_min)",
        "variables": (
            "r = Original intensity\n"
            "s = Stretched intensity\n"
            "r_min = 1st percentile\n"
            "r_max = 99th percentile."
        ),
        "function": contrast_stretching
    },

    "histogram_equalization": {
        "name": "Histogram Equalization",
        "purpose": "Redistributes image intensity values using the cumulative histogram.",
        "formula": "s_k = round((L - 1) × CDF(r_k))",
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
        "purpose": "Improves local contrast while limiting excessive histogram amplification.",
        "formula": "s_k = round((L - 1) × CDF_clip(r_k))",
        "variables": (
            "s_k = Mapped output intensity\n"
            "L = 256 intensity levels\n"
            "CDF_clip = Clipped cumulative distribution\n"
            "Default clip limit = 2.0\n"
            "Default tile grid = 8 × 8."
        ),
        "function": clahe_enhancement
    },

    "adaptive_gamma": {
        "name": "Adaptive Gamma Correction",
        "purpose": "Automatically estimates gamma using average image luminance.",
        "formula": "γ = log(T) / log(μ)",
        "variables": (
            "γ = Estimated gamma parameter\n"
            "T = Target mean intensity (0.5)\n"
            "μ = Normalized mean luminance\n"
            "Output = 255 × (r / 255)^γ."
        ),
        "function": adaptive_gamma_correction
    },

    "ssr": {
        "name": "Single-Scale Retinex (SSR)",
        "purpose": "Estimates illumination using one Gaussian scale and enhances reflectance.",
        "formula": "R(x,y) = log(I+1) - log(F+1)",
        "variables": (
            "R = Retinex output\n"
            "I = Input luminance\n"
            "F = Gaussian-smoothed illumination\n"
            "sigma = Gaussian scale (default 80)."
        ),
        "function": single_scale_retinex
    },

    "msr": {
        "name": "Multi-Scale Retinex (MSR)",
        "purpose": "Combines Retinex responses at multiple Gaussian scales.",
        "formula": "MSR(x,y) = Σ w_i × SSR_i(x,y)",
        "variables": (
            "MSR = Multi-scale Retinex output\n"
            "SSR_i = Retinex at scale i\n"
            "w_i = Scale weight\n"
            "Default scales = 15, 80, 250."
        ),
        "function": multi_scale_retinex
    },

    "lime": {
        "name": "LIME",
        "purpose": "Estimates an illumination map and enhances low-light image regions.",
        "formula": "T = max(R,G,B); I_enhanced = I / T^α",
        "variables": (
            "T = Estimated illumination map\n"
            "R, G, B = Normalized color channels\n"
            "I = Input image\n"
            "α = Enhancement strength (default 0.8)."
        ),
        "function": lime_enhancement
    },

    "homomorphic": {
        "name": "Homomorphic Filtering",
        "purpose": "Uses frequency-domain filtering to reduce uneven illumination and emphasize details.",
        "formula": "H(u,v) = (γH-γL)(1-e^(-cD²/D0²))+γL",
        "variables": (
            "H = Frequency-domain filter response\n"
            "γH = High-frequency gain (1.5)\n"
            "γL = Low-frequency gain (0.5)\n"
            "D = Distance from frequency origin\n"
            "D0 = Cutoff frequency (30)\n"
            "c = Filter sharpness (1.0)."
        ),
        "function": homomorphic_filtering
    }
}


# =========================================================
# COMMON ENHANCEMENT INTERFACE
# =========================================================

def apply_enhancement(
    algorithm_key,
    image,
    gamma=0.6
):
    """
    Apply an enhancement algorithm using a common interface.

    Existing GUI compatibility:
        apply_enhancement(key, image, gamma)

    Gamma is used only by the Gamma Correction algorithm.
    Other algorithms use their documented defaults.
    """

    if algorithm_key not in ALGORITHMS:
        raise ValueError(
            f"Unknown enhancement algorithm: {algorithm_key}"
        )

    original = validate_image(image)

    function = ALGORITHMS[algorithm_key]["function"]

    if algorithm_key == "gamma":
        result = function(original, gamma)
    else:
        result = function(original)

    if not isinstance(result, np.ndarray):
        raise RuntimeError("Enhancement did not return a NumPy image.")

    if result.shape != original.shape:
        raise RuntimeError(
            "Enhancement changed the image dimensions."
        )

    if result.dtype != np.uint8:
        raise RuntimeError(
            "Enhancement returned an unexpected image format."
        )

    if not np.all(np.isfinite(result)):
        raise RuntimeError(
            "Enhancement returned invalid pixel values."
        )

    return result
