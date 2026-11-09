"""
ATHER-ACP: Adaptive Classical Processing Engine
------------------------------------------------
Deterministic image analysis and recommendation rules for ATHER.

This module does NOT enhance images and does NOT execute algorithms.
It analyzes measurable image properties and produces an explainable,
user-reviewable recommendation plan.

Strictly classical image processing:
- No AI, ML, neural networks, pretrained models, or cloud processing.
- Recommendations are deterministic rules, not learned predictions.
- All noise measurements are explicitly approximate proxies.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


ENGINE_NAME = "ATHER-ACP"
ENGINE_VERSION = "0.3.0"

# These are initial engineering thresholds, not universal perceptual
# standards. They must be calibrated against a documented image dataset.
DEFAULT_THRESHOLDS = {
    "very_dark_mean": 75.0,             # 8-bit luminance
    "dark_pixel_fraction": 0.35,        # luminance <= 51
    "low_contrast_std": 32.0,           # 8-bit luminance standard deviation
    "low_percentile_span": 70.0,        # P90 - P10
    "weak_local_contrast": 18.0,        # mean tile standard deviation
    "uneven_illumination_std": 0.12,    # normalized low-frequency luminance
    "high_noise_proxy": 8.0,            # approximate 8-bit residual MAD estimate
    "black_clipping_fraction": 0.02,    # luminance <= 5
    "highlight_clipping_fraction": 0.02 # luminance >= 250
}


def validate_image(image: np.ndarray) -> np.ndarray:
    """Accept uint8 grayscale, one-channel, BGR, or BGRA arrays."""
    if image is None:
        raise ValueError("Input image is empty or None.")
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")
    if image.dtype != np.uint8:
        raise TypeError("ATHER-ACP currently supports uint8 images only.")
    if image.ndim == 2:
        if image.size == 0:
            raise ValueError("Input image is empty.")
        return image
    if image.ndim == 3 and image.shape[2] in (1, 3, 4):
        if image.size == 0:
            raise ValueError("Input image is empty.")
        return image
    raise ValueError("Supported formats: grayscale, one-channel, BGR, BGRA.")


def _get_gray_and_lab(image: np.ndarray):
    """Return grayscale luminance and optional LAB representation."""
    validate_image(image)

    if image.ndim == 2:
        return image.copy(), None, "grayscale"
    if image.shape[2] == 1:
        return image[:, :, 0].copy(), None, "grayscale"
    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return gray, lab, "BGRA"
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray, lab, "BGR"


def _fraction(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask) / mask.size) if mask.size else 0.0


def _local_contrast(gray: np.ndarray, grid_size: int = 8) -> dict[str, float]:
    """Measure standard deviation inside non-overlapping image tiles."""
    height, width = gray.shape
    rows = min(grid_size, height)
    cols = min(grid_size, width)
    y_edges = np.linspace(0, height, rows + 1, dtype=int)
    x_edges = np.linspace(0, width, cols + 1, dtype=int)

    tile_stds = []
    tile_means = []
    for yi in range(rows):
        for xi in range(cols):
            tile = gray[y_edges[yi]:y_edges[yi + 1],
                        x_edges[xi]:x_edges[xi + 1]]
            if tile.size:
                values = tile.astype(np.float32)
                tile_stds.append(float(np.std(values)))
                tile_means.append(float(np.mean(values)))

    if not tile_stds:
        return {
            "mean_tile_contrast": 0.0,
            "std_tile_contrast": 0.0,
            "std_tile_mean": 0.0,
            "tiles_analyzed": 0
        }

    return {
        "mean_tile_contrast": float(np.mean(tile_stds)),
        "std_tile_contrast": float(np.std(tile_stds)),
        "std_tile_mean": float(np.std(tile_means)),
        "tiles_analyzed": len(tile_stds)
    }


def _estimate_illumination(gray: np.ndarray) -> dict[str, float]:
    """
    Estimate low-frequency illumination variation using Gaussian smoothing.

    This is a spatial-statistics proxy, not a physical measurement of
    scene illumination and not a learned illumination estimator.
    """
    normalized = gray.astype(np.float32) / 255.0
    min_dimension = min(gray.shape)
    sigma = max(1.0, min_dimension / 20.0)
    illumination = cv2.GaussianBlur(
        normalized, (0, 0), sigmaX=sigma, sigmaY=sigma
    )
    p10, p50, p90 = np.percentile(illumination, [10, 50, 90])
    return {
        "illumination_mean": float(np.mean(illumination)),
        "illumination_std": float(np.std(illumination)),
        "illumination_p10": float(p10),
        "illumination_median": float(p50),
        "illumination_p90": float(p90),
        "illumination_p90_p10_span": float(p90 - p10),
        "illumination_sigma_pixels": float(sigma)
    }


def _estimate_noise_proxy(gray: np.ndarray) -> dict[str, Any]:
    """
    Estimate high-frequency residual dispersion in relatively flat regions.

    Edges and fine texture can still affect this estimate. It is therefore
    named a proxy and must not be reported as ground-truth sensor noise.
    """
    gray_f = gray.astype(np.float32)
    median = cv2.medianBlur(gray, 3).astype(np.float32)
    residual = gray_f - median

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    gradient_limit = float(np.percentile(gradient, 40.0))
    flat_mask = gradient <= gradient_limit
    minimum_samples = max(64, int(gray.size * 0.05))

    used_flat_region = int(np.count_nonzero(flat_mask)) >= minimum_samples
    selected = residual[flat_mask] if used_flat_region else residual.reshape(-1)

    if selected.size == 0:
        selected = residual.reshape(-1)

    center = float(np.median(selected))
    mad = float(np.median(np.abs(selected - center)))
    estimate = 1.4826 * mad

    return {
        "noise_estimate_proxy": float(max(0.0, estimate)),
        "noise_estimation_method": "flat-region median-filter residual MAD",
        "noise_estimation_samples": int(selected.size),
        "noise_estimation_used_flat_region": bool(used_flat_region),
        "noise_estimation_limitation": (
            "Approximate proxy only; residual texture, compression artifacts, "
            "and edges may still influence the value."
        )
    }


def analyze_image(image: np.ndarray) -> dict[str, Any]:
    """
    Calculate deterministic image statistics used by the ACP rule engine.

    Returns a JSON-friendly dictionary. No enhancement is performed.
    """
    gray, lab, color_mode = _get_gray_and_lab(image)
    gray_f = gray.astype(np.float32)
    height, width = gray.shape

    p1, p10, p25, p50, p75, p90, p99 = np.percentile(
        gray_f, [1, 10, 25, 50, 75, 90, 99]
    )
    local = _local_contrast(gray)
    illumination = _estimate_illumination(gray)
    noise = _estimate_noise_proxy(gray)

    black_fraction = _fraction(gray <= 5)
    highlight_fraction = _fraction(gray >= 250)
    dark_fraction = _fraction(gray <= 51)
    very_bright_fraction = _fraction(gray >= 245)

    color_metrics = {
        "mean_chroma_a": None,
        "mean_chroma_b": None,
        "std_chroma_a": None,
        "std_chroma_b": None,
        "mean_hsv_saturation": None
    }
    if lab is not None:
        a = lab[:, :, 1].astype(np.float32)
        b = lab[:, :, 2].astype(np.float32)
        bgr = image[:, :, :3] if image.shape[2] == 4 else image
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        color_metrics = {
            "mean_chroma_a": float(np.mean(a)),
            "mean_chroma_b": float(np.mean(b)),
            "std_chroma_a": float(np.std(a)),
            "std_chroma_b": float(np.std(b)),
            "mean_hsv_saturation": float(np.mean(hsv[:, :, 1]))
        }

    return {
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "analysis_type": "deterministic_classical_image_statistics",
        "image": {
            "width": int(width),
            "height": int(height),
            "total_pixels": int(width * height),
            "channels": int(1 if image.ndim == 2 else image.shape[2]),
            "color_mode": color_mode,
            "dtype": str(image.dtype),
            "memory_kb": float(image.nbytes / 1024.0)
        },
        "luminance": {
            "mean": float(np.mean(gray_f)),
            "median": float(p50),
            "minimum": int(np.min(gray)),
            "maximum": int(np.max(gray)),
            "standard_deviation": float(np.std(gray_f)),
            "dynamic_range": int(np.max(gray) - np.min(gray)),
            "p01": float(p1),
            "p10": float(p10),
            "p25": float(p25),
            "p50": float(p50),
            "p75": float(p75),
            "p90": float(p90),
            "p99": float(p99),
            "p90_p10_span": float(p90 - p10),
            "dark_pixel_fraction_le_51": dark_fraction,
            "very_bright_pixel_fraction_ge_245": very_bright_fraction,
            "black_clipping_fraction_le_5": black_fraction,
            "highlight_clipping_fraction_ge_250": highlight_fraction
        },
        "local_contrast": local,
        "illumination": illumination,
        "noise": noise,
        "color": color_metrics,
        "thresholds_used": dict(DEFAULT_THRESHOLDS)
    }


def classify_image_conditions(
    analysis: dict[str, Any],
    thresholds: dict[str, float] | None = None
) -> dict[str, Any]:
    """Classify image conditions while separating measurements from intent.

    A dark-looking image is not automatically defective: it may be a night
    scene or an intentionally low-key photograph. The classifier therefore
    reports a dark-dominant profile without claiming to know the photographer's
    intent. Thresholds remain configurable and are engineering defaults, not
    universal perceptual standards.
    """
    if not isinstance(analysis, dict):
        raise TypeError("analysis must be the dictionary returned by analyze_image().")

    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        unknown = set(thresholds) - set(limits)
        if unknown:
            raise ValueError(f"Unknown threshold names: {sorted(unknown)}")
        for key, value in thresholds.items():
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"Threshold {key} must be finite.")
            if value < 0:
                raise ValueError(f"Threshold {key} cannot be negative.")
            limits[key] = value

    lum = analysis["luminance"]
    local = analysis["local_contrast"]
    illum = analysis["illumination"]
    noise = analysis["noise"]

    mean = float(lum["mean"])
    p25 = float(lum["p25"])
    p75 = float(lum["p75"])
    dark_fraction = float(lum["dark_pixel_fraction_le_51"])
    bright_fraction = float(lum["very_bright_pixel_fraction_ge_245"])
    spread = float(lum["p90_p10_span"])

    # Require several mutually supporting signals for a dark-dominant profile.
    # This reduces false positives caused by a single dark object/background.
    dark_dominant = (
        mean <= limits["very_dark_mean"]
        and p75 <= 100.0
        and dark_fraction >= limits["dark_pixel_fraction"]
    )
    bright_dominant = mean >= 180.0 and p25 >= 160.0
    mixed_lighting = dark_fraction >= 0.20 and bright_fraction >= 0.02
    low_global = (
        float(lum["standard_deviation"]) <= limits["low_contrast_std"]
        or spread <= limits["low_percentile_span"]
    )
    weak_local = (
        float(local["mean_tile_contrast"]) <= limits["weak_local_contrast"]
    )

    conditions = {
        "very_dark": bool(dark_dominant),
        "low_global_contrast": bool(low_global),
        "weak_local_contrast": bool(weak_local),
        "uneven_illumination": bool(
            float(illum["illumination_std"]) >= limits["uneven_illumination_std"]
        ),
        "high_noise_proxy": bool(
            float(noise["noise_estimate_proxy"]) >= limits["high_noise_proxy"]
        ),
        "black_clipping_risk": bool(
            float(lum["black_clipping_fraction_le_5"]) >= limits["black_clipping_fraction"]
        ),
        "highlight_clipping_risk": bool(
            float(lum["highlight_clipping_fraction_ge_250"]) >= limits["highlight_clipping_fraction"]
        ),
        "bright_dominant_scene": bool(bright_dominant),
        "mixed_lighting": bool(mixed_lighting),
    }

    # Compound profiles describe co-occurring measurements. They do not
    # introduce a learned class or infer the photographer's intent.
    compound_rules = {
        "dark_with_uneven_illumination": conditions["very_dark"] and conditions["uneven_illumination"],
        "dark_with_elevated_noise": conditions["very_dark"] and conditions["high_noise_proxy"],
        "global_and_local_contrast_weak": conditions["low_global_contrast"] and conditions["weak_local_contrast"],
        "uneven_mixed_lighting": conditions["uneven_illumination"] and conditions["mixed_lighting"],
        "noise_with_weak_local_contrast": conditions["high_noise_proxy"] and conditions["weak_local_contrast"],
    }
    conditions.update({name: bool(enabled) for name, enabled in compound_rules.items()})

    explanations = {
        "very_dark": (
            f"Dark-dominant tonal distribution: mean {mean:.2f}/255, "
            f"P75 {p75:.2f}/255, and {dark_fraction * 100:.1f}% of pixels <= 51. "
            "This does not establish whether the darkness is intentional."
        ),
        "low_global_contrast": (
            f"Luminance standard deviation is {lum['standard_deviation']:.2f} "
            f"and P90-P10 span is {spread:.2f}."
        ),
        "weak_local_contrast": (
            f"Mean tile contrast is {local['mean_tile_contrast']:.2f}/255."
        ),
        "uneven_illumination": (
            f"Gaussian low-frequency illumination standard deviation is "
            f"{illum['illumination_std']:.3f} on a 0-1 scale."
        ),
        "high_noise_proxy": (
            f"Flat-region residual MAD noise proxy is "
            f"{noise['noise_estimate_proxy']:.2f}/255; this is not ground truth."
        ),
        "black_clipping_risk": (
            f"{lum['black_clipping_fraction_le_5'] * 100:.2f}% of pixels have luminance <= 5."
        ),
        "highlight_clipping_risk": (
            f"{lum['highlight_clipping_fraction_ge_250'] * 100:.2f}% of pixels have luminance >= 250."
        ),
        "bright_dominant_scene": (
            f"Bright-dominant distribution: mean {mean:.2f}/255 and "
            f"P25 {p25:.2f}/255. Avoid unnecessary global brightening."
        ),
        "mixed_lighting": (
            f"The image contains both substantial dark areas ({dark_fraction * 100:.1f}%) "
            f"and very bright areas ({bright_fraction * 100:.1f}%). Global remapping may "
            "damage one of these regions."
        ),
        "dark_with_uneven_illumination": (
            "Combined profile: the image is dark-dominant and its low-frequency "
            "illumination varies spatially. Compare illumination-aware methods "
            "against conservative tonal mapping."
        ),
        "dark_with_elevated_noise": (
            "Combined profile: dark tonal levels coincide with an elevated residual "
            "noise proxy. Brightening may make noise more visible; use mild settings."
        ),
        "global_and_local_contrast_weak": (
            "Combined profile: both whole-image luminance spread and average tile "
            "contrast are limited. Compare global and local contrast methods."
        ),
        "uneven_mixed_lighting": (
            "Combined profile: illumination varies spatially while dark and bright "
            "regions coexist. Avoid assuming one global correction will suit all regions."
        ),
        "noise_with_weak_local_contrast": (
            "Combined profile: local contrast is weak while the residual noise proxy "
            "is elevated. Aggressive local enhancement may amplify texture and noise."
        ),
    }

    active = [name for name, enabled in conditions.items() if enabled]
    if dark_dominant:
        exposure_interpretation = "dark_dominant_intent_unknown"
    elif bright_dominant:
        exposure_interpretation = "bright_dominant"
    elif mixed_lighting:
        exposure_interpretation = "mixed_dark_and_bright_regions"
    else:
        exposure_interpretation = "no_strong_global_exposure_bias_detected"

    return {
        "conditions": conditions,
        "active_conditions": active,
        "compound_conditions": [name for name, enabled in compound_rules.items() if enabled],
        "explanations": {name: explanations[name] for name in active},
        "exposure_interpretation": exposure_interpretation,
        "exposure_intent_note": (
            "Image statistics cannot determine whether a dark or bright artistic "
            "look is intentional. User review is required before enhancement."
        ),
        "thresholds_used": limits,
        "classification_method": "multi-signal deterministic rules with intent ambiguity disclosure"
    }


def recommend_algorithms(
    analysis: dict[str, Any],
    classification: dict[str, Any] | None = None,
    max_recommendations: int = 5
) -> list[dict[str, Any]]:
    """Rank candidates conservatively; scores are not quality probabilities."""
    if not isinstance(max_recommendations, int) or isinstance(max_recommendations, bool) or max_recommendations < 1:
        raise ValueError("max_recommendations must be a positive integer.")

    classified = classification or classify_image_conditions(analysis)
    active = set(classified["active_conditions"])
    scores = {key: 0 for key in (
        "gamma", "adaptive_gamma", "contrast_stretching",
        "histogram_equalization", "clahe", "ssr", "msr",
        "lime", "homomorphic"
    )}
    reasons = {key: [] for key in scores}
    score_breakdown = {key: [] for key in scores}

    def add(key: str, points: int, reason: str) -> None:
        scores[key] += points
        reasons[key].append(reason)
        score_breakdown[key].append({"points": int(points), "rule": reason})

    bright_scene = "bright_dominant_scene" in active
    mixed_scene = "mixed_lighting" in active
    noisy = "high_noise_proxy" in active

    if "very_dark" in active:
        add("gamma", 72, "A dark-dominant tonal distribution makes conservative brightness mapping worth testing.")
        add("adaptive_gamma", 68, "Adaptive gamma is a candidate because the measured tonal distribution is dark-dominant.")
        add("lime", 42, "Illumination compensation is a candidate; inspect noise and highlight preservation.")
    if "low_global_contrast" in active and not bright_scene:
        add("contrast_stretching", 52, "The measured luminance distribution has limited global spread.")
        add("histogram_equalization", 38, "Global histogram redistribution may expand tonal use; inspect for over-enhancement.")
    if "weak_local_contrast" in active and not bright_scene:
        add("clahe", 62, "Local contrast enhancement may reveal detail; inspect noise amplification.")
        add("msr", 34, "Multi-scale reflectance processing is a secondary candidate for local detail visibility.")
    if "uneven_illumination" in active:
        add("msr", 70, "Spatial illumination variation makes multi-scale Retinex worth evaluating.")
        add("ssr", 58, "Single-scale Retinex is a candidate for spatial illumination compensation.")
        add("lime", 54, "Illumination-map compensation is relevant to spatially varying light.")
        add("homomorphic", 42, "Frequency-domain illumination/reflectance separation is a candidate for comparison.")
    if mixed_scene:
        for key in ("gamma", "adaptive_gamma", "contrast_stretching", "histogram_equalization"):
            add(key, -22, "Mixed dark and bright regions make global remapping riskier; prefer local methods for comparison.")
    if noisy:
        for key in ("clahe", "ssr", "msr", "lime", "homomorphic"):
            add(key, -24, "Elevated noise proxy may be amplified by this method; denoise or inspect before use.")
        for key in ("gamma", "adaptive_gamma"):
            add(key, -12, "Brightness amplification may make existing noise more visible.")
    if "black_clipping_risk" in active:
        for key in scores:
            add(key, -18, "Clipped shadow values cannot be reconstructed reliably; use conservative settings.")
    if "highlight_clipping_risk" in active:
        for key in ("gamma", "adaptive_gamma", "lime", "histogram_equalization", "contrast_stretching"):
            add(key, -22, "Clipped highlights require conservative processing and visual inspection.")

    # Compound-profile adjustments reward methods that address the measured
    # combination while retaining the existing penalties for noise/clipping.
    compounds = set(classified.get("compound_conditions", []))
    if "dark_with_uneven_illumination" in compounds:
        add("msr", 12, "Combined dark + uneven-illumination profile: a multi-scale illumination/reflectance method is a stronger comparison candidate.")
        add("lime", 8, "Combined dark + uneven-illumination profile: compare illumination compensation, while checking noise and highlights.")
        add("gamma", -8, "Spatially uneven lighting means one global gamma curve may not improve all regions equally.")
    if "dark_with_elevated_noise" in compounds:
        add("gamma", -6, "Dark + elevated noise-proxy profile: keep global brightening conservative to limit visible noise amplification.")
        add("adaptive_gamma", -6, "Dark + elevated noise-proxy profile: use a mild adaptive setting and inspect flat regions.")
    if "global_and_local_contrast_weak" in compounds:
        add("clahe", 8, "Both global and tile-level contrast measurements are weak; CLAHE is worth a controlled local-contrast comparison.")
        add("contrast_stretching", 5, "Both global and local contrast measurements are weak; include a conservative global baseline.")
    if "uneven_mixed_lighting" in compounds:
        add("msr", 8, "Uneven illumination plus mixed dark/bright regions favors testing spatially adaptive reflectance methods.")
        add("ssr", 5, "Uneven illumination plus mixed regions supports a cautious single-scale Retinex comparison.")
    if "noise_with_weak_local_contrast" in compounds:
        add("clahe", -8, "Weak local contrast with elevated noise proxy: reduce local amplification strength and inspect noise carefully.")

    # Parameter notes are conservative starting points for manual experiments.
    # They never invoke or alter the enhancement implementation.
    parameter_guidance = {
        "gamma": "Starting point: gamma 0.60 (existing ATHER default); compare nearby mild values such as 0.55 and 0.70.",
        "adaptive_gamma": "Starting point: use the current ATHER adaptive-gamma defaults; compare a conservative setting before increasing strength.",
        "contrast_stretching": "Starting point: use a narrow percentile trim (about 1st–99th percentile) only if the implementation exposes percentile limits.",
        "histogram_equalization": "Starting point: prefer luminance-channel equalization when the implementation supports it; compare against the original to check for tonal shifts.",
        "clahe": "Starting point: clip limit around 2.0 with an 8×8 tile grid, if those controls are exposed; lower the clip limit if noise becomes prominent.",
        "ssr": "Starting point: retain the implementation's default Gaussian scale first; compare one smaller and one larger scale rather than making a large jump.",
        "msr": "Starting point: retain the implementation's default scale set and gains first; change one scale-related control at a time.",
        "lime": "Starting point: retain the implementation's default illumination-map settings; avoid stronger enhancement when the noise proxy is elevated.",
        "homomorphic": "Starting point: retain the implementation's defaults and use mild frequency-gain settings if exposed; inspect halos and colour shifts.",
    }

    # Do not manufacture recommendations for a bright, low-contrast image
    # merely because its standard deviation is small (e.g. snow/fog scenes).
    # Such images may still be enhanced manually, but ACP should not imply need.
    if bright_scene and not ("uneven_illumination" in active or mixed_scene):
        return []

    ranked = sorted(scores, key=lambda key: (-scores[key], key))
    output = []
    names = {
        "gamma": "Gamma Correction",
        "adaptive_gamma": "Adaptive Gamma Correction",
        "contrast_stretching": "Contrast Stretching",
        "histogram_equalization": "Histogram Equalization",
        "clahe": "CLAHE",
        "ssr": "Single-Scale Retinex",
        "msr": "Multi-Scale Retinex",
        "lime": "LIME-Style Enhancement",
        "homomorphic": "Homomorphic Filtering",
    }
    for key in ranked:
        if scores[key] <= 0:
            continue
        guidance = parameter_guidance[key]
        visible_reasons = list(reasons[key])
        visible_reasons.append("Parameter starting point (manual trial only): " + guidance)
        output.append({
            "algorithm_key": key,
            "algorithm_name": names[key],
            "priority_score": int(min(100, scores[key])),
            "raw_rule_score": int(scores[key]),
            "score_type": "heuristic_rule_priority_not_probability",
            "score_breakdown": score_breakdown[key],
            "reasons": visible_reasons,
            "parameter_guidance": guidance,
            "parameter_guidance_status": "manual_starting_point_not_automatically_applied",
        })
        if len(output) >= max_recommendations:
            break
    return output

def generate_explanation(
    analysis: dict[str, Any],
    classification: dict[str, Any] | None = None,
    recommendations: list[dict[str, Any]] | None = None
) -> str:
    """Create a human-readable explanation of measured conditions and rules."""
    classified = classification or classify_image_conditions(analysis)
    recs = recommendations or recommend_algorithms(analysis, classified)

    lum = analysis["luminance"]
    lines = [
        "ATHER-ACP IMAGE ANALYSIS",
        "Analysis is deterministic and uses classical image statistics.",
        f"Exposure interpretation: {classified.get('exposure_interpretation', 'not_assessed')}",
        "",
        f"Image size: {analysis['image']['width']} x {analysis['image']['height']}",
        f"Mean luminance: {lum['mean']:.2f}/255",
        f"Contrast (standard deviation): {lum['standard_deviation']:.2f}/255",
        f"P90-P10 luminance span: {lum['p90_p10_span']:.2f}/255",
        f"Estimated noise proxy: {analysis['noise']['noise_estimate_proxy']:.2f}/255",
        "",
        "Detected conditions:"
    ]

    compound_names = set(classified.get("compound_conditions", []))
    basic_conditions = [
        name for name in classified["active_conditions"]
        if name not in compound_names
    ]
    if basic_conditions:
        for name in basic_conditions:
            lines.append(f"- {name.replace('_', ' ').title()}: "
                         f"{classified['explanations'][name]}")
    else:
        lines.append("- No configured condition threshold was crossed.")

    compounds = classified.get("compound_conditions", [])
    lines.extend(["", "Combined-condition analysis:"])
    if compounds:
        for name in compounds:
            lines.append(f"- {name.replace('_', ' ').title()}: {classified['explanations'][name]}")
    else:
        lines.append("- No configured compound-condition rule was triggered.")

    lines.extend(["", "Recommended candidates (not automatically executed):"])
    if recs:
        for item in recs:
            lines.append(
                f"- {item['algorithm_name']} "
                f"(priority {item['priority_score']}/100)"
            )
            for reason in item["reasons"]:
                lines.append(f"  Reason / guidance: {reason}")
    else:
        lines.append("- No algorithm recommendation was produced.")

    lines.extend([
        "",
        "Important: priority scores are rule-based ranking values, not "
        "probabilities or proof of visual quality.",
        "Noise is an approximate residual proxy; inspect results before "
        "drawing conclusions.",
        classified.get("exposure_intent_note", "")
    ])
    return "\n".join(lines)


def build_processing_plan(
    image: np.ndarray,
    max_recommendations: int = 5,
    thresholds: dict[str, float] | None = None
) -> dict[str, Any]:
    """Build a fresh, reviewable plan for the supplied image only."""
    if not isinstance(max_recommendations, int) or isinstance(max_recommendations, bool) or max_recommendations < 1:
        raise ValueError("max_recommendations must be a positive integer.")

    analysis = analyze_image(image)
    classification = classify_image_conditions(analysis, thresholds)
    recommendations = recommend_algorithms(analysis, classification, max_recommendations)
    explanation = generate_explanation(analysis, classification, recommendations)

    cautions = []
    active = set(classification["active_conditions"])
    if classification["exposure_interpretation"] == "dark_dominant_intent_unknown":
        cautions.append(
            "The image is dark-dominant, but ACP cannot know whether this is intentional "
            "(for example, a night or low-key photograph). Review the original before brightening."
        )
    if "mixed_lighting" in active:
        cautions.append(
            "Both dark and bright regions are present. Global brightness or contrast changes "
            "may improve one region while damaging another. Compare local methods carefully."
        )
    if "dark_with_elevated_noise" in active:
        cautions.append(
            "Darkness and an elevated noise proxy occur together. Treat every suggested "
            "parameter as a mild starting point; inspect noise before accepting an output."
        )
    if "high_noise_proxy" in active:
        cautions.append(
            "Noise proxy is elevated. Enhancement may amplify noise; inspect fine details and "
            "consider a classical denoising step before strong enhancement."
        )
    if "black_clipping_risk" in active:
        cautions.append(
            "Some shadow values are near black clipping; enhancement cannot reconstruct "
            "information that is absent from the input."
        )
    if "highlight_clipping_risk" in active:
        cautions.append(
            "Some highlights are near clipping; avoid aggressive brightening."
        )
    if classification["exposure_interpretation"] == "bright_dominant":
        cautions.append(
            "The image is bright-dominant. ACP has withheld routine enhancement recommendations "
            "to avoid unnecessary processing; manual algorithm testing remains available."
        )
    if not cautions:
        cautions.append(
            "Review candidate outputs visually and compare measurements; higher brightness, "
            "contrast, or entropy alone does not prove better quality."
        )

    return {
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "execution_mode": "recommendation_only",
        "automatic_execution": False,
        "analysis": analysis,
        "classification": classification,
        "recommendations": recommendations,
        "cautions": cautions,
        "explanation": explanation,
    }

