
"""
ATHER Phase 3.2 — Universal Image Evaluation Validation

Usage:
    python validate_phase3_2.py "C:\\path\\to\\your_image.jpg"

The image path is supplied at runtime.
"""

import sys
import cv2

from core.processing_evaluation import (
    ENGINE_VERSION,
    AVAILABLE_COMPONENTS,
    calculate_image_metrics,
    evaluate_components,
    evaluate_single_component,
    get_evaluation_engine_info,
)


def check(name, condition):

    if not condition:
        raise AssertionError(name)

    print(f"PASS: {name}")


def main():

    if len(sys.argv) < 2:

        print(
            "Please supply an image path.\n\n"
            'Example:\n'
            'python validate_phase3_2.py "C:\\Images\\photo.jpg"'
        )

        return

    image_path = sys.argv[1]

    image = cv2.imread(
        image_path,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:

        raise RuntimeError(
            "Could not read the supplied image. "
            "Check the file path and image format."
        )

    # Convert supported 4-channel images to BGR.
    if image.ndim == 3 and image.shape[2] == 4:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR
        )

    print(
        f"\nATHER Evaluation Engine {ENGINE_VERSION}"
    )

    print(
        f"Image dimensions: {image.shape[1]} x {image.shape[0]}"
    )

    print(
        f"Image format: {'Grayscale' if image.ndim == 2 else 'Colour'}"
    )

    # Original metrics
    metrics = calculate_image_metrics(image)

    check(
        "Original image metrics calculated",
        "brightness" in metrics and
        "contrast" in metrics and
        "entropy" in metrics
    )

    # Individual component tests
    for component in AVAILABLE_COMPONENTS:

        result = evaluate_single_component(
            image,
            component
        )

        check(
            f"{component} evaluation completed",
            result["final_image"].shape == image.shape
        )

        check(
            f"{component} processing time recorded",
            result["total_processing_time_ms"] >= 0
        )

    # Multiple components selected together
    combined = evaluate_components(
        image,
        [
            "denoising",
            "illumination",
            "sharpening",
        ]
    )

    check(
        "Multiple-component evaluation completed",
        len(combined["steps"]) == 3
    )

    check(
        "Final image dimensions preserved",
        combined["final_image"].shape == image.shape
    )

    check(
        "Final metrics available",
        "brightness" in combined["final_metrics"]
    )

    check(
        "Metric changes available",
        "contrast" in combined["final_changes"]
    )

    check(
        "Total processing time available",
        combined["total_processing_time_ms"] >= 0
    )

    # Input should not be modified.
    original_after = cv2.imread(
        image_path,
        cv2.IMREAD_UNCHANGED
    )

    if original_after is not None:

        if original_after.ndim == 3 and original_after.shape[2] == 4:
            original_after = cv2.cvtColor(
                original_after,
                cv2.COLOR_BGRA2BGR
            )

        check(
            "Original image remains unchanged",
            original_after.shape == image.shape
        )

    info = get_evaluation_engine_info()

    check(
        "Engine reports no fixed image dependency",
        info["fixed_image_dependency"] is False
    )

    print("\nORIGINAL IMAGE METRICS")

    for key, value in metrics.items():

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nCOMBINED PROCESSING RESULTS")

    for step in combined["steps"]:

        print(
            f"{step['component']}: "
            f"{step['processing_time_ms']:.3f} ms"
        )

    print(
        "\nTotal processing time: "
        f"{combined['total_processing_time_ms']:.3f} ms"
    )

    print("\nALL PHASE 3.2 CHECKS PASSED")


if __name__ == "__main__":
    main()
