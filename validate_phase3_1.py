
import cv2
import numpy as np

from core.advanced_processing import (
    MODULE_VERSION,
    classical_denoise,
    edge_preserving_sharpen,
    illumination_correction,
    preserve_original_chroma,
    apply_selected_components,
    get_component_descriptions
)


def make_test_image():

    height, width = 240, 320

    y, x = np.mgrid[0:height, 0:width]

    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    image[:, :, 0] = np.clip(
        35 + x * 0.35, 0, 255
    ).astype(np.uint8)

    image[:, :, 1] = np.clip(
        25 + y * 0.45, 0, 255
    ).astype(np.uint8)

    image[:, :, 2] = np.clip(
        45 + (x + y) * 0.25, 0, 255
    ).astype(np.uint8)

    cv2.rectangle(
        image, (70, 55), (245, 185),
        (180, 210, 230), -1
    )

    cv2.circle(
        image, (155, 120), 38,
        (30, 80, 220), -1
    )

    return image


def check(name, condition):

    if not condition:
        raise AssertionError(name)

    print(f"PASS: {name}")


def main():

    print(
        f"ATHER Phase 3.1 — Version {MODULE_VERSION}"
    )

    image = make_test_image()

    rng = np.random.default_rng(42)

    noisy = np.clip(
        image.astype(np.int16) +
        rng.normal(0, 12, image.shape),
        0,
        255
    ).astype(np.uint8)

    # 1. Denoising
    denoised = classical_denoise(noisy)

    check(
        "Denoising output dimensions",
        denoised.shape == image.shape
    )

    check(
        "Denoising output data type",
        denoised.dtype == np.uint8
    )

    check(
        "Zero-strength denoising identity",
        np.array_equal(
            classical_denoise(image, 0),
            image
        )
    )

    # 2. Sharpening
    sharpened = edge_preserving_sharpen(image)

    check(
        "Sharpening output dimensions",
        sharpened.shape == image.shape
    )

    check(
        "Zero-strength sharpening identity",
        np.array_equal(
            edge_preserving_sharpen(image, 0),
            image
        )
    )

    # 3. Illumination correction
    corrected = illumination_correction(image)

    check(
        "Illumination output dimensions",
        corrected.shape == image.shape
    )

    check(
        "Zero-strength illumination identity",
        np.array_equal(
            illumination_correction(image, 0),
            image
        )
    )

    # 4. Chroma preservation
    chroma = preserve_original_chroma(
        image, corrected
    )

    check(
        "Chroma output dimensions",
        chroma.shape == image.shape
    )

    check(
        "Zero-strength chroma identity",
        np.array_equal(
            preserve_original_chroma(
                image, corrected, 0
            ),
            corrected
        )
    )

    original_lab = cv2.cvtColor(
        image, cv2.COLOR_BGR2LAB
    )

    chroma_lab = cv2.cvtColor(
        chroma, cv2.COLOR_BGR2LAB
    )

    # Allow one intensity level for BGR/LAB integer rounding.
    chroma_difference = np.abs(
        chroma_lab[:, :, 1:].astype(np.int16) -
        original_lab[:, :, 1:].astype(np.int16)
    )

    check(
        "Original chroma retained within rounding tolerance",
        int(chroma_difference.max()) <= 1
    )

    # Grayscale support
    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    check(
        "Grayscale denoising",
        classical_denoise(gray).shape == gray.shape
    )

    check(
        "Grayscale sharpening",
        edge_preserving_sharpen(gray).shape == gray.shape
    )

    check(
        "Grayscale illumination correction",
        illumination_correction(gray).shape == gray.shape
    )

    # Explicit processing pipeline
    pipeline, applied = apply_selected_components(
        image,
        denoise=True,
        illumination=True,
        sharpen=True
    )

    check(
        "Explicit processing pipeline",
        pipeline.shape == image.shape
    )

    check(
        "Selected component reporting",
        applied == [
            "denoising",
            "illumination",
            "sharpening"
        ]
    )

    check(
        "Component descriptions",
        len(get_component_descriptions()) == 4
    )

    # Invalid input tests
    invalid_tests = [
        (
            "Invalid denoising strength",
            lambda: classical_denoise(image, 2)
        ),
        (
            "Invalid sharpening amount",
            lambda: edge_preserving_sharpen(image, -1)
        ),
        (
            "Invalid illumination gain",
            lambda: illumination_correction(
                image, max_gain=4
            )
        ),
        (
            "Empty image",
            lambda: classical_denoise(
                np.array([], dtype=np.uint8)
            )
        ),
        (
            "Incorrect image data type",
            lambda: classical_denoise(
                image.astype(np.float32)
            )
        ),
        (
            "Mismatched chroma dimensions",
            lambda: preserve_original_chroma(
                image, image[:100, :100]
            )
        )
    ]

    for name, function in invalid_tests:

        try:
            function()

        except (TypeError, ValueError):

            print(f"PASS: {name}")

        else:

            raise AssertionError(
                f"Expected rejection: {name}"
            )

    print("\nALL PHASE 3.1 CHECKS PASSED")


if __name__ == "__main__":
    main()
