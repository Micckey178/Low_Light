"""ATHER - Safe image loading and saving (Unicode paths, EXIF orientation)."""

from pathlib import Path

import cv2
import numpy as np


_ORIENTATION_FIXES = {
    2: lambda a: cv2.flip(a, 1),
    3: lambda a: cv2.rotate(a, cv2.ROTATE_180),
    4: lambda a: cv2.flip(a, 0),
    5: lambda a: cv2.flip(cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE), 1),
    6: lambda a: cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE),
    7: lambda a: cv2.flip(cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE), 1),
    8: lambda a: cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE),
}


def _exif_orientation(path):
    try:
        from PIL import Image
        with Image.open(path) as img:
            return int(img.getexif().get(274, 1))
    except Exception:
        return 1


def load_image(path):
    """Load an 8-bit grayscale or BGR image. Raises ValueError on failure."""

    data = np.fromfile(str(path), dtype=np.uint8)

    if data.size == 0:
        raise ValueError("The file is empty or cannot be read.")

    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("Unable to decode the selected image.")

    if image.dtype != np.uint8:
        raise ValueError("Please select an 8-bit image.")

    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif channels == 1:
            image = image[:, :, 0]
        elif channels != 3:
            raise ValueError(
                "Please select a grayscale or three-channel color image."
            )
    elif image.ndim != 2:
        raise ValueError("The selected image format is not supported.")

    fix = _ORIENTATION_FIXES.get(_exif_orientation(path))

    if fix is not None:
        image = fix(image)

    return np.ascontiguousarray(image)


def save_image(path, image):
    """Save an image (extension decides format). Returns True on success."""

    extension = Path(path).suffix.lower() or ".png"

    try:
        success, buffer = cv2.imencode(extension, image)
        if not success:
            return False
        buffer.tofile(str(path))
        return True
    except Exception:
        return False