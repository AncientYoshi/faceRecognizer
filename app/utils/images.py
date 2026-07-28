"""OpenCV image decoding with basic decompression safety checks."""

import cv2
import numpy as np
from numpy.typing import NDArray


class ImageDecodeError(ValueError):
    """Raised when uploaded bytes cannot be decoded into a safe color image."""


def decode_image(
    image_bytes: bytes,
    max_pixels: int,
) -> NDArray[np.uint8]:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise ImageDecodeError(
            "The uploaded file could not be decoded as an image."
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageDecodeError("The uploaded image must have three color channels.")

    height, width = image.shape[:2]
    if height * width > max_pixels:
        raise ImageDecodeError(
            f"The decoded image exceeds the {max_pixels:,}-pixel limit."
        )

    return image


def calculate_blur_score(image: NDArray[np.uint8]) -> float:
    """Return the variance of the Laplacian; lower values are usually blurrier."""

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
