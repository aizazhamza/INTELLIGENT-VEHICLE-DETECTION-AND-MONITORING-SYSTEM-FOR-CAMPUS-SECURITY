# -----------------------------------------------------------
# Require: RawImage
# Ensure: PreprocessedImage
# -----------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple


def _to_uint8_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure an image is uint8 BGR (0..255) for OpenCV operations."""
    if img is None:
        raise ValueError("RawImage is None")
    if img.dtype == np.uint8:
        return img
    # common case: normalized float [0,1]
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0)
        return (img * 255.0).astype(np.uint8)
    return img.astype(np.uint8)


# 1: function PREPROCESSIMAGE(RawImage)
def PreprocessImage(
    RawImage: np.ndarray,
    target_dimensions: Tuple[int, int] = (640, 640),
    normalize: bool = True,
) -> np.ndarray:

    RawImage = _to_uint8_bgr(RawImage)

    # 2: Denoised ← REDUCENOISE(RawImage)
    Denoised = cv2.GaussianBlur(RawImage, (5, 5), 0)

    # 3: ContrastEnhanced ← ADJUSTCONTRAST(Denoised)
    lab = cv2.cvtColor(Denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    ContrastEnhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # 4: Resized ← RESIZEIMAGE(ContrastEnhanced, TargetDimensions)
    Resized = cv2.resize(ContrastEnhanced, target_dimensions)

    # 5: Normalized ← NORMALIZEPIXELVALUES(Resized)
    if normalize:
        PreprocessedImage = Resized.astype(np.float32) / 255.0
    else:
        PreprocessedImage = Resized

    # 7: return PreprocessedImage
    return PreprocessedImage

# 8: end function
