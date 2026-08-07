import cv2
import numpy as np
import re
import easyocr

# Initialize EasyOCR once
reader = easyocr.Reader(["en"], gpu=False)

# ------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------

def clean_plate(text: str) -> str:
    """Keep only uppercase letters, digits and hyphens."""
    return re.sub(r"[^A-Z0-9-]", "", text.upper())


def fix_common_ocr_errors(text: str) -> str:
    """
    Fix common OCR mistakes.
    Adjust mappings according to your plate format.
    """
    replacements = {
        "O": "0",
        "Q": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
    }

    return "".join(replacements.get(ch, ch) for ch in text)


def is_valid_plate(text: str) -> bool:
    """
    Basic plate validation.
    Modify according to your country's plate format.
    """
    if not text:
        return False

    if len(text) < 4 or len(text) > 10:
        return False

    return bool(re.fullmatch(r"[A-Z0-9-]+", text))


def is_overexposed(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # np.mean returns a numpy scalar (e.g. numpy.float64 or numpy.bool_ depending
    # on comparison). Explicitly cast to built-in bool for type checkers.
    return bool(np.mean(gray) > 220)


def is_underexposed(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return bool(np.mean(gray) < 40)


# ------------------------------------------------------------------
# Preprocessing Variants
# ------------------------------------------------------------------

def _build_preprocess_variants(crop: np.ndarray) -> list:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    c_img = clahe.apply(gray)

    blur = cv2.GaussianBlur(c_img, (0, 0), 1.0)

    sharp = cv2.addWeighted(
        c_img,
        1.7,
        blur,
        -0.7,
        0
    )

    thresh = cv2.adaptiveThreshold(
        c_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    thresh_inv = cv2.adaptiveThreshold(
        c_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    kernel = np.ones((2, 2), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    thresh_inv = cv2.morphologyEx(
        thresh_inv,
        cv2.MORPH_CLOSE,
        kernel
    )

    variants = [sharp, c_img, thresh, thresh_inv]

    if not (is_overexposed(crop) or is_underexposed(crop)):
        variants[0], variants[1] = variants[1], variants[0]

    return variants


# ------------------------------------------------------------------
# OCR Recognition
# ------------------------------------------------------------------

def recognize_plate_easyocr(crop: np.ndarray) -> tuple[str, float]:
    """
    Recognize a license plate using EasyOCR.

    Returns:
        (plate_text, confidence)
    """
    if crop is None or crop.size == 0:
        return "", 0.0

    best_text = ""
    best_conf = 0.0

    for variant in _build_preprocess_variants(crop):

        results = reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        )

        for _, text, conf in results:
            conf = float(conf)

            cleaned = fix_common_ocr_errors(
                clean_plate(text)
            )

            if is_valid_plate(cleaned) and conf > best_conf:
                best_text = cleaned
                best_conf = conf

        if best_conf >= 0.55:
            break

    return best_text, best_conf


# ------------------------------------------------------------------
# Wrapper
# ------------------------------------------------------------------

def RecognizePlate(image: np.ndarray) -> str:
    """
    Compatibility wrapper returning only text.
    """
    plate, conf = recognize_plate_easyocr(image)
    return plate


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------

if __name__ == "__main__":
    img = cv2.imread("plate.jpg")

    if img is not None:
        plate, confidence = recognize_plate_easyocr(img)

    print("Plate:", plate)
    print("Confidence:", round(confidence, 3))