import os
import re
from typing import Dict, List, Tuple, Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image


try:
    import pytesseract

    DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    DEFAULT_TESSDATA_PATH = r"C:\Program Files\Tesseract-OCR\tessdata"

    TESSERACT_CMD = os.getenv("TESSERACT_CMD", DEFAULT_TESSERACT_PATH)
    TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX", DEFAULT_TESSDATA_PATH)

    if os.path.exists(TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    if os.path.exists(TESSDATA_PREFIX):
        os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX

    OCR_AVAILABLE = True

except Exception:
    pytesseract = None
    OCR_AVAILABLE = False


def safe_float(value):
    try:
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def load_image_for_ocr(file_path: str) -> Optional[Image.Image]:
    """
    Loads an image or renders the first page of a PDF for OCR.
    """

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(file_path)

        if doc.page_count == 0:
            doc.close()
            return None

        page = doc[0]

        # Higher scale improves OCR quality.
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        doc.close()
        return image

    if ext in [".jpg", ".jpeg", ".png"]:
        image = Image.open(file_path).convert("RGB")

        max_width = 1000

        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height))

        return image

    return None


def extract_ocr_text(file_path: str) -> Tuple[str, int, str]:
    """
    Returns:
    - OCR text
    - OCR error flag: 0 means OK, 1 means OCR failed/unavailable
    - OCR error message
    """

    if not OCR_AVAILABLE or pytesseract is None:
        return "", 1, "Tesseract OCR is not available"

    try:
        image = load_image_for_ocr(file_path)

        if image is None:
            return "", 1, "Could not load file for OCR"

        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--psm 6"
        )

        return text, 0, ""

    except Exception as e:
        return "", 1, str(e)


def get_ocr_features(ocr_text: str, ocr_error_flag: int) -> Dict:
    digit_count = sum(ch.isdigit() for ch in ocr_text)
    special_char_count = sum(
        not ch.isalnum() and not ch.isspace()
        for ch in ocr_text
    )
    line_count = len([line for line in ocr_text.splitlines() if line.strip()])

    return {
        "ocr_text_length": len(ocr_text),
        "ocr_digit_count": digit_count,
        "ocr_special_char_count": special_char_count,
        "ocr_line_count": line_count,
        "ocr_error_flag": ocr_error_flag,
    }


def infer_document_type(filename: str, ocr_text: str) -> str:
    combined = f"{filename} {ocr_text}".lower()

    if "aadhaar" in combined or "aadhar" in combined or "uidai" in combined:
        return "Aadhaar"

    if "pan" in combined or "permanent account" in combined or "income tax" in combined:
        return "PAN"

    if "bank" in combined or "statement" in combined or "account no" in combined:
        return "Bank Statement"

    if "certificate" in combined:
        return "Certificate"

    if "id card" in combined or "identity" in combined:
        return "ID Card"

    return "Unknown"


def find_digit_candidates(text: str, min_len: int = 12, max_len: int = 16) -> List[str]:
    """
    Finds Aadhaar-like digit sequences, allowing spaces or hyphens.

    Examples:
    - 123456789012
    - 1234 5678 9012
    - 1234-5678-9012
    - 1234 5678 90123
    """

    pattern = re.compile(r"(?<!\d)(?:\d[\s-]?){%d,%d}(?!\d)" % (min_len, max_len))
    raw_matches = pattern.findall(text)

    candidates = []

    for match in raw_matches:
        digits = re.sub(r"\D", "", match)

        if min_len <= len(digits) <= max_len:
            candidates.append(digits)

    return candidates


def validate_aadhaar_rules(ocr_text: str) -> Dict:
    anomalies = []
    risk_points = 0
    critical = False

    # Find digit sequences that look like Aadhaar numbers.
    # Aadhaar is usually 12 digits, often written as 4-4-4.
    candidates = find_digit_candidates(ocr_text, min_len=11, max_len=16)

    valid_12_digit_candidates = [
        candidate for candidate in candidates
        if len(candidate) == 12
    ]

    invalid_length_candidates = [
        candidate for candidate in candidates
        if len(candidate) in [13, 14, 15, 16]
    ]

    near_miss_candidates = [
        candidate for candidate in candidates
        if len(candidate) == 11
    ]

    # If a valid Aadhaar-like 12-digit number is found, no Aadhaar anomaly.
    if valid_12_digit_candidates:
        return {
            "anomalies": [],
            "risk_points": 0,
            "critical": False,
            "aadhaar_candidates": candidates,
        }

    # If OCR clearly found a 13+ digit Aadhaar-like number, this is a strong semantic anomaly.
    if invalid_length_candidates:
        anomalies.append(
            "Invalid Aadhaar-like number length detected. Aadhaar numbers should contain exactly 12 digits."
        )
        risk_points += 80
        critical = True

    # If OCR found an 11-digit near miss, treat it as OCR uncertainty, not definite forgery.
    elif near_miss_candidates:
        anomalies.append(
            "Aadhaar-like number was partially detected, but OCR may have missed one digit. Manual review recommended."
        )
        risk_points += 15
        critical = False

    # If no Aadhaar-like sequence is detected, keep it as a weak OCR warning.
    # Do NOT strongly override the ML model based only on OCR failure.
    else:
        anomalies.append(
            "No clear 12-digit Aadhaar-like number was detected from OCR text. This may be due to OCR quality."
        )
        risk_points += 10
        critical = False

    return {
        "anomalies": anomalies,
        "risk_points": risk_points,
        "critical": critical,
        "aadhaar_candidates": candidates,
    }


def validate_pan_rules(ocr_text: str) -> Dict:
    anomalies = []
    risk_points = 0
    critical = False

    # PAN format: ABCDE1234F
    compact_text = re.sub(r"[^A-Za-z0-9]", "", ocr_text).upper()
    pan_matches = re.findall(r"[A-Z]{5}[0-9]{4}[A-Z]", compact_text)

    if not pan_matches:
        anomalies.append(
            "No valid PAN format detected. Expected format is five letters, four digits, and one letter."
        )
        risk_points += 45

    return {
        "anomalies": anomalies,
        "risk_points": risk_points,
        "critical": critical,
        "pan_matches": pan_matches,
    }


def check_photo_region_anomaly(file_path: str, document_type: str) -> Dict:
    """
    Lightweight photo-region heuristic.

    This does not identify a person. It only checks whether an approximate
    photo region has noticeably different blur/noise/contrast properties
    compared to the full document.
    """

    if document_type not in ["Aadhaar", "PAN", "ID Card"]:
        return {
            "anomalies": [],
            "risk_points": 0,
            "region": None,
        }

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in [".jpg", ".jpeg", ".png"]:
        return {
            "anomalies": [],
            "risk_points": 0,
            "region": None,
        }

    image = cv2.imread(file_path)

    if image is None:
        return {
            "anomalies": [],
            "risk_points": 0,
            "region": None,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Approximate left-side photo region for many Aadhaar/PAN/ID-card style layouts.
    # This is a heuristic, not a face recognition system.
    x1 = int(0.04 * w)
    y1 = int(0.25 * h)
    x2 = int(0.35 * w)
    y2 = int(0.78 * h)

    region = gray[y1:y2, x1:x2]

    if region.size == 0:
        return {
            "anomalies": [],
            "risk_points": 0,
            "region": None,
        }

    full_blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    region_blur = cv2.Laplacian(region, cv2.CV_64F).var()

    full_std = np.std(gray)
    region_std = np.std(region)

    full_noise = np.std(cv2.absdiff(gray, cv2.GaussianBlur(gray, (5, 5), 0)))
    region_noise = np.std(cv2.absdiff(region, cv2.GaussianBlur(region, (5, 5), 0)))

    blur_ratio = region_blur / max(full_blur, 1)
    std_diff = abs(region_std - full_std)
    noise_diff = abs(region_noise - full_noise)

    anomalies = []
    risk_points = 0

    if blur_ratio > 2.8 or blur_ratio < 0.25 or std_diff > 35 or noise_diff > 10:
        anomalies.append(
            "Photo region quality differs noticeably from the rest of the document, indicating possible photo replacement or pasted-region inconsistency."
        )
        risk_points += 30

    return {
        "anomalies": anomalies,
        "risk_points": risk_points,
        "region": {
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "blur_ratio": safe_float(blur_ratio),
            "brightness_std_difference": safe_float(std_diff),
            "noise_difference": safe_float(noise_diff),
        },
    }


def validate_document_rules(file_path: str, filename: str) -> Dict:
    """
    Runs OCR and document-specific validation rules.

    Returns semantic anomaly evidence, optional decision override,
    OCR-derived numeric features, and suspicious region data.
    """

    ocr_text, ocr_error_flag, ocr_error_message = extract_ocr_text(file_path)
    ocr_features = get_ocr_features(ocr_text, ocr_error_flag)

    document_type = infer_document_type(filename, ocr_text)

    semantic_anomalies = []
    photo_region_anomalies = []
    suspicious_regions = []

    semantic_risk_points = 0
    critical_rule_triggered = False

    aadhaar_candidates = []
    pan_matches = []

    if document_type == "Aadhaar":
        aadhaar_result = validate_aadhaar_rules(ocr_text)

        semantic_anomalies.extend(aadhaar_result["anomalies"])
        semantic_risk_points += aadhaar_result["risk_points"]
        critical_rule_triggered = critical_rule_triggered or aadhaar_result["critical"]
        aadhaar_candidates = aadhaar_result["aadhaar_candidates"]

    elif document_type == "PAN":
        pan_result = validate_pan_rules(ocr_text)

        semantic_anomalies.extend(pan_result["anomalies"])
        semantic_risk_points += pan_result["risk_points"]
        critical_rule_triggered = critical_rule_triggered or pan_result["critical"]
        pan_matches = pan_result["pan_matches"]

    photo_result = check_photo_region_anomaly(file_path, document_type)

    photo_region_anomalies.extend(photo_result["anomalies"])
    semantic_risk_points += photo_result["risk_points"]

    if photo_result["region"] is not None and photo_region_anomalies:
        suspicious_regions.append(
            {
                "type": "photo_region",
                "reason": photo_region_anomalies[0],
                **photo_result["region"],
            }
        )

    if critical_rule_triggered:
        decision_override = "Forged"
    elif semantic_risk_points >= 30:
        decision_override = "Suspicious"
    else:
        decision_override = None

    return {
    "document_type": document_type,
    "ocr_available": OCR_AVAILABLE,
    "ocr_error_flag": ocr_error_flag,
    "ocr_error_message": ocr_error_message,
    "ocr_text_preview": ocr_text[:500],
    "ocr_features": ocr_features,
    "semantic_anomalies": semantic_anomalies,
    "photo_region_anomalies": photo_region_anomalies,
    "suspicious_regions": suspicious_regions,
    "semantic_risk_points": semantic_risk_points,
    "critical_rule_triggered": critical_rule_triggered,
    "decision_override": decision_override,
    "aadhaar_candidates": aadhaar_candidates,
    "pan_matches": pan_matches,
}