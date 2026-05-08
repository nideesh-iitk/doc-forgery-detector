import os
import cv2
import numpy as np
import fitz  # PyMuPDF


def safe_float(value):
    try:
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def extract_image_features(file_path):
    features = {}

    image = cv2.imread(file_path)

    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    file_size_kb = os.path.getsize(file_path) / 1024

    features["file_size_kb"] = safe_float(file_size_kb)
    features["image_width"] = width
    features["image_height"] = height
    features["aspect_ratio"] = safe_float(width / max(height, 1))

    features["mean_brightness"] = safe_float(np.mean(gray))
    features["brightness_std"] = safe_float(np.std(gray))
    features["contrast_score"] = safe_float(gray.max() - gray.min())

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features["blur_score"] = safe_float(laplian_var := laplacian.var())

    edges = cv2.Canny(gray, 100, 200)
    features["edge_density"] = safe_float(np.sum(edges > 0) / edges.size)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = cv2.absdiff(gray, blurred)
    features["noise_score"] = safe_float(np.std(residual))

    patch_scores = []
    h_step = max(height // 4, 1)
    w_step = max(width // 4, 1)

    for y in range(0, height, h_step):
        for x in range(0, width, w_step):
            patch = gray[y:y + h_step, x:x + w_step]
            if patch.size > 0:
                patch_lap = cv2.Laplacian(patch, cv2.CV_64F)
                patch_scores.append(patch_lap.var())

    features["sharpness_variation"] = safe_float(np.std(patch_scores))

    if width > 8:
        block_diff = np.abs(np.diff(gray[:, ::8], axis=1))
        features["jpeg_artifact_score"] = safe_float(np.mean(block_diff))
    else:
        features["jpeg_artifact_score"] = 0.0

    pixel_count = width * height
    features["compression_ratio_estimate"] = safe_float(file_size_kb / max(pixel_count, 1))

    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features["contour_count"] = len(contours)

    features["dark_pixel_ratio"] = safe_float(np.sum(gray < 80) / gray.size)
    features["white_pixel_ratio"] = safe_float(np.sum(gray > 240) / gray.size)

    return features


def extract_pdf_features(file_path):
    features = {}

    file_size_kb = os.path.getsize(file_path) / 1024
    features["file_size_kb"] = safe_float(file_size_kb)

    doc = fitz.open(file_path)

    features["pdf_page_count"] = doc.page_count
    features["pdf_object_count"] = doc.xref_length()

    metadata = doc.metadata or {}

    metadata_fields = [
        "title",
        "author",
        "subject",
        "keywords",
        "creator",
        "producer",
        "creationDate",
        "modDate",
    ]

    missing_count = 0
    for field in metadata_fields:
        if not metadata.get(field):
            missing_count += 1

    features["metadata_missing_count"] = missing_count

    creator = str(metadata.get("creator", "")).lower()
    producer = str(metadata.get("producer", "")).lower()

    suspicious_tools = ["photoshop", "canva", "gimp", "illustrator", "editor", "modified"]
    features["suspicious_creator_flag"] = int(
        any(tool in creator or tool in producer for tool in suspicious_tools)
    )

    creation_date = metadata.get("creationDate", "")
    modified_date = metadata.get("modDate", "")
    features["modified_after_created_flag"] = int(
        bool(creation_date and modified_date and creation_date != modified_date)
    )

    total_text = ""
    embedded_images = 0

    for page in doc:
        total_text += page.get_text()
        embedded_images += len(page.get_images(full=True))

    features["visible_text_length"] = len(total_text)
    features["embedded_image_count"] = embedded_images

    if doc.page_count > 0:
        page = doc[0]
        pix = page.get_pixmap()
        temp_image_path = file_path + "_temp_render.png"
        pix.save(temp_image_path)

        image_features = extract_image_features(temp_image_path)
        os.remove(temp_image_path)

        for key, value in image_features.items():
            features[key] = value

    doc.close()
    return features


def extract_features(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    base_features = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_extension": ext,
    }

    if ext in [".jpg", ".jpeg", ".png"]:
        extracted = extract_image_features(file_path)

        extracted["pdf_page_count"] = 0
        extracted["pdf_object_count"] = 0
        extracted["metadata_missing_count"] = 0
        extracted["suspicious_creator_flag"] = 0
        extracted["modified_after_created_flag"] = 0
        extracted["visible_text_length"] = 0
        extracted["embedded_image_count"] = 0

    elif ext == ".pdf":
        extracted = extract_pdf_features(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    base_features.update(extracted)
    return base_features