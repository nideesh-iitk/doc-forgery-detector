def generate_evidence(features):
    evidence = {
        "visual_anomalies": [],
        "compression_anomalies": [],
        "structural_anomalies": [],
        "metadata_anomalies": []
    }

    risk_points = 0

    # ---------------- Visual Forensics ----------------

    blur_score = features.get("blur_score", 0)
    sharpness_variation = features.get("sharpness_variation", 0)
    edge_density = features.get("edge_density", 0)
    noise_score = features.get("noise_score", 0)
    contour_count = features.get("contour_count", 0)
    dark_pixel_ratio = features.get("dark_pixel_ratio", 0)
    white_pixel_ratio = features.get("white_pixel_ratio", 0)

    if blur_score < 50:
        evidence["visual_anomalies"].append(
            "Low blur score detected, indicating possible poor image clarity or reprocessing."
        )
        risk_points += 15

    if sharpness_variation > 1000:
        evidence["visual_anomalies"].append(
            "High sharpness variation detected across image regions, which may indicate pasted or edited areas."
        )
        risk_points += 20

    if edge_density > 0.25:
        evidence["visual_anomalies"].append(
            "High edge density detected, suggesting unusually dense text, borders, or inserted elements."
        )
        risk_points += 10

    if noise_score > 12:
        evidence["visual_anomalies"].append(
            "High noise variation detected, which may indicate inconsistent image generation or editing."
        )
        risk_points += 10

    if contour_count > 3000:
        evidence["visual_anomalies"].append(
            "Large number of contours detected, suggesting possible layout irregularities or excessive inserted components."
        )
        risk_points += 10

    if dark_pixel_ratio > 0.35:
        evidence["visual_anomalies"].append(
            "High dark pixel ratio detected, which may indicate abnormal stamps, overlays, or dense inserted regions."
        )
        risk_points += 10

    if white_pixel_ratio < 0.15:
        evidence["visual_anomalies"].append(
            "Low white pixel ratio detected, suggesting unusual background or document formatting."
        )
        risk_points += 10

    # ---------------- Compression Forensics ----------------

    jpeg_artifact_score = features.get("jpeg_artifact_score", 0)
    compression_ratio = features.get("compression_ratio_estimate", 0)

    if jpeg_artifact_score > 20:
        evidence["compression_anomalies"].append(
            "High JPEG artifact score detected, suggesting possible repeated compression or editing."
        )
        risk_points += 15

    if compression_ratio > 0.01:
        evidence["compression_anomalies"].append(
            "Unusual compression ratio detected compared to image dimensions."
        )
        risk_points += 10

    # ---------------- PDF / Structural Forensics ----------------

    pdf_object_count = features.get("pdf_object_count", 0)
    embedded_image_count = features.get("embedded_image_count", 0)
    visible_text_length = features.get("visible_text_length", 0)

    if pdf_object_count > 100:
        evidence["structural_anomalies"].append(
            "High PDF object count detected, which may indicate complex editing history or embedded objects."
        )
        risk_points += 15

    if embedded_image_count > 10:
        evidence["structural_anomalies"].append(
            "Large number of embedded images detected inside the PDF."
        )
        risk_points += 15

    if visible_text_length == 0 and features.get("pdf_page_count", 0) > 0:
        evidence["structural_anomalies"].append(
            "PDF contains no extractable text, suggesting it may be scanned, flattened, or image-based."
        )
        risk_points += 10

    # ---------------- Metadata Forensics ----------------

    metadata_missing_count = features.get("metadata_missing_count", 0)
    suspicious_creator_flag = features.get("suspicious_creator_flag", 0)
    modified_after_created_flag = features.get("modified_after_created_flag", 0)

    if metadata_missing_count > 5:
        evidence["metadata_anomalies"].append(
            "Several PDF metadata fields are missing."
        )
        risk_points += 10

    if suspicious_creator_flag == 1:
        evidence["metadata_anomalies"].append(
            "Suspicious creator or producer software detected in metadata."
        )
        risk_points += 20

    if modified_after_created_flag == 1:
        evidence["metadata_anomalies"].append(
            "PDF modification timestamp differs from creation timestamp."
        )
        risk_points += 15

    authenticity_score = max(0, 100 - risk_points)

    if authenticity_score >= 75:
        risk_label = "Authentic"
    elif authenticity_score >= 40:
        risk_label = "Suspicious"
    else:
        risk_label = "Forged"

    return {
        "authenticity_score": authenticity_score,
        "risk_label": risk_label,
        "risk_points": risk_points,
        "evidence": evidence
    }