import pandas as pd

from app.feature_extractor import extract_features
from app.document_rules import validate_document_rules
from app.evidence_generator import generate_evidence


def get_aadhaar_feature_summary(aadhaar_candidates):
    lengths = [len(str(candidate)) for candidate in aadhaar_candidates]

    valid_count = sum(1 for length in lengths if length == 12)
    invalid_count = sum(1 for length in lengths if length != 12)

    return {
        "aadhaar_candidate_count": len(aadhaar_candidates),
        "valid_aadhaar_candidate_count": valid_count,
        "invalid_aadhaar_candidate_count": invalid_count,
        "aadhaar_has_valid_12_digit": int(valid_count > 0),
        "aadhaar_has_invalid_length": int(invalid_count > 0),
        "aadhaar_max_candidate_length": max(lengths) if lengths else 0,
        "aadhaar_min_candidate_length": min(lengths) if lengths else 0,
    }


def get_pan_feature_summary(pan_matches):
    return {
        "pan_match_count": len(pan_matches),
        "pan_format_valid": int(len(pan_matches) > 0),
    }


def get_photo_region_feature_summary(suspicious_regions):
    photo_regions = [
        region for region in suspicious_regions
        if region.get("type") == "photo_region"
    ]

    if not photo_regions:
        return {
            "photo_region_anomaly_flag": 0,
            "photo_region_blur_ratio": 0.0,
            "photo_region_noise_difference": 0.0,
            "photo_region_brightness_std_difference": 0.0,
        }

    region = photo_regions[0]

    return {
        "photo_region_anomaly_flag": 1,
        "photo_region_blur_ratio": region.get("blur_ratio", 0.0),
        "photo_region_noise_difference": region.get("noise_difference", 0.0),
        "photo_region_brightness_std_difference": region.get("brightness_std_difference", 0.0),
    }


def get_evidence_count_features(evidence):
    visual_count = len(evidence.get("visual_anomalies", []))
    compression_count = len(evidence.get("compression_anomalies", []))
    structural_count = len(evidence.get("structural_anomalies", []))
    metadata_count = len(evidence.get("metadata_anomalies", []))
    semantic_count = len(evidence.get("semantic_anomalies", []))
    photo_count = len(evidence.get("photo_region_anomalies", []))

    total_count = (
        visual_count
        + compression_count
        + structural_count
        + metadata_count
        + semantic_count
        + photo_count
    )

    return {
        "visual_anomaly_count": visual_count,
        "compression_anomaly_count": compression_count,
        "structural_anomaly_count": structural_count,
        "metadata_anomaly_count": metadata_count,
        "semantic_anomaly_count": semantic_count,
        "photo_region_anomaly_count": photo_count,
        "total_anomaly_count": total_count,
    }


def build_feature_store(labels_csv="data/labels.csv", output_csv="data/feature_store.csv"):
    labels_df = pd.read_csv(labels_csv)

    rows = []

    for idx, row in labels_df.iterrows():
        file_path = row["path"]
        label = row["label"]
        category = row["category"]

        print(f"[{idx + 1}/{len(labels_df)}] Processing: {file_path}")

        try:
            # Base visual/compression/PDF features
            features = extract_features(file_path)

            # OCR + document-specific semantic validation
            document_rule_result = validate_document_rules(
                file_path,
                features.get("file_name", "")
            )

            ocr_features = document_rule_result.get("ocr_features", {})
            features.update(ocr_features)

            # Aadhaar-specific numeric features
            aadhaar_candidates = document_rule_result.get("aadhaar_candidates", [])
            features.update(get_aadhaar_feature_summary(aadhaar_candidates))

            # PAN-specific numeric features
            pan_matches = document_rule_result.get("pan_matches", [])
            features.update(get_pan_feature_summary(pan_matches))

            # OCR-visible text mismatch feature
            visible_text_length = features.get("visible_text_length", 0)
            ocr_text_length = features.get("ocr_text_length", 0)

            if visible_text_length > 0:
                features["ocr_visible_mismatch_ratio"] = abs(
                    ocr_text_length - visible_text_length
                ) / max(visible_text_length, 1)
            else:
                features["ocr_visible_mismatch_ratio"] = 0.0

            # Semantic score features
            features["semantic_risk_points"] = document_rule_result.get(
                "semantic_risk_points", 0
            )

            features["critical_rule_triggered"] = int(
                bool(document_rule_result.get("critical_rule_triggered", False))
            )

            decision_override = document_rule_result.get("decision_override")
            features["semantic_override_forged"] = int(decision_override == "Forged")
            features["semantic_override_suspicious"] = int(decision_override == "Suspicious")

            # Suspicious region features
            suspicious_regions = document_rule_result.get("suspicious_regions", [])
            features["suspicious_region_count"] = len(suspicious_regions)
            features.update(get_photo_region_feature_summary(suspicious_regions))

            # Generate evidence and convert anomaly categories into numeric features
            rule_result = generate_evidence(features)
            evidence = rule_result.get("evidence", {})

            evidence.setdefault("semantic_anomalies", [])
            evidence.setdefault("photo_region_anomalies", [])

            evidence["semantic_anomalies"].extend(
                document_rule_result.get("semantic_anomalies", [])
            )

            evidence["photo_region_anomalies"].extend(
                document_rule_result.get("photo_region_anomalies", [])
            )

            features.update(get_evidence_count_features(evidence))

            # Document type as non-numeric reference column
            features["document_type"] = document_rule_result.get("document_type", "Unknown")

            # Ground-truth labels
            features["label"] = label
            features["category"] = category

            rows.append(features)

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    feature_df = pd.DataFrame(rows)
    feature_df.to_csv(output_csv, index=False)

    print(f"\nFeature store saved to: {output_csv}")
    print(f"Total processed documents: {len(feature_df)}")
    print(f"Feature columns: {len(feature_df.columns)}")

    return feature_df


if __name__ == "__main__":
    build_feature_store()