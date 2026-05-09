import os
import shutil
import tempfile
import time

from fastapi import FastAPI, UploadFile, File, HTTPException

from app.feature_extractor import extract_features
from app.evidence_generator import generate_evidence
from app.model import predict_with_model
from app.document_rules import validate_document_rules


app = FastAPI(
    title="Document Forgery Detection API",
    description=(
        "API for detecting forged or tampered documents using forensic features, "
        "semantic validation, and a baseline ML model."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Document Forgery Detection API is running",
        "endpoint": "POST /doc/analyze",
    }


@app.post("/doc/analyze")
async def analyze_document(file: UploadFile = File(...)):
    start_time = time.time()

    allowed_extensions = [".jpg", ".jpeg", ".png", ".pdf"]

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload JPG, JPEG, PNG, or PDF.",
        )

    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Extract visual/PDF forensic features
        features = extract_features(temp_file_path)

        # 2. Run OCR + document-specific semantic validation
        document_rule_result = validate_document_rules(temp_file_path, filename)

        # Add OCR-derived numeric features to final API output.
        # The current ML model ignores these unless retrained with OCR features.
        features.update(document_rule_result.get("ocr_features", {}))

        # 3. Generate rule-based forensic evidence
        rule_result = generate_evidence(features)

        # 4. Generate ML prediction
        ml_result = predict_with_model(features)

        rule_score = rule_result["authenticity_score"]
        rule_label = rule_result["risk_label"]

        ml_prediction = ml_result["ml_prediction"]
        ml_confidence = ml_result["ml_confidence"]

        # 5. Merge normal forensic evidence with semantic/OCR evidence
        evidence = rule_result["evidence"]
        evidence.setdefault("semantic_anomalies", [])
        evidence.setdefault("photo_region_anomalies", [])

        evidence["semantic_anomalies"].extend(
            document_rule_result.get("semantic_anomalies", [])
        )

        evidence["photo_region_anomalies"].extend(
            document_rule_result.get("photo_region_anomalies", [])
        )

        # 6. ML-first decision strategy
        # ML-first decision strategy, but rule-based forensic evidence can soften weak ML decisions.

        if ml_prediction == "Authentic":
            final_label = "Authentic"
            authenticity_score = int(ml_confidence * 100)

            # If ML confidence is weak and rule-based forensic evidence is suspicious,
            # do not allow the final output to remain confidently Authentic.
            if rule_score < 60 and ml_confidence < 0.75:
                final_label = "Suspicious"
                authenticity_score = min(authenticity_score, 60)

            # If rules are extremely suspicious, force Suspicious even if ML is moderately confident.
            elif rule_score < 45:
                final_label = "Suspicious"
                authenticity_score = min(authenticity_score, 55)

        elif ml_prediction == "Forged":
            final_label = "Forged"
            authenticity_score = int((1 - ml_confidence) * 100)

            # If model is not very confident, soften final output to Suspicious.
            if ml_confidence < 0.65:
                final_label = "Suspicious"
                authenticity_score = max(40, authenticity_score)

        else:
            final_label = rule_label
            authenticity_score = rule_score
        # 7. Semantic validation override
        # Critical rules, such as invalid Aadhaar number length, can override ML.
        decision_override = document_rule_result.get("decision_override")

        if decision_override == "Forged":
            final_label = "Forged"
            authenticity_score = min(authenticity_score, 10)

        elif decision_override == "Suspicious" and final_label == "Authentic":
            final_label = "Suspicious"
            authenticity_score = min(authenticity_score, 60)

        tampering_score = 100 - authenticity_score
        processing_time_seconds = round(time.time() - start_time, 4)

        response = {
            "filename": filename,

            # Assignment-compatible fields
            "authenticity_score": authenticity_score,
            "tampering_score": tampering_score,
            "risk_label": final_label,

            # Clearer final score fields
            "final_authenticity_score": authenticity_score,
            "final_tampering_risk_score": tampering_score,

            # ML output
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
            "class_probabilities": ml_result["class_probabilities"],

            # Rule-based forensic output
            "rule_based_authenticity_score": rule_score,
            "rule_based_tampering_risk_score": 100 - rule_score,
            "rule_based_label": rule_label,

            # OCR and semantic validation output
            "document_type": document_rule_result.get("document_type"),
            "semantic_risk_points": document_rule_result.get("semantic_risk_points"),
            "semantic_decision_override": decision_override,
            "ocr_available": document_rule_result.get("ocr_available"),
            "ocr_error_flag": document_rule_result.get("ocr_error_flag"),
            "ocr_error_message": document_rule_result.get("ocr_error_message"),
            "ocr_text_preview": document_rule_result.get("ocr_text_preview"),
            "aadhaar_candidates": document_rule_result.get("aadhaar_candidates"),
            "pan_matches": document_rule_result.get("pan_matches"),

            # Suspicious region output
            "suspicious_regions": document_rule_result.get("suspicious_regions", []),

            # Runtime
            "processing_time_seconds": processing_time_seconds,

            "interpretation_note": (
                "The final risk_label uses an ML-first strategy, with document-specific "
                "semantic rules used as strong override signals where applicable. "
                "Rule-based evidence is used as supporting forensic explanation. "
                "In this project, Authentic means no strong tampering pattern was detected "
                "within the synthetic dataset context; it does not verify legal genuineness."
            ),

            "evidence": evidence,
            "features": features,
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)