import os
import shutil
import tempfile
import time

from fastapi import FastAPI, UploadFile, File, HTTPException

from app.feature_extractor import extract_features
from app.evidence_generator import generate_evidence
from app.model import predict_with_model


app = FastAPI(
    title="Document Forgery Detection API",
    description="API for detecting forged or tampered documents using forensic features and a baseline ML model.",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Document Forgery Detection API is running",
        "endpoint": "POST /doc/analyze"
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
            detail="Unsupported file type. Please upload JPG, JPEG, PNG, or PDF."
        )

    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        features = extract_features(temp_file_path)

        rule_result = generate_evidence(features)
        ml_result = predict_with_model(features)

        rule_score = rule_result["authenticity_score"]
        rule_label = rule_result["risk_label"]
        ml_prediction = ml_result["ml_prediction"]
        ml_confidence = ml_result["ml_confidence"]

        if ml_prediction == "Authentic":
            final_label = "Authentic"
            authenticity_score = int(ml_confidence * 100)

            if rule_score < 50:
                authenticity_score = max(60, authenticity_score - 15)

        elif ml_prediction == "Forged":
            final_label = "Forged"
            authenticity_score = int((1 - ml_confidence) * 100)

            if ml_confidence < 0.65:
                final_label = "Suspicious"
                authenticity_score = max(40, authenticity_score)

        else:
            final_label = rule_label
            authenticity_score = rule_score

        tampering_score = 100 - authenticity_score
        processing_time_seconds = round(time.time() - start_time, 4)

        response = {
            "filename": filename,

            "authenticity_score": authenticity_score,
            "tampering_score": tampering_score,
            "risk_label": final_label,

            "final_authenticity_score": authenticity_score,
            "final_tampering_risk_score": tampering_score,

            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
            "class_probabilities": ml_result["class_probabilities"],

            "rule_based_authenticity_score": rule_score,
            "rule_based_tampering_risk_score": 100 - rule_score,
            "rule_based_label": rule_label,

            "processing_time_seconds": processing_time_seconds,

            "interpretation_note": (
                "The final risk_label and final scores are primarily based on the trained ML classifier. "
                "Rule-based evidence is used as supporting forensic explanation. "
                "In this project, Authentic means no strong tampering pattern was detected "
                "within the synthetic dataset context; it does not verify legal genuineness."
            ),

            "evidence": rule_result["evidence"],
            "features": features
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)