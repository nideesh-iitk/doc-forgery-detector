import os
import shutil
import tempfile

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

        # Hybrid decision:
        # Rule-based score gives forensic risk.
        # ML prediction gives learned classification.
        authenticity_score = rule_result["authenticity_score"]
        rule_label = rule_result["risk_label"]
        ml_prediction = ml_result["ml_prediction"]

        if ml_prediction == "Forged" and authenticity_score < 75:
            final_label = "Forged" if authenticity_score < 50 else "Suspicious"
        elif ml_prediction == "Authentic" and authenticity_score >= 60:
            final_label = "Authentic"
        elif authenticity_score >= 75:
            final_label = "Authentic"
        elif authenticity_score >= 40:
            final_label = "Suspicious"
        else:
            final_label = "Forged"

        response = {
            "filename": filename,
            "authenticity_score": authenticity_score,
            "risk_label": final_label,
            "rule_based_label": rule_label,
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_result["ml_confidence"],
            "class_probabilities": ml_result["class_probabilities"],
            "evidence": rule_result["evidence"],
            "features": features
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)