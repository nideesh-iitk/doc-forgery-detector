import joblib
import pandas as pd
import os


MODEL_PATH = "models/rf_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_COLUMNS_PATH):
        return None, None

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    return model, feature_columns


def predict_with_model(features):
    model, feature_columns = load_model()

    if model is None or feature_columns is None:
        return {
            "ml_prediction": "Unavailable",
            "ml_confidence": 0.0,
            "class_probabilities": {}
        }

    row = {}

    for col in feature_columns:
        row[col] = features.get(col, 0)

    X = pd.DataFrame([row], columns=feature_columns)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    class_probabilities = {
        cls: float(prob)
        for cls, prob in zip(model.classes_, probabilities)
    }

    ml_confidence = float(max(probabilities))

    return {
        "ml_prediction": prediction,
        "ml_confidence": round(ml_confidence, 4),
        "class_probabilities": class_probabilities
    }