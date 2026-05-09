import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier


def train_aadhar_model():
    df = pd.read_csv("data/retrain_feature_store_aadhar.csv")

    print("Dataset shape:", df.shape)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    drop_cols = [
        "file_path",
        "file_name",
        "file_extension",
        "label",
        "category",
        "document_type",
    ]

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["label"]

    # Keep only numeric columns
    X = X.select_dtypes(include=["int64", "float64", "int32", "float32"])

    feature_columns = X.columns.tolist()

    print("\nNumber of training features:", len(feature_columns))
    print("\nFeatures used:")
    print(feature_columns)

    if len(y.value_counts()) < 2:
        raise ValueError("Need both Authentic and Forged samples.")

    min_class_count = y.value_counts().min()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )

    # Optional cross-validation only for reference.
    # This is NOT the final test result.
    if min_class_count >= 2:
        n_splits = min(5, min_class_count)

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )

        print(f"\n{n_splits}-Fold Cross Validation Accuracy:")
        print(scores)
        print("Mean CV Accuracy:", scores.mean())

    # Final training on the full 15-document Aadhaar training set
    model.fit(X, y)

    joblib.dump(model, "models/rf_model.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")

    print("\nFinal Aadhaar-focused model trained on the full training set.")
    print("Model saved to models/rf_model.pkl")
    print("Feature columns saved to models/feature_columns.pkl")
    print("\nNote: Final testing should be done separately on unseen Aadhaar documents.")


if __name__ == "__main__":
    train_aadhar_model()