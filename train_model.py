import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_model():
    df = pd.read_csv("data/feature_store.csv")

    print("Dataset shape:", df.shape)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    drop_cols = [
        "file_path",
        "file_name",
        "file_extension",
        "label",
        "category"
    ]

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["label"]

    # Keep only numeric columns
    X = X.select_dtypes(include=["int64", "float64", "int32", "float32"])

    feature_columns = X.columns.tolist()

    print("\nNumber of training features:", len(feature_columns))
    print("\nFeatures used:")
    print(feature_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nTest Accuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    min_class_count = y.value_counts().min()
    n_splits = min(5, min_class_count)

    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

        print(f"\n{n_splits}-Fold Cross Validation Accuracy:")
        print(scores)
        print("Mean CV Accuracy:", scores.mean())

    joblib.dump(model, "models/rf_model.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")

    print("\nModel saved to models/rf_model.pkl")
    print("Feature columns saved to models/feature_columns.pkl")


if __name__ == "__main__":
    train_model()