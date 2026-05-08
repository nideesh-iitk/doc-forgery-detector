import pandas as pd
from app.feature_extractor import extract_features


def build_feature_store(labels_csv="data/labels.csv", output_csv="data/feature_store.csv"):
    labels_df = pd.read_csv(labels_csv)

    rows = []

    for idx, row in labels_df.iterrows():
        file_path = row["path"]
        label = row["label"]
        category = row["category"]

        print(f"[{idx + 1}/{len(labels_df)}] Processing: {file_path}")

        try:
            features = extract_features(file_path)
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