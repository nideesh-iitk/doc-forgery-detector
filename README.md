# Document Forgery & Tampering Detection Pipeline

An end-to-end document forgery and tampering detection pipeline built as part of the internship assignment. The system analyzes document images and PDFs, extracts forensic features, detects tampering-related signals, applies a baseline machine learning classifier, and exposes the final result through a FastAPI microservice.

The pipeline is designed to detect signs of manipulation such as compression artifacts, noise inconsistencies, sharpness variations, metadata anomalies, and structural irregularities. It returns a structured JSON response containing an authenticity score, tampering score, risk label, forensic evidence, extracted features, and processing time.

---

## 1. Project Overview

The goal of this project is to build a practical document forgery detection pipeline covering:

- Forensic feature extraction
- Tampering signal detection
- Baseline ML-based scoring
- Rule-based explainability
- API integration
- Testing and documentation

The system supports document categories such as:

- Aadhaar-style documents
- PAN card-style documents
- Bank statements
- Certificates
- ID cards

The current implementation works with image files and includes PDF support through PyMuPDF-based metadata and structural extraction.

Supported input formats:

```txt
.jpg
.jpeg
.png
.pdf
```

---

## 2. Important Dataset Note

This project does **not** use real official or personal documents.

Due to privacy, legal, and security concerns, the dataset consists of synthetic/generated document samples. Therefore, the labels are defined as follows:

| Label | Meaning |
|---|---|
| `Authentic` | Original untampered synthetic/base document |
| `Forged` | Deliberately modified or tampered synthetic document |

In this project, `Authentic` does **not** mean legally genuine or government-verified. It means that no strong tampering pattern was detected within the synthetic dataset context.

The system is therefore a **document tampering detection pipeline**, not a legal document verification system.

---

## 3. Current Dataset Summary

The dataset used for this MVP contains:

```txt
Total documents: 70
Authentic documents: 34
Forged documents: 36
```

The documents are arranged across categories such as Aadhaar, PAN card, bank statement, certificate, and ID card.

The dataset is organized as:

```txt
data/raw/
├── authentic/
│   ├── Aadhar/
│   ├── Bank_Statement/
│   ├── Certificate/
│   ├── ID_card/
│   └── PAN_Card/
│
└── forged/
    ├── Aadhar/
    ├── Bank_Statement/
    ├── Certificate/
    ├── ID_card/
    └── PAN_Card/
```

The raw document files are not pushed to GitHub. This is intentional because even synthetic identity and bank-style documents should not be treated casually in a public repository.

---

## 4. Pipeline Architecture

The system follows this pipeline:

```txt
Input PDF/Image
        ↓
Feature Extraction
        ↓
Feature Store CSV
        ↓
RandomForest Baseline Classifier
        ↓
Rule-Based Evidence Generator
        ↓
ML-first Decision Logic
        ↓
FastAPI JSON Response
```

The final API result is ML-first. The trained RandomForest classifier gives the primary prediction. The rule-based system is used to provide supporting forensic evidence and explanatory signals.

---

## 5. Project Structure

```txt
doc-forgery-detector/
│
├── app/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── evidence_generator.py
│   ├── feature_extractor.py
│   ├── main.py
│   └── model.py
│
├── data/
│   ├── feature_store.csv
│   ├── labels.csv
│   └── raw/                  # ignored by Git
│
├── models/                   # model files generated locally
│   ├── rf_model.pkl           # ignored by Git
│   └── feature_columns.pkl    # ignored by Git
│
├── reports/
│   └── test_report.md
│
├── notebooks/
│   └── demo.ipynb
│
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 6. Forensic Feature Extraction

The project extracts more than 20 forensic features from each document.

The current model uses 23 numeric training features. These features are grouped into visual, compression, structural, and metadata-based indicators.

### 6.1 Basic File and Image Features

| Feature | Description |
|---|---|
| `file_size_kb` | Size of the input file in KB |
| `image_width` | Image width in pixels |
| `image_height` | Image height in pixels |
| `aspect_ratio` | Width divided by height |

These features help detect abnormal resizing, cropping, or export behavior.

### 6.2 Brightness and Contrast Features

| Feature | Description |
|---|---|
| `mean_brightness` | Average brightness of the grayscale image |
| `brightness_std` | Variation in brightness |
| `contrast_score` | Difference between brightest and darkest pixel |

Edited or pasted regions may have lighting or contrast differences compared to the rest of the document.

### 6.3 Sharpness, Blur, and Edge Features

| Feature | Description |
|---|---|
| `blur_score` | Laplacian variance used as a sharpness/blur indicator |
| `sharpness_variation` | Variation in sharpness across image patches |
| `edge_density` | Ratio of edge pixels detected using Canny edge detection |
| `contour_count` | Number of detected visual contours |

Tampered documents may contain pasted text, edited blocks, or inconsistent region sharpness.

### 6.4 Noise and Compression Features

| Feature | Description |
|---|---|
| `noise_score` | Estimate of high-frequency image noise |
| `jpeg_artifact_score` | Heuristic score for JPEG/block compression artifacts |
| `compression_ratio_estimate` | File size relative to pixel count |

Repeated editing and re-exporting can introduce compression inconsistencies and visible block artifacts.

### 6.5 Pixel Distribution Features

| Feature | Description |
|---|---|
| `dark_pixel_ratio` | Ratio of very dark pixels |
| `white_pixel_ratio` | Ratio of very bright pixels |

These help capture abnormal document backgrounds, overlays, stamps, dense text blocks, or unusual visual layouts.

### 6.6 PDF Structural and Metadata Features

| Feature | Description |
|---|---|
| `pdf_page_count` | Number of pages in the PDF |
| `pdf_object_count` | Number of internal PDF objects |
| `metadata_missing_count` | Number of missing metadata fields |
| `suspicious_creator_flag` | Flag for editing-related software in metadata |
| `modified_after_created_flag` | Flag if modification date differs from creation date |
| `visible_text_length` | Length of extractable PDF text |
| `embedded_image_count` | Number of embedded images inside a PDF |

For image inputs, these PDF-specific values are set to zero.

---

## 7. Feature Store

The extracted features are stored in:

```txt
data/feature_store.csv
```

The feature store contains:

- File information
- Category
- Label
- Extracted forensic features

To regenerate the feature store, run:

```bash
python -m app.analyzer
```

This reads `data/labels.csv`, processes the files listed there, extracts features, and writes the output to `data/feature_store.csv`.

---

## 8. Baseline Machine Learning Model

A RandomForest classifier is used as the baseline ML model.

RandomForest was selected because it is lightweight, robust for small tabular datasets, and easy to train quickly.

Model details:

```txt
Model: RandomForestClassifier
Training features: 23 numeric forensic features
Classes: Authentic, Forged
```

The API may additionally output `Suspicious` when the model confidence is low or when the decision is softened.

---

## 9. Model Evaluation

The trained RandomForest model achieved:

```txt
Held-out test accuracy: 77.78%
5-fold cross-validation mean accuracy: 90%
```

The held-out test accuracy is treated as the main evaluation metric because the dataset is small and may contain similar synthetic pairs.

The cross-validation score is reported as supporting evidence, not as a production-level guarantee.

---

## 10. Explainability and Evidence Generation

The system uses rule-based explainability. Instead of only returning a black-box prediction, the API also returns forensic evidence grouped into anomaly categories.

Evidence categories include:

```txt
visual_anomalies
compression_anomalies
structural_anomalies
metadata_anomalies
```

Examples of generated evidence:

```txt
High JPEG artifact score detected, suggesting possible repeated compression or editing.
High sharpness variation detected across image regions, which may indicate pasted or edited areas.
Several PDF metadata fields are missing.
Suspicious creator or producer software detected in metadata.
```

The rule-based module is not treated as the final classifier. It is used to provide supporting forensic explanations.

---

## 11. Decision Strategy

The final decision is ML-first.

The RandomForest model produces the primary prediction and confidence. The rule-based evidence generator provides supporting explanations.

This design was chosen because the dataset consists of synthetic/generated and re-exported images. Even untampered synthetic documents may contain compression artifacts, sharpness variations, or noise patterns. Therefore, rule-based signals are useful for explanation but are not always reliable as a direct final label.

Interpretation of outputs:

| Field | Meaning |
|---|---|
| `risk_label` | Final ML-first decision |
| `ml_prediction` | Raw RandomForest prediction |
| `ml_confidence` | Confidence of the model prediction |
| `rule_based_label` | Label from heuristic forensic rules only |
| `evidence` | Human-readable forensic indicators |
| `features` | Extracted numerical forensic features |

---

## 12. FastAPI Microservice

The project exposes a FastAPI endpoint:

```txt
POST /doc/analyze
```

Input:

```txt
PDF/Image file
```

Output:

```json
{
  "authenticity_score": 0,
  "tampering_score": 100,
  "risk_label": "Forged",
  "evidence": {},
  "features": {}
}
```

The API returns structured JSON containing:

- Authenticity score
- Tampering score
- Final risk label
- ML prediction
- ML confidence
- Rule-based evidence
- Extracted features
- Processing time

---

## 13. Sample API Output

Example forged-document output:

```json
{
  "filename": "AIGen_PAN Card_4.png",
  "authenticity_score": 1,
  "tampering_score": 99,
  "risk_label": "Forged",
  "final_authenticity_score": 1,
  "final_tampering_risk_score": 99,
  "ml_prediction": "Forged",
  "ml_confidence": 0.9868,
  "class_probabilities": {
    "Authentic": 0.013179141564516904,
    "Forged": 0.986820858435483
  },
  "rule_based_authenticity_score": 85,
  "rule_based_tampering_risk_score": 15,
  "rule_based_label": "Authentic",
  "processing_time_seconds": 3.5328,
  "evidence": {
    "visual_anomalies": [],
    "compression_anomalies": [
      "High JPEG artifact score detected, suggesting possible repeated compression or editing."
    ],
    "structural_anomalies": [],
    "metadata_anomalies": []
  }
}
```

In this example, the rule-based module alone detected only a compression anomaly. However, the ML classifier identified the document as forged with high confidence. Therefore, the final label is `Forged`.

---

## 14. Setup Instructions

### 14.1 Create Virtual Environment

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\activate
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\activate
```

### 14.2 Install Dependencies

```bash
pip install -r requirements.txt
```

If installing manually:

```bash
pip install fastapi uvicorn opencv-python pillow pymupdf pandas numpy scikit-learn joblib python-multipart
```

---

## 15. Running the Project

### 15.1 Generate Feature Store

```bash
python -m app.analyzer
```

This creates:

```txt
data/feature_store.csv
```

### 15.2 Train the Model

```bash
python train_model.py
```

This creates the following local files:

```txt
models/rf_model.pkl
models/feature_columns.pkl
```

These model files are ignored by Git and must be regenerated after cloning the repository.

### 15.3 Run the FastAPI Server

```bash
uvicorn app.main:app --reload
```

Open the Swagger UI:

```txt
http://127.0.0.1:8000/docs
```

Use:

```txt
POST /doc/analyze
```

Upload a PDF or image document and execute the request.

---

## 16. API Testing with curl

Example:

```bash
curl -X POST "http://127.0.0.1:8000/doc/analyze" -F "file=@sample.png"
```

On Windows PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/doc/analyze" -F "file=@data/raw/forged/PAN_Card/AIGen_PAN Card_4.png"
```

---

## 17. Assignment Success Criteria Mapping

| Requirement | Status |
|---|---|
| Extract 20+ forensic features | Completed |
| Support PDFs and images | Implemented in code |
| Support Aadhaar, PAN, bank statements, certificates | Covered in synthetic dataset categories |
| Feature store CSV/JSON | `data/feature_store.csv` generated |
| Baseline ML model | RandomForest classifier implemented |
| Model accuracy ≥ 75% | Held-out accuracy: 77.78% |
| Explainability | Rule-based evidence generator implemented |
| FastAPI `/doc/analyze` endpoint | Implemented |
| Structured JSON output | Implemented |
| Evidence with multiple anomaly types | Visual, compression, structural, metadata |
| End-to-end analysis < 5 seconds/document | Example API run: 3.5328 seconds |
| Test report | Provided in `reports/test_report.md` |
| Developer documentation | This README |
| Demo notebook | Provided in `notebooks/demo.ipynb` |

---

## 18. Assumptions

- The dataset uses synthetic documents, not real official documents.
- `Authentic` means untampered synthetic/base document.
- `Forged` means deliberately modified or tampered synthetic document.
- The system detects tampering patterns, not legal genuineness.
- The RandomForest model is a baseline classifier, not a production fraud detector.
- Rule-based evidence is used for explanation and should not be interpreted as a standalone final classifier.

---

## 19. Limitations

- The dataset is small and synthetic.
- The model is not validated on real-world official documents.
- The system does not certify whether a real Aadhaar, PAN card, bank statement, certificate, or ID card is legally genuine.
- Some untampered synthetic documents may naturally contain compression artifacts or generation artifacts.
- Rule-based thresholds may sometimes flag synthetic documents as suspicious even when they are labelled authentic.
- Full pixel-level suspicious region localization is limited in the current MVP.
- Advanced OCR mismatch detection and font inconsistency detection can be improved further.
- Full PDF layer analysis is approximated using structural indicators such as object count and embedded image count.

---

## 20. Future Work

Possible improvements include:

- Stronger OCR-based text consistency checks
- Font inconsistency detection for PDFs
- Suspicious region heatmaps
- Grad-CAM-based tampering localization
- Aadhaar/PAN template matching
- Larger and more diverse dataset
- Robust duplicate removal
- Image EXIF metadata analysis
- AI-generated document detection
- Watermark or copyright detection

---

## 21. Tech Stack

```txt
Python
OpenCV
PIL/Pillow
PyMuPDF
Pandas
NumPy
Scikit-learn
Joblib
FastAPI
Uvicorn
```

---

## 22. Repository Notes

The following files/folders are intentionally ignored:

```txt
venv/
__pycache__/
data/raw/
models/*.pkl
models/*.joblib
```

This keeps the repository clean and avoids pushing raw document images or generated model binaries.

To reproduce the full pipeline after cloning:

```bash
pip install -r requirements.txt
python -m app.analyzer
python train_model.py
uvicorn app.main:app --reload
```

Then open:

```txt
http://127.0.0.1:8000/docs
```
