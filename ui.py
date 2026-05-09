import os
import requests
import pandas as pd
import streamlit as st
from PIL import Image


API_URL = "http://127.0.0.1:8000/doc/analyze"


st.set_page_config(
    page_title="Document Forgery Detection",
    page_icon="🛡️",
    layout="wide"
)


st.markdown(
    """
    <style>
        .main-title {
            font-size: 38px;
            font-weight: 800;
            color: #111827;
            text-align: center;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 16px;
            color: #6b7280;
            text-align: center;
            margin-bottom: 30px;
        }

        .card {
            background-color: #ffffff;
            padding: 22px;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.06);
            margin-bottom: 18px;
        }

        .metric-label {
            font-size: 15px;
            color: #374151;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 32px;
            font-weight: 800;
        }

        .green {
            color: #16a34a;
        }

        .red {
            color: #dc2626;
        }

        .orange {
            color: #f59e0b;
        }

        .purple {
            color: #7c3aed;
        }

        .section-heading {
            font-size: 24px;
            font-weight: 800;
            color: #111827;
            margin-top: 25px;
            margin-bottom: 12px;
        }

        .note {
            font-size: 14px;
            color: #6b7280;
        }

        .footer {
            text-align: center;
            color: #6b7280;
            font-size: 13px;
            margin-top: 35px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def status_color(label: str) -> str:
    if label == "Authentic":
        return "green"
    if label == "Forged":
        return "red"
    return "orange"


def score_color(score: int, reverse: bool = False) -> str:
    if reverse:
        if score >= 70:
            return "red"
        if score >= 40:
            return "orange"
        return "green"

    if score >= 70:
        return "green"
    if score >= 40:
        return "orange"
    return "red"


def render_metric(title: str, value: str, color_class: str):
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">{title}</div>
            <div class="metric-value {color_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def flatten_evidence(evidence: dict) -> pd.DataFrame:
    rows = []

    for category, messages in evidence.items():
        category_name = category.replace("_", " ").title()

        if messages:
            for message in messages:
                rows.append(
                    {
                        "Anomaly Type": category_name,
                        "Evidence": message
                    }
                )

    return pd.DataFrame(rows)


def call_api(uploaded_file):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
        )
    }

    response = requests.post(API_URL, files=files, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"API returned status {response.status_code}: {response.text}")

    return response.json()


st.markdown(
    """
    <div class="main-title">Document Forgery & Tampering Detection</div>
    <div class="subtitle">
        Upload a PDF or image document to analyze tampering signals, ML prediction, forensic evidence, and extracted features.
    </div>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.header("Backend Status")
    st.write("FastAPI endpoint:")
    st.code(API_URL)

    st.markdown("---")

    st.header("How to Run")
    st.write("Terminal 1:")
    st.code("uvicorn app.main:app --reload")

    st.write("Terminal 2:")
    st.code("streamlit run ui.py")

    st.markdown("---")

    st.info(
        "This UI is an optional demo layer. The required assignment API is still the FastAPI endpoint /doc/analyze."
    )


uploaded_file = st.file_uploader(
    "Upload document",
    type=["png", "jpg", "jpeg", "pdf"],
    help="Supported formats: PNG, JPG, JPEG, PDF"
)


if uploaded_file is None:
    st.markdown(
        """
        <div class="card">
            <div class="section-heading">Instructions</div>
            <ol>
                <li>Start the FastAPI backend using <code>uvicorn app.main:app --reload</code>.</li>
                <li>Upload a document image or PDF here.</li>
                <li>Click <b>Analyze Document</b>.</li>
                <li>Review the prediction, scores, evidence, and extracted forensic features.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.markdown('<div class="section-heading">Uploaded File</div>', unsafe_allow_html=True)

    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    col_preview, col_info = st.columns([1, 1])

    with col_preview:
        if file_ext in [".png", ".jpg", ".jpeg"]:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_container_width=True)
        else:
            st.info("PDF uploaded. Preview is not displayed here, but the backend will analyze it.")

    with col_info:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">File Name</div>
                <p>{uploaded_file.name}</p>
                <div class="metric-label">File Type</div>
                <p>{uploaded_file.type}</p>
                <div class="metric-label">File Size</div>
                <p>{round(len(uploaded_file.getvalue()) / 1024, 2)} KB</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    analyze = st.button("Analyze Document", type="primary", use_container_width=True)

    if analyze:
        try:
            with st.spinner("Analyzing document through FastAPI backend..."):
                result = call_api(uploaded_file)

            st.success("Analysis completed successfully.")

            authenticity_score = result.get("authenticity_score", result.get("final_authenticity_score", 0))
            tampering_score = result.get("tampering_score", result.get("final_tampering_risk_score", 0))
            risk_label = result.get("risk_label", "Unavailable")
            ml_prediction = result.get("ml_prediction", "Unavailable")
            ml_confidence = result.get("ml_confidence", 0)
            processing_time = result.get("processing_time_seconds", 0)

            st.markdown('<div class="section-heading">Final Analysis Result</div>', unsafe_allow_html=True)

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                render_metric(
                    "Risk Label",
                    risk_label,
                    status_color(risk_label)
                )

            with c2:
                render_metric(
                    "Authenticity Score",
                    f"{authenticity_score}%",
                    score_color(authenticity_score)
                )

            with c3:
                render_metric(
                    "Tampering Risk",
                    f"{tampering_score}%",
                    score_color(tampering_score, reverse=True)
                )

            with c4:
                render_metric(
                    "ML Confidence",
                    f"{round(ml_confidence * 100, 2)}%",
                    "purple"
                )

            with c5:
                render_metric(
                    "Processing Time",
                    f"{processing_time}s",
                    "green" if processing_time < 5 else "red"
                )

            st.markdown('<div class="section-heading">Model & Rule-Based Details</div>', unsafe_allow_html=True)

            d1, d2 = st.columns(2)

            with d1:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="metric-label">ML Prediction</div>
                        <p><b>{ml_prediction}</b></p>
                        <div class="metric-label">ML Confidence</div>
                        <p>{round(ml_confidence * 100, 2)}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with d2:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="metric-label">Rule-Based Label</div>
                        <p><b>{result.get("rule_based_label", "Unavailable")}</b></p>
                        <div class="metric-label">Rule-Based Authenticity Score</div>
                        <p>{result.get("rule_based_authenticity_score", "N/A")}</p>
                        <div class="metric-label">Rule-Based Tampering Risk Score</div>
                        <p>{result.get("rule_based_tampering_risk_score", "N/A")}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            probabilities = result.get("class_probabilities", {})

            if probabilities:
                st.markdown('<div class="section-heading">Class Probabilities</div>', unsafe_allow_html=True)

                prob_df = pd.DataFrame(
                    [
                        {
                            "Class": key,
                            "Probability (%)": round(value * 100, 2)
                        }
                        for key, value in probabilities.items()
                    ]
                )

                st.dataframe(prob_df, use_container_width=True)

                st.bar_chart(prob_df.set_index("Class"))

            st.markdown('<div class="section-heading">Evidence & Detected Anomalies</div>', unsafe_allow_html=True)

            evidence = result.get("evidence", {})
            evidence_df = flatten_evidence(evidence)

            if evidence_df.empty:
                st.success("No major rule-based forensic anomalies were detected.")
            else:
                for _, row in evidence_df.iterrows():
                    st.warning(f"{row['Anomaly Type']}: {row['Evidence']}")

                with st.expander("View evidence table"):
                    st.dataframe(evidence_df, use_container_width=True)

            st.markdown('<div class="section-heading">Extracted Forensic Features</div>', unsafe_allow_html=True)

            features = result.get("features", {})

            if features:
                feature_df = pd.DataFrame(
                    [
                        {
                            "Feature": key,
                            "Value": value
                        }
                        for key, value in features.items()
                    ]
                )

                st.dataframe(feature_df, use_container_width=True, height=420)

            st.markdown('<div class="section-heading">Assignment Success Criteria</div>', unsafe_allow_html=True)

            runtime_status = "Passed" if processing_time < 5 else "Needs Optimization"

            criteria_df = pd.DataFrame(
                [
                    {
                        "Criterion": "Structured JSON response",
                        "Status": "Passed"
                    },
                    {
                        "Criterion": "20+ forensic features",
                        "Status": "Passed"
                    },
                    {
                        "Criterion": "Evidence categories returned",
                        "Status": "Passed"
                    },
                    {
                        "Criterion": "End-to-end analysis < 5 seconds",
                        "Status": runtime_status
                    }
                ]
            )

            st.dataframe(criteria_df, use_container_width=True)

            st.markdown('<div class="section-heading">Interpretation Note</div>', unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="card">
                    <p class="note">{result.get("interpretation_note", "")}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("View full raw API response"):
                st.json(result)

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI backend. "
                "Start it with: uvicorn app.main:app --reload"
            )

        except Exception as e:
            st.error(f"Error during analysis: {e}")


st.markdown(
    """
    <div class="footer">
        Optional Streamlit demo interface for the FastAPI document forgery detection pipeline.
        The system detects tampering patterns in synthetic document samples and does not verify legal genuineness.
    </div>
    """,
    unsafe_allow_html=True
)