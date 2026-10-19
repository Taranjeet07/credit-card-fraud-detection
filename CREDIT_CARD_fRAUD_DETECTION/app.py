
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------- SIMPLE STYLING ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
[data-testid="stMetric"] {
    background: #151922;
    border: 1px solid #303642;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("💳 Credit Card Fraud Detection System")
st.caption("ML-powered transaction fraud screening")

st.divider()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    model_path = Path("fraud_model.pkl")
    columns_path = Path("feature_columns.pkl")

    if not model_path.exists() or not columns_path.exists():
        raise FileNotFoundError(
            "Keep fraud_model.pkl and feature_columns.pkl "
            "in the same folder as app.py."
        )

    model = joblib.load(model_path)
    feature_columns = joblib.load(columns_path)

    return model, feature_columns


try:
    model, feature_columns = load_artifacts()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# ---------------- CSV UPLOAD ----------------
st.subheader("📂 Upload Transactions CSV")

st.info(
    "Upload a CSV containing Time"
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.write(
        "Upload your transaction data to detect potentially "
        "fraudulent transactions."
    )
    st.stop()

# ---------------- PROCESS DATA ----------------
try:
    data = pd.read_csv(uploaded_file)

    if data.empty:
        st.error("The uploaded CSV is empty.")
        st.stop()

    missing_columns = [
        col for col in feature_columns
        if col not in data.columns
    ]

    if missing_columns:
        st.error(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    # Use only the features required by the trained model
    X = data[feature_columns].copy()

    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        st.error(
            "Some required feature values are missing or invalid. "
            "Please upload a complete numeric CSV."
        )
        st.stop()

    # ---------------- PREDICTION ----------------
    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = None

    # ---------------- SUMMARY ----------------
    total = len(predictions)
    fraud_count = int(np.sum(predictions == 1))
    legitimate_count = total - fraud_count
    fraud_percentage = fraud_count / total * 100

    st.divider()
    st.subheader("📊 Prediction Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Transactions Checked", f"{total:,}")
    col2.metric("Flagged as Fraud", f"{fraud_count:,}")
    col3.metric("Classified Legitimate", f"{legitimate_count:,}")
    col4.metric("Flagged Percentage", f"{fraud_percentage:.2f}%")

    # ---------------- RESULTS ----------------
    st.divider()
    st.subheader("🔍 Transaction Results")

    results = pd.DataFrame({
        "Prediction": np.where(
            predictions == 1, "Fraud", "Legitimate"
        )
    })

    if probabilities is not None:
        results["Fraud Probability (%)"] = (
            probabilities * 100
        ).round(2)

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    # ---------------- DOWNLOAD ----------------
    st.download_button(
        label="⬇️ Download Prediction Report",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="fraud_prediction_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "Predictions are model-generated flags, not confirmation "
        "of fraud. Fraud probability is the model's estimated score."
    )

except Exception as e:
    st.error(f"Could not process the CSV: {e}")
