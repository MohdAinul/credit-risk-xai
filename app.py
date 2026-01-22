import streamlit as st
import pandas as pd
import joblib

# Load saved EBM model
pipeline = joblib.load("model/ebm_pipeline.joblib")
ebm = pipeline.named_steps["ebm"]
lr_pipeline = joblib.load("model/logistic_pipeline.joblib")

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("Credit Default Risk Predictor (Explainable AI)")

st.write("Fill borrower details below and click Predict.")

# ------------------- STREAMLIT INPUTS -------------------
LIMIT_BAL = st.number_input("Credit Limit", value=30000, step=1000)
AGE = st.number_input("Age", value=30, step=1)
PAY_0 = st.number_input("Last month payment status", value=0, step=1)
PAY_2 = st.number_input("2 months ago payment status", value=0, step=1)
PAY_3 = st.number_input("3 months ago payment status", value=0, step=1)
BILL_AMT1 = st.number_input("Last bill amount", value=5000, step=100)
BILL_AMT2 = st.number_input("Previous bill amount", value=4000, step=100)
PAY_AMT1 = st.number_input("Amount paid last month", value=2000, step=100)
NUM_OF_CARDS = st.number_input("Number of cards", value=2, step=1)
ANNUAL_INCOME = st.number_input("Annual Income", value=60000, step=1000)

# Convert all inputs to float
input_data = [
    float(LIMIT_BAL), float(AGE), float(PAY_0), float(PAY_2), float(PAY_3),
    float(BILL_AMT1), float(BILL_AMT2), float(PAY_AMT1),
    float(NUM_OF_CARDS), float(ANNUAL_INCOME)
]

columns = [
    "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3",
    "BILL_AMT1", "BILL_AMT2", "PAY_AMT1",
    "NUM_OF_CARDS", "ANNUAL_INCOME"
]

# ------------------- FEATURE NAME MAPPING -------------------
name_map = {
    "feature_0000": "LIMIT_BAL",
    "feature_0001": "AGE",
    "feature_0002": "PAY_0",
    "feature_0003": "PAY_2",
    "feature_0004": "PAY_3",
    "feature_0005": "BILL_AMT1",
    "feature_0006": "BILL_AMT2",
    "feature_0007": "PAY_AMT1",
    "feature_0008": "NUM_OF_CARDS",
    "feature_0009": "ANNUAL_INCOME"
}

# ------------------- ON CLICK PREDICT -------------------
if st.button("Predict"):

    df = pd.DataFrame([input_data], columns=columns)

    # Prediction
    proba = float(pipeline.predict_proba(df)[0][1])
    pred = int(pipeline.predict(df)[0])

    st.subheader("📌 Prediction Result")
    st.write("Default Risk Probability:", round(proba, 3))

    if pred == 1:
        st.error("⚠️ Borrower is likely to DEFAULT!")
    else:
        st.success("✅ Borrower is unlikely to default.")

    # ------------------- TOP 3 REASONS (SAFE VERSION) -------------------
    st.subheader("📌 Top 3 Reasons")

    try:
        global_exp = ebm.explain_global()
        data = global_exp.data()

        feature_names = data["names"]
        feature_scores = data["scores"]

        reasons = []

        for name, score_list in zip(feature_names, feature_scores):

            # Handle combined features like "feature_0005 & feature_0009"
            if "&" in name:
                parts = [p.strip() for p in name.split("&")]
                mapped_parts = [name_map.get(p, p) for p in parts]
                readable_name = " & ".join(mapped_parts)
            else:
                readable_name = name_map.get(name, name)

            # strongest effect
            if isinstance(score_list, list) or isinstance(score_list, tuple):
                strongest = max(score_list, key=abs)
            else:
                strongest = float(score_list)

            reasons.append((readable_name, strongest))

        top3 = sorted(reasons, key=lambda x: -abs(x[1]))[:3]

        for feature, score in top3:
            if score > 0:
                st.write(f"🔴 {feature}: increases default risk ({round(score,3)})")
            else:
                st.write(f"🟢 {feature}: decreases default risk ({round(score,3)})")

    except Exception as e:
        st.write("⚠️ Could not generate explanation.")
        st.write(e)
# ------------------- MODEL COMPARISON -------------------
    st.subheader("📊 Model Comparison")

    lr_proba = float(lr_pipeline.predict_proba(df)[0][1])
    ebm_proba = float(proba)

    comparison_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Explainable Boosting Machine (EBM)"],
        "Default Probability": [round(lr_proba, 3), round(ebm_proba, 3)]
    })

    st.table(comparison_df)

    if ebm_proba < lr_proba:
        st.success("✅ EBM predicts lower risk and is more reliable due to non-linear learning.")
    else:
        st.info("ℹ️ Logistic Regression predicts lower risk but lacks feature interaction modeling.")