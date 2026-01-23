import streamlit as st
import pandas as pd
import joblib
import shap
import xgboost

# Load saved models
# Using caching to avoid reloading on every interaction
@st.cache_resource
def load_models():
    lr_pipeline = joblib.load("model/lr_pipeline.joblib")
    rf_pipeline = joblib.load("model/rf_pipeline.joblib")
    xgb_pipeline = joblib.load("model/xgb_pipeline.joblib")
    return lr_pipeline, rf_pipeline, xgb_pipeline

lr_pipeline, rf_pipeline, xgb_pipeline = load_models()

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("Credit Default Risk Predictor")

# Hiding the Deploy button
st.markdown(
    """
    <style>
    .stDeployButton {
            display: none;
        }
    [data-testid="stDeployButton"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# ------------------- ON CLICK PREDICT -------------------
if st.button("Predict"):

    df = pd.DataFrame([input_data], columns=columns)

    # Predictions
    lr_proba = lr_pipeline.predict_proba(df)[0][1]
    rf_proba = rf_pipeline.predict_proba(df)[0][1]
    xgb_proba = xgb_pipeline.predict_proba(df)[0][1]

    lr_pred = int(lr_pipeline.predict(df)[0])
    rf_pred = int(rf_pipeline.predict(df)[0])
    xgb_pred = int(xgb_pipeline.predict(df)[0])

    # Using XGBoost as the primary model for display
    primary_proba = xgb_proba
    primary_pred = xgb_pred

    st.subheader("Prediction Result")
    st.write(f"Default Risk Probability: {primary_proba * 100:.2f}%")

    if primary_pred == 1:
        st.write("Result: Borrower is likely to DEFAULT.")
    else:
        st.write("Result: Borrower is unlikely to default.")

    # ------------------- TOP 3 REASONS -------------------
    st.subheader("Top 3 Reasons (XGBoost)")

    feature_names = {
        "LIMIT_BAL": "Credit Limit",
        "AGE": "Age",
        "PAY_0": "Last month payment status",
        "PAY_2": "2 months ago payment status",
        "PAY_3": "3 months ago payment status",
        "BILL_AMT1": "Last bill amount",
        "BILL_AMT2": "Previous bill amount",
        "PAY_AMT1": "Amount paid last month",
        "NUM_OF_CARDS": "Number of cards",
        "ANNUAL_INCOME": "Annual Income"
    }

    try:
        # Extract model and preprocessor
        model = xgb_pipeline.named_steps["model"]
        preprocessor = xgb_pipeline.named_steps["pre"]

        # Transform input
        X_transformed = preprocessor.transform(df)

        # SHAP Explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

        # Handle SHAP output format
        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]

        feature_importance = pd.DataFrame({
            'feature': columns,
            'shap_value': vals
        })

        feature_importance['abs_value'] = feature_importance['shap_value'].abs()
        top3 = feature_importance.sort_values(by='abs_value', ascending=False).head(3)

        for _, row in top3.iterrows():
            feature = row['feature']
            shap_val = row['shap_value']

            # Get the value
            feature_value = df[feature].iloc[0]
            # Get friendly name
            friendly_name = feature_names.get(feature, feature)

            # SHAP value > 0 pushes prediction towards 1 (Default)
            # SHAP value < 0 pushes prediction towards 0 (No Default)
            if shap_val > 0:
                st.write(f"The value **{feature_value}** for **{friendly_name}** increases the risk of default.")
            else:
                st.write(f"The value **{feature_value}** for **{friendly_name}** decreases the risk of default.")

    except Exception as e:
        st.write("Could not generate explanation.")
        st.write(str(e))

    # ------------------- MODEL COMPARISON -------------------
    st.subheader("Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Prediction": [
            "Default" if lr_pred == 1 else "No Default",
            "Default" if rf_pred == 1 else "No Default",
            "Default" if xgb_pred == 1 else "No Default"
        ],
        "Probability (%)": [
            round(lr_proba * 100, 2),
            round(rf_proba * 100, 2),
            round(xgb_proba * 100, 2)
        ]
    })

    st.table(comparison_df)
