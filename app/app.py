# =============================================================================
# TELCO CUSTOMER CHURN PREDICTION APP
# Member 3 — Streamlit App + SHAP + Business Insights
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION — must be the very first Streamlit command
# =============================================================================
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# =============================================================================
# LOAD MODELS & FEATURE COLUMNS
# =============================================================================


@st.cache_resource
def load_models():
    rf = joblib.load('../models/random_forest_model.pkl')
    xgb = joblib.load('../models/xgboost_model.pkl')
    with open('../models/feature_columns.json') as f:
        cols = json.load(f)
    return rf, xgb, cols


rf_model, xgb_model, feature_columns = load_models()

# =============================================================================
# APP TITLE
# =============================================================================
st.title("📡 Telecom Customer Churn Predictor")
st.markdown(
    "Enter customer details below to predict whether they are likely to churn.")
st.divider()

# =============================================================================
# SIDEBAR — Customer Input Form
# =============================================================================
st.sidebar.header("Customer Details")

# --- Demographics ---
st.sidebar.subheader("Demographics")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])

# --- Account Info ---
st.sidebar.subheader("Account Info")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
contract = st.sidebar.selectbox("Contract Type",
                                ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"])
monthly = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
total = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)

# --- Services ---
st.sidebar.subheader("Services")
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multi = st.sidebar.selectbox(
    "Multiple Lines", ["Yes", "No", "No phone service"])
internet = st.sidebar.selectbox("Internet Service",
                                ["Fiber optic", "DSL", "No"])
online_sec = st.sidebar.selectbox(
    "Online Security", ["Yes", "No", "No internet service"])
online_bak = st.sidebar.selectbox(
    "Online Backup", ["Yes", "No", "No internet service"])
device_prot = st.sidebar.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"])
tech_sup = st.sidebar.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"])
tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
movies = st.sidebar.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"])

# --- Model Selection ---
st.sidebar.divider()
model_choice = st.sidebar.radio("Choose Model", ["Random Forest", "XGBoost"])

# =============================================================================
# FEATURE ENGINEERING — match exactly what Member 2 did
# =============================================================================


def prepare_input():
    # Binary encoding
    data = {
        'gender':           1 if gender == "Male" else 0,
        'SeniorCitizen':    1 if senior == "Yes" else 0,
        'Partner':          1 if partner == "Yes" else 0,
        'Dependents':       1 if dependents == "Yes" else 0,
        'tenure':           tenure,
        'PhoneService':     1 if phone == "Yes" else 0,
        'MultipleLines':    1 if multi == "Yes" else 0,
        'OnlineSecurity':   1 if online_sec == "Yes" else 0,
        'OnlineBackup':     1 if online_bak == "Yes" else 0,
        'DeviceProtection': 1 if device_prot == "Yes" else 0,
        'TechSupport':      1 if tech_sup == "Yes" else 0,
        'StreamingTV':      1 if tv == "Yes" else 0,
        'StreamingMovies':  1 if movies == "Yes" else 0,
        'PaperlessBilling': 1 if paperless == "Yes" else 0,
        'MonthlyCharges':   monthly,
        'TotalCharges':     total,

        # One-hot: InternetService
        'InternetService_DSL':         1 if internet == "DSL" else 0,
        'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
        'InternetService_No':          1 if internet == "No" else 0,

        # One-hot: Contract
        'Contract_Month-to-month': 1 if contract == "Month-to-month" else 0,
        'Contract_One year':       1 if contract == "One year" else 0,
        'Contract_Two year':       1 if contract == "Two year" else 0,

        # One-hot: PaymentMethod
        'PaymentMethod_Bank transfer (automatic)': 1 if payment == "Bank transfer (automatic)" else 0,
        'PaymentMethod_Credit card (automatic)':   1 if payment == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check':          1 if payment == "Electronic check" else 0,
        'PaymentMethod_Mailed check':              1 if payment == "Mailed check" else 0,

        # Engineered features
        'ChargesPerMonth': total / (tenure + 1),
        'ContractRisk':    2 if contract == "Month-to-month" else (1 if contract == "One year" else 0),
    }

    # ServiceCount
    service_cols = ['PhoneService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    data['ServiceCount'] = sum(data[c] for c in service_cols)

    # Build dataframe with exact column order
    df = pd.DataFrame([data])
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


# =============================================================================
# PREDICT BUTTON
# =============================================================================
if st.sidebar.button("🔍 Predict Churn", use_container_width=True):

    input_df = prepare_input()
    model = rf_model if model_choice == "Random Forest" else xgb_model

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # ── Result Display ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("### 🔴 Likely to Churn")
            st.metric("Churn Probability", f"{probability:.1%}")
        else:
            st.success("### 🟢 Not Likely to Churn")
            st.metric("Churn Probability", f"{probability:.1%}")

    with col2:
        st.info(f"**Model used:** {model_choice}")
        st.metric("Retention Probability", f"{1 - probability:.1%}")

    st.divider()

    # ── SHAP Explanation ──────────────────────────────────────────────────────
    st.subheader("🔍 Why did the model make this prediction?")
    st.caption(
        "SHAP values show which features pushed the prediction toward churn (red) or away from churn (blue).")

    with st.spinner("Calculating SHAP explanation... please wait"):
        explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots(figsize=(10, 5))

    # For Random Forest shap_values is a list [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    shap.waterfall_plot(
        shap.Explanation(
            values=sv[0],
            base_values=explainer.expected_value[1] if isinstance(
                explainer.expected_value, list) else explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=feature_columns
        ),
        max_display=10,
        show=False
    )
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── Business Insight ──────────────────────────────────────────────────────
    st.subheader("💼 Business Insight & Recommendation")

    if prediction == 1:
        st.warning("""
        **This customer is at high risk of leaving.**

        Key actions to retain them:
        - 🎯 Offer a **discounted long-term contract** (1 or 2 year)
        - 💰 Provide a **loyalty discount** on monthly charges
        - 🛡️ Bundle **Tech Support or Online Security** for free
        - 📞 Assign a **dedicated customer success manager**
        """)
    else:
        st.success("""
        **This customer is likely to stay.**

        Suggested actions:
        - 🌟 Enrol them in a **loyalty rewards program**
        - 📦 Offer **premium service upgrades** at a discount
        - 📊 Monitor their usage quarterly to catch early warning signs
        """)

else:
    st.info("👈 Fill in customer details in the sidebar and click **Predict Churn** to get started.")

# =============================================================================
# GLOBAL FEATURE IMPORTANCE — shown always on the page
# =============================================================================
st.divider()
st.subheader("📊 Overall Feature Importance")
st.caption(
    "These are the top factors that drive churn across ALL customers — not just one.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Random Forest — Top 10 Features**")
    rf_importance = pd.Series(
        rf_model.feature_importances_,
        index=feature_columns
    ).sort_values(ascending=False).head(10)

    fig_rf, ax_rf = plt.subplots(figsize=(6, 4))
    colors_rf = ['#E05C5C' if i == 0 else '#4A90D9'
                 for i in range(len(rf_importance))]
    ax_rf.barh(
        rf_importance.index[::-1],
        rf_importance.values[::-1],
        color=colors_rf[::-1]
    )
    ax_rf.set_xlabel("Importance Score")
    ax_rf.spines['top'].set_visible(False)
    ax_rf.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_rf)
    plt.close()

with col2:
    st.markdown("**XGBoost — Top 10 Features**")
    xgb_importance = pd.Series(
        xgb_model.feature_importances_,
        index=feature_columns
    ).sort_values(ascending=False).head(10)

    fig_xgb, ax_xgb = plt.subplots(figsize=(6, 4))
    colors_xgb = ['#E05C5C' if i == 0 else '#4A90D9'
                  for i in range(len(xgb_importance))]
    ax_xgb.barh(
        xgb_importance.index[::-1],
        xgb_importance.values[::-1],
        color=colors_xgb[::-1]
    )
    ax_xgb.set_xlabel("Importance Score")
    ax_xgb.spines['top'].set_visible(False)
    ax_xgb.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_xgb)
    plt.close()

st.caption("🔴 Red = most important feature  |  🔵 Blue = other important features")
