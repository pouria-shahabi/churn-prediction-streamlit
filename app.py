


# ------------------------------
# Imports
# ------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from preprocessing import to_numeric, clean_internet

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# ------------------------------
# Load model with cache
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load('Churn_model.pkl')

model = load_model()

# ------------------------------
# Columns
# ------------------------------
columns = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

# Columns used in model (PhoneService dropped)
model_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

# Empty DataFrame for manual input
manual_input = pd.DataFrame(columns=columns)

# ------------------------------
# Upload CSV
# ------------------------------
uploaded_file = st.file_uploader("📂 آپلود فایل CSV", type="csv")
data = None
manual_data = None

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("🔍 پیش‌نمایش داده CSV")
    st.dataframe(data.head())

# ------------------------------
# Manual input form
# ------------------------------
st.subheader("🖊️ وارد کردن دستی اطلاعات مشتری")

with st.form("manual_input_form"):

    # Categorical inputs
    gender = st.selectbox("جنسیت مشتری", ["Male", "Female"])
    SeniorCitizen = st.selectbox("آیا مشتری سالمند است؟", [0, 1])
    Partner = st.selectbox("آیا مشتری پارتنر دارد؟", ["Yes", "No"])
    Dependents = st.selectbox("آیا مشتری افراد تحت تکفل دارد؟", ["Yes", "No"])
    PhoneService = st.selectbox("آیا سرویس تلفن دارد؟", ["Yes", "No"])
    MultipleLines = st.selectbox("چند خط تلفن دارد؟", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox("نوع سرویس اینترنت", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("سرویس امنیت آنلاین", ["No internet service", "No", "Yes"])
    OnlineBackup = st.selectbox("سرویس بکاپ آنلاین", ["No internet service", "No", "Yes"])
    DeviceProtection = st.selectbox("بیمه/محافظت دستگاه", ["No internet service", "No", "Yes"])
    TechSupport = st.selectbox("پشتیبانی فنی", ["No internet service", "No", "Yes"])
    StreamingTV = st.selectbox("سرویس استریم تلویزیون", ["No internet service", "No", "Yes"])
    StreamingMovies = st.selectbox("سرویس استریم فیلم", ["No internet service", "No", "Yes"])
    Contract = st.selectbox("نوع قرارداد", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("صورتحساب الکترونیکی است؟", ["Yes", "No"])
    PaymentMethod = st.selectbox("روش پرداخت", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

    # Numeric inputs
    tenure = st.number_input("مدت زمان همکاری مشتری (ماه)", min_value=0, max_value=100, value=12)
    MonthlyCharges = st.number_input("پرداخت ماهانه", min_value=0.0, max_value=10000.0, value=100.0)
    TotalCharges = st.number_input("مجموع پرداختی تا الان", min_value=0.0, max_value=100000.0, value=1200.0)

    # Submit button
    submitted = st.form_submit_button("ثبت اطلاعات")

    if submitted:
        manual_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [SeniorCitizen],
            "Partner": [Partner],
            "Dependents": [Dependents],
            "tenure": [tenure],
            "PhoneService": [PhoneService],
            "MultipleLines": [MultipleLines],
            "InternetService": [InternetService],
            "OnlineSecurity": [OnlineSecurity],
            "OnlineBackup": [OnlineBackup],
            "DeviceProtection": [DeviceProtection],
            "TechSupport": [TechSupport],
            "StreamingTV": [StreamingTV],
            "StreamingMovies": [StreamingMovies],
            "Contract": [Contract],
            "PaperlessBilling": [PaperlessBilling],
            "PaymentMethod": [PaymentMethod],
            "MonthlyCharges": [MonthlyCharges],
            "TotalCharges": [TotalCharges]
        })

        st.success("✅ اطلاعات دستی ثبت شد!")

# ------------------------------
# Prepare final data
# ------------------------------
if data is not None and manual_data is not None:
    final_data = pd.concat([data, manual_data], ignore_index=True)
elif data is not None:
    final_data = data.copy()
elif manual_data is not None:
    final_data = manual_data.copy()
else:
    final_data = None

# Keep only model features and predict
if final_data is not None:
    final_data = final_data[model_features]
    predictions = model.predict(final_data)
    final_data["Churn_Prediction"] = predictions

# ------------------------------
# Display results with color
# ------------------------------
def color_churn(val):
    return 'background-color: lightgreen'  # سبز برای همه جواب‌ها

st.subheader("📊 نتایج پیش‌بینی")
if final_data is not None:
    st.dataframe(final_data.style.applymap(color_churn, subset=["Churn_Prediction"]))
else:
    st.warning("⚠️ داده‌ای برای پیش‌بینی وجود ندارد!")
