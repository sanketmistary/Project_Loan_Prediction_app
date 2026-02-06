# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Load Model, Scaler, Encoders
# -------------------------------
with open("loan_approval_model.pkl", "rb") as f:
    model, scaler, encoders = pickle.load(f)

# -------------------------------
# Header Section
# -------------------------------
st.title("🏦 Loan Approval Prediction App by Sanket")
st.markdown(
    """
    ### Welcome!
    This app helps predict whether a loan application is likely to be **approved or rejected**  
    based on applicant details.  
    Fill in the information in the sidebar and click **Predict** to see the result.
    """
)

st.divider()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("📋 Applicant Details")

gender = st.sidebar.radio("Gender", ["Male", "Female"])
married = st.sidebar.radio("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.radio("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.radio("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, step=100)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, step=100)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=10)
loan_amount_term = st.sidebar.selectbox(
    "Loan Amount Term (months)", [12, 36, 60, 120, 180, 240, 300, 360, 480]
)
credit_history = st.sidebar.radio("Credit History", [0, 1])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# -------------------------------
# Create Input DataFrame
# -------------------------------
input_dict = {
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area],
}

df_input = pd.DataFrame(input_dict)

# Apply label encoding
for col in df_input.columns:
    if col in encoders:
        df_input[col] = encoders[col].transform(df_input[col].astype(str))

# Scale numerical features
df_input_scaled = scaler.transform(df_input)

# -------------------------------
# Prediction Section
# -------------------------------
st.subheader("🔮 Prediction Result")

if st.button("Predict Loan Approval", use_container_width=True):
    prediction = model.predict(df_input_scaled)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df_input_scaled)[0][1]

    if prediction == 1:
        st.success("✅ Loan Approved! 🎉")
        if prob is not None:
            st.progress(int(prob * 100))
            st.write(f"**Approval Probability:** {prob:.2%}")
    else:
        st.error("❌ Loan Rejected")
        if prob is not None:
            st.progress(int(prob * 100))
            st.write(f"**Approval Probability:** {prob:.2%}")

# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption("Built with ❤️ using Streamlit & scikit-learn")