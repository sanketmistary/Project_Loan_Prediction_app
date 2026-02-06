# 🏦 Loan Approval Prediction System

This project is a **Mini Project: Loan Prediction using Data Engineering & Machine Learning**.  
It demonstrates an end‑to‑end pipeline combining **data engineering**, **machine learning**, and **deployment** with Streamlit.

---

## 📋 Problem Statement
The goal is to build a Loan Approval Prediction System that:
1. Loads multiple JSON files into a MySQL database.
2. Retrieves and preprocesses the data in Python.
3. Trains a classification model to predict loan approval status.
4. Saves the trained model as a `.pkl` file.
5. Deploys the model as an interactive Streamlit web application.

---

## 📂 Dataset
The dataset is split into three JSON files:
- `applicant_info.json` → Demographic details (age, gender, education)
- `financial_info.json` → Financial details (income, credit history, etc.)
- `loan_info.json` → Loan details (loan amount, term, status)

These are merged in MySQL to form a unified dataset for training.

---

## ⚙️ Requirements
### Python Libraries
- `pandas`, `numpy` → Data handling
- `scikit-learn` → Model training & evaluation
- `pymysql` / `sqlalchemy` → MySQL connection
- `pickle` → Save/load trained models
- `streamlit` → Web app deployment

### Database
- **MySQL** for structured data storage

---

## 🧠 Model
- Classification algorithms: **RandomForest**, Logistic Regression, or Decision Trees
- Preprocessing: Encoding categorical variables, scaling numeric features
- Model saved as `loan_approval_model.pkl`

---

## 🚀 Streamlit App
- Interactive web interface (`streamlit_app.py`)
- User inputs applicant details
- Preprocesses input data consistently with training
- Predicts loan approval status (Approved / Rejected)
- Displays results in a user‑friendly format

---

## ▶️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/loan-approval-prediction.git
   cd loan-approval-prediction
