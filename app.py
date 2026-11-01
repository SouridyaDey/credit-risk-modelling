import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

load_dotenv()

# Database Initialization
def init_db():
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id SERIAL PRIMARY KEY,
                timestamp_ist TIMESTAMP,
                income REAL,
                schufa REAL,
                term_length REAL,
                install_to_inc REAL,
                num_applic REAL,
                loan_amount REAL,
                marital VARCHAR(50),
                occup VARCHAR(50),
                prediction INTEGER,
                probability REAL
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database initialization failed: {e}")

# Run once on startup
init_db()

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('credit_risk_model.pkl')

model = load_model()

st.title("Credit Decisioning App for Mortgage Loans")
st.write("Enter the applicant's details below to predict their default risk.")

with st.form("applicant_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input("Income", min_value=0, value=None)
        schufa = st.number_input("Schufa Credit Score", min_value=6000, max_value=15000, value=None)
        term_length = st.number_input("Term Length (Months)", min_value=1, value=None)
        install_to_inc = st.number_input("Installment-to-Income Ratio", min_value=0.0001, value=None)
        num_applic = st.number_input("Number of borrowers/applicants tied to the Mortgage", min_value=1, value=None)
        loan_amount = st.number_input("Mortgage Loan amount applied (in thousands USD)", min_value=1000, value=None)
        
    with col2:
        marital = st.selectbox("Marital Status", ['Single', 'Separated', 'Married', 'Living together', 'Divorced'], index=None)
        occup = st.selectbox("Occupation", ['Worker', 'Employee', 'Student', 'Others'], index=None)

    submit_button = st.form_submit_button(label="Predict Risk and Probability of Default (PD)")

if submit_button:
    user_inputs = {
        'income': [income],
        'schufa': [schufa],
        'term_length': [term_length],
        'install_to_inc': [install_to_inc],
        'num_applic': [num_applic],
        'loan_amount': [loan_amount],
        'marital': [marital],
        'occup': [occup]
    }
    
    user_data = pd.DataFrame(user_inputs)
    user_data = user_data[model.feature_names_in_]
    user_data.fillna(value=np.nan, inplace=True)
    
    # Calculate probability 
    probability = round(float(model.predict_proba(user_data)[0][1]), 2)

    # Reject the top 30% riskiest applicants and Decile 3 Bottom Boundary: 0.4765
    custom_threshold = 0.48
    final_prediction = 1 if probability > custom_threshold else 0
    
    # Save to AWS PostgreSQL Database
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        cursor = conn.cursor()
        current_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
        
        # PostgreSQL uses %s placeholders instead of SQLite's ?
        cursor.execute('''
            INSERT INTO prediction_logs (
                timestamp_ist, income, schufa, term_length, install_to_inc, 
                num_applic, loan_amount, marital, occup, prediction, probability
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            current_time, income, schufa, term_length, install_to_inc, 
            num_applic, loan_amount, marital, occup, final_prediction, probability
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Failed to log prediction to AWS database: {e}")

    # Display results
    st.divider()
    if final_prediction == 1:
        st.error(f"High Risk: The applicant is likely to default. (PD: {probability:.1%})")
    else:
        st.success(f"Low Risk: The applicant is likely to repay. (PD: {probability:.1%})")