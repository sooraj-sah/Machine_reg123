import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Price Predictor ML", layout="centered")

st.title("Automated Cost & Valuation Predictor")
st.write("Enter details below to generate an instant estimate using ML Pipeline.")

# Model Load
@st.cache_resource
def load_model():
    return joblib.load('best_regression_pipeline.pkl')

pipeline = load_model()

# Form Inputs
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 65, 30)
    bmi = st.number_input("BMI Index", 15.0, 50.0, 24.5)
    children = st.selectbox("Children / Dependents", [0, 1, 2, 3, 4, 5])
with col2:
    sex = st.selectbox("Sex", ['male', 'female'])
    smoker = st.selectbox("Smoker?", ['no', 'yes'])
    region = st.selectbox("Region", ['southwest', 'southeast', 'northwest', 'northeast'])

if st.button("Calculate Prediction", type="primary"):
    input_data = pd.DataFrame([{
        'age': age, 'bmi': bmi, 'children': children,
        'sex': sex, 'smoker': smoker, 'region': region
    }])
    pred = pipeline.predict(input_data)[0]
    st.success(f"Estimated Cost: **${pred:,.2f}**")