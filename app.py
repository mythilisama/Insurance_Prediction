import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("artifacts/model.pkl","rb"))
scaler = pickle.load(open("artifacts/scaler.pkl","rb"))

st.title("Insurance Premium Prediction")

age = st.number_input("Age",18,100)
income = st.number_input("Annual Income (LPA)",1.0,50.0)
policy_term = st.number_input("Policy Term",5,40)
sum_assured = st.number_input("Sum Assured",10,500)

if st.button("Predict"):
    data = pd.DataFrame({
        "Age":[age],
        "Annual_Income_LPA":[income],
        "Policy_Term_Years":[policy_term],
        "Sum_Assured_Lakhs":[sum_assured]
    })

    scaled = scaler.transform(data)
    prediction = model.predict(scaled)

    st.success(f"Predicted Premium: {prediction[0]:.2f} Thousand")