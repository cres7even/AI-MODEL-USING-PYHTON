import streamlit as st
import joblib
import numpy as np

# Load saved model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")  # Only if you used scaling

st.title("My First AI Model Website")
st.subheader("Enter input to get prediction")

# Replace with your actual features
feature1 = st.number_input("Enter Feature 1")
feature2 = st.number_input("Enter Feature 2")
feature3 = st.number_input("Enter Feature 3")

if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Prediction: {prediction[0]}")
