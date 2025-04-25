import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Climate Prediction", layout="centered")
st.title("🌍 Climate Change Temperature Prediction")

st.markdown("Enter values for the following 7 features to predict the temperature anomaly:")

# Real feature names from your dataset
co2 = st.number_input("CO2 (ppm)", value=400.0)
ch4 = st.number_input("CH4 (ppb)", value=1800.0)
n2o = st.number_input("N2O (ppb)", value=320.0)
solar = st.number_input("Solar Radiation (W/m²)", value=1361.0)
deforestation = st.number_input("Deforestation Rate (%)", value=1.0)
ocean = st.number_input("Ocean Current Index", value=0.0)
ice = st.number_input("Ice Sheet Mass (Gt)", value=-200.0)

# Prepare input for prediction
features = np.array([[co2, ch4, n2o, solar, deforestation, ocean, ice]])
scaled_features = scaler.transform(features)

# Predict and display result
if st.button("Predict Temperature Anomaly"):
    prediction = model.predict(scaled_features)
    st.success(f"🌡️ Predicted Temperature Anomaly: {prediction[0]:.2f} °C")
