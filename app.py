import streamlit as st
import pandas as pd
import joblib

# Load model
pipeline = joblib.load("delay_model.pkl")
y_pred = pipeline.predict(X_test)

st.title("🚆 Mumbai Local Train Delay Analytics")

# User inputs
distance = st.number_input("Distance (km)", min_value=1.0, step=1.0)
speed = st.number_input("Speed (kmph)", min_value=1.0, step=1.0)
passengers = st.number_input("Passengers (daily)", min_value=1000, step=1000)
station = st.text_input("Station Name")
line = st.text_input("Line Name")

if st.button("Predict Delay (minutes)"):
    input_data = pd.DataFrame([[distance, speed, passengers, station, line]],
                              columns=["Distance_km", "Speed_kmph", "Passengers_daily", "Station", "Line"])
    prediction = model.predict(input_data)[0]
    st.success(f"⏱ Predicted Delay: {prediction:.2f} minutes")

