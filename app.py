import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Bangalore House Price Predictor", layout="centered")

st.title("🏠 Bangalore House Price Predictor")
st.write("Enter details below to predict house price in Bangalore.")

# Load Model Safely
try:
    with open('bangalore_home_prices_model (1).pkl', 'rb') as f:
        model = pickle.load(f)

    # Extract locations safely
    locations = model.named_steps['columntransformer'] \
        .transformers_[0][1] \
        .categories_[0]

except Exception as e:
    st.error("❌ Model file not found or failed to load.")
    st.stop()

# User Inputs
location = st.selectbox("Select Location", locations)
sqft = st.number_input("Total Sqft", min_value=300)
bath = st.number_input("Number of Bathrooms", min_value=1, step=1)
bhk = st.number_input("BHK", min_value=1, step=1)

# Prediction
if st.button("Predict Price"):
    input_df = pd.DataFrame(
        [[location, sqft, bath, bhk]],
        columns=['location', 'total_sqft', 'bath', 'bhk']
    )

    try:
        price = model.predict(input_df)[0]
        st.success(f"💰 Estimated Price: ₹ {round(price,2)} Lakhs")
    except:
        st.error("Prediction failed. Check model format.")
