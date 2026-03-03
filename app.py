import streamlit as st
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    layout="centered"
)

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style='text-align:center; white-space:nowrap;'>
    🏠 Bangalore House Price Predictor
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("Enter details below to predict house price in Bangalore.")

# ---------------- LOAD MODEL ----------------
try:
    with open("bangalore_home_prices_model (1).pkl", "rb") as f:
        model = pickle.load(f)

    locations = (
        model.named_steps["columntransformer"]
        .transformers_[0][1]
        .categories_[0]
    )

except Exception:
    st.error("❌ Model file not found or failed to load.")
    st.stop()

# ---------------- USER INPUT ----------------
location = st.selectbox(
    "Select Location",
    ["Select Location"] + list(locations)
)

sqft = st.text_input(
    "Total Sqft",
    placeholder="Enter total square feet"
)

bath = st.text_input(
    "Number of Bathrooms",
    placeholder="Enter number of bathrooms"
)

bhk = st.text_input(
    "BHK",
    placeholder="Enter BHK"
)

# ---------------- PREDICTION ----------------
if st.button("Predict Price"):

    if (
        location == "Select Location"
        or sqft.strip() == ""
        or bath.strip() == ""
        or bhk.strip() == ""
    ):
        st.warning("⚠ Please fill all fields correctly.")
    else:
        try:
            input_df = pd.DataFrame(
                [[
                    location,
                    float(sqft),
                    int(bath),
                    int(bhk)
                ]],
                columns=[
                    "location",
                    "total_sqft",
                    "bath",
                    "bhk"
                ]
            )

            price_lakhs = model.predict(input_df)[0]

            # Convert Lakhs → Rupees
            price_rupees = int(price_lakhs * 100000)
            formatted_rupees = format(price_rupees, ",")

            # Lakhs and Crores
            lakhs = round(price_lakhs, 2)
            crores = round(price_lakhs / 100, 2)

            st.success(
                f"""
💰 Estimated Price:

₹ {formatted_rupees}

({lakhs} Lakhs | {crores} Crore)
"""
            )

        except ValueError:
            st.error("❌ Please enter valid numeric values.")

        except Exception:
            st.error("❌ Prediction failed. Check model format.")
