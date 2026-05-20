import joblib
import pandas as pd
import streamlit as st


MODEL_FEATURES = ['bedrooms', 'bathrooms', 'car_spaces', 'land_size_m2', 'building_size_m2', 'schools_nearby_count', 'sold_year', 'sold_month', 'sold_quarter', 'total_rooms', 'suburb', 'property_type', 'suburb_property_type', 'land_size_missing']
BEST_MODEL_NAME = 'Ridge Regression'

st.set_page_config(page_title="Melbourne Housing Price Demo", layout="centered")
st.title("Melbourne Housing Price Predictor")
st.caption("Educational model for Ringwood, Bayswater, and Nunawading sold-property data.")

model = joblib.load("model/best_housing_price_model.joblib")

suburb = st.selectbox("Suburb", ["Ringwood", "Bayswater", "Nunawading"])
property_type = st.selectbox("Property type", ["House", "Unit", "Townhouse", "Apartment"])
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=8, value=3, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2, step=1)
car_spaces = st.number_input("Car spaces", min_value=0, max_value=8, value=2, step=1)
land_size_m2 = st.number_input("Land size (m2)", min_value=0, max_value=2000, value=600, step=10)
building_size_m2 = st.number_input("Building size estimate (m2)", min_value=40, max_value=400, value=160, step=5)
schools_nearby_count = st.slider("Schools nearby count", min_value=3, max_value=10, value=7)
sold_year = st.selectbox("Sale year", [2025, 2026], index=1)
sold_month = st.slider("Sale month", min_value=1, max_value=12, value=5)
sold_quarter = int((sold_month - 1) // 3 + 1)
total_rooms = bedrooms + bathrooms
land_size_missing = land_size_m2 == 0
suburb_property_type = f"{suburb}_{property_type}"

row = pd.DataFrame([{
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "car_spaces": car_spaces,
    "land_size_m2": None if land_size_missing else land_size_m2,
    "building_size_m2": building_size_m2,
    "schools_nearby_count": schools_nearby_count,
    "sold_year": sold_year,
    "sold_month": sold_month,
    "sold_quarter": sold_quarter,
    "total_rooms": total_rooms,
    "suburb": suburb,
    "property_type": property_type,
    "suburb_property_type": suburb_property_type,
    "land_size_missing": land_size_missing,
}])[MODEL_FEATURES]

if st.button("Predict sold price", type="primary"):
    prediction = float(model.predict(row)[0])
    st.metric("Estimated sold price", f"${prediction:,.0f}")
    st.info(
        "The model was trained on a small Ringwood/Bayswater/Nunawading sample and is not a certified valuation.")

with st.expander("Model details"):
    st.write(f"Best saved model: **{BEST_MODEL_NAME}**")
    st.write("The app uses the same preprocessing pipeline as the notebook.")
