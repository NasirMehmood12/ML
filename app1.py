import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load saved model and preprocessing objects
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# App Title
st.title("House Price Prediction App")

# User Inputs
st.sidebar.header("Enter House Details:")
area = st.sidebar.number_input("Area (in square feet):", min_value=0, step=1)
bedrooms = st.sidebar.number_input("Number of Bedrooms:", min_value=0, step=1)
bathrooms = st.sidebar.number_input("Number of Bathrooms:", min_value=0, step=1)
floors = st.sidebar.number_input("Number of Floors:", min_value=0, step=1)
year_built = st.sidebar.number_input("Year Built:", min_value=1800, max_value=2024, step=1)
location = st.sidebar.selectbox("Location:", ['Urban', 'Suburban', 'Rural'])
condition = st.sidebar.selectbox("Condition:", ['Good', 'Bad', 'Average'])
garage = st.sidebar.selectbox("Garage:", ['Yes', 'No'])

# When Predict Button is Clicked
if st.sidebar.button("Predict Price"):
    # Prepare input dictionary
    input_data = {
        'Area': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Floors': floors,
        'YearBuilt': year_built,
        'Location': location,
        'Condition': condition,
        'Garage': garage
    }

    # Encode categorical features
    for col in label_encoders:
        if col in input_data:
            input_data[col] = label_encoders[col].transform([input_data[col]])[0]

    # Convert to DataFrame and scale numerical features
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Predict using the model
    prediction = model.predict(input_scaled)

    # Display Prediction
    st.subheader(f"Predicted Price for the House: ${prediction[0]:,.2f}")

