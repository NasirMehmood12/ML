{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "494bf12c-c002-4cef-b635-0f40d100e696",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "import joblib\n",
    "\n",
    "# Load saved model and preprocessing objects\n",
    "model = joblib.load('house_price_model.pkl')\n",
    "scaler = joblib.load('scaler.pkl')\n",
    "label_encoders = joblib.load('label_encoders.pkl')\n",
    "\n",
    "# App Title\n",
    "st.title(\"House Price Prediction App\")\n",
    "\n",
    "# User Inputs\n",
    "st.sidebar.header(\"Enter House Details:\")\n",
    "area = st.sidebar.number_input(\"Area (in square feet):\", min_value=0, step=1)\n",
    "bedrooms = st.sidebar.number_input(\"Number of Bedrooms:\", min_value=0, step=1)\n",
    "bathrooms = st.sidebar.number_input(\"Number of Bathrooms:\", min_value=0, step=1)\n",
    "floors = st.sidebar.number_input(\"Number of Floors:\", min_value=0, step=1)\n",
    "year_built = st.sidebar.number_input(\"Year Built:\", min_value=1800, max_value=2024, step=1)\n",
    "location = st.sidebar.selectbox(\"Location:\", ['Urban', 'Suburban', 'Rural'])\n",
    "condition = st.sidebar.selectbox(\"Condition:\", ['Good', 'Bad', 'Average'])\n",
    "garage = st.sidebar.selectbox(\"Garage:\", ['Yes', 'No'])\n",
    "\n",
    "# When Predict Button is Clicked\n",
    "if st.sidebar.button(\"Predict Price\"):\n",
    "    # Prepare input dictionary\n",
    "    input_data = {\n",
    "        'Area': area,\n",
    "        'Bedrooms': bedrooms,\n",
    "        'Bathrooms': bathrooms,\n",
    "        'Floors': floors,\n",
    "        'YearBuilt': year_built,\n",
    "        'Location': location,\n",
    "        'Condition': condition,\n",
    "        'Garage': garage\n",
    "    }\n",
    "\n",
    "    # Encode categorical features\n",
    "    for col in label_encoders:\n",
    "        if col in input_data:\n",
    "            input_data[col] = label_encoders[col].transform([input_data[col]])[0]\n",
    "\n",
    "    # Convert to DataFrame and scale numerical features\n",
    "    input_df = pd.DataFrame([input_data])\n",
    "    input_scaled = scaler.transform(input_df)\n",
    "\n",
    "    # Predict using the model\n",
    "    prediction = model.predict(input_scaled)\n",
    "\n",
    "    # Display Prediction\n",
    "    st.subheader(f\"Predicted Price for the House: ${prediction[0]:,.2f}\")\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
