import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import numpy as np

# Load dataset
@st.cache
def load_data():
    data = pd.read_csv("crop_data2.csv")
    return data

# Preprocess dataset
@st.cache
def preprocess_data(data):
    # Manually label encode suitability based on features
    data['suitability'] = data.apply(lambda row: classify_suitability(row), axis=1)
    
    # Encode categorical suitability column
    le = LabelEncoder()
    data['suitability'] = le.fit_transform(data['suitability'])
    return data, le

# Function to classify suitability based on feature thresholds
def classify_suitability(row):
    if row['temperature'] > 25 and row['rainfall'] > 150 and row['soil_ph'] >= 6.0 and row['market_demand'] >= 7:
        return 'Highly Suitable'
    elif 15 <= row['temperature'] <= 25 and 100 <= row['rainfall'] <= 200:
        return 'Moderately Suitable'
    else:
        return 'Not Suitable'

# Train ML model
@st.cache
def train_model(data):
    X = data[['temperature', 'rainfall', 'soil_ph', 'market_demand']]
    y = data['suitability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Streamlit app
st.title("AI-Based Crop Recommendation System 🌾")
st.write("Predict crop suitability based on environmental conditions and market demand.")

# Load and preprocess data
data = load_data()
processed_data, label_encoder = preprocess_data(data)

# Train model
model, X_test, y_test = train_model(processed_data)

# User inputs
st.subheader("Input Conditions:")

# Crop name selection
crop_name = st.selectbox("Choose a crop (Optional):", ["None"] + list(data['crop_name'].unique()))

# Condition sliders
temperature = st.slider("Temperature (°C)", min_value=0, max_value=50, value=25)
rainfall = st.slider("Rainfall (mm)", min_value=0, max_value=500, value=150)
soil_ph = st.slider("Soil pH", min_value=4.0, max_value=9.0, value=6.5, step=0.1)
market_demand = st.slider("Market Demand (1-10)", min_value=1, max_value=10, value=7)

# Handle crop-specific data if a crop is selected
if crop_name != "None":
    crop_data = data[data['crop_name'] == crop_name].iloc[0]
    st.write(f"Conditions for **{crop_name}** from the dataset:")
    st.write(f"- Temperature: {crop_data['temperature']}°C")
    st.write(f"- Rainfall: {crop_data['rainfall']} mm")
    st.write(f"- Soil pH: {crop_data['soil_ph']}")
    st.write(f"- Market Demand: {crop_data['market_demand']}")

    # Override sliders with crop-specific data
    temperature = crop_data['temperature']
    rainfall = crop_data['rainfall']
    soil_ph = crop_data['soil_ph']
    market_demand = crop_data['market_demand']

# Predict crop suitability
if st.button("Recommend Crops"):
    input_data = np.array([[temperature, rainfall, soil_ph, market_demand]])
    predictions = model.predict(input_data)
    predicted_label = label_encoder.inverse_transform(predictions)
    st.write(f"The predicted suitability is: **{predicted_label[0]}**")

# Evaluate model accuracy
from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

accuracy = evaluate_model(model, X_test, y_test)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display dataset
if st.checkbox("Show Dataset"):
    st.write(data)
