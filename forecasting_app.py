import streamlit as st
import joblib
import numpy as np

# Load the Linear Regression model
model = joblib.load('linear_regression_model.pkl')

# Set the number of time steps (should match the value used during training)
n_steps = 30

# Create a Streamlit web app
st.title("Stock Price Forecasting App")

# Provide user input for forecasting
st.header("Enter Data for Forecasting")

# Input fields for Volume, Open, High, and Low (for 30 days)
volume = st.text_area("Volume (Enter 30 values, separated by commas):")
open_price = st.text_area("Open Price (Enter 30 values, separated by commas):")
high_price = st.text_area("High Price (Enter 30 values, separated by commas):")
low_price = st.text_area("Low Price (Enter 30 values, separated by commas):")

# Helper function to convert input strings into lists
def convert_input_string(input_string):
    return [float(x.strip()) for x in input_string.split(',')]

# Make prediction for 30 days
if st.button("Predict 30 Days"):
    # Convert the input strings into lists
    volume_data = convert_input_string(volume)
    open_data = convert_input_string(open_price)
    high_data = convert_input_string(high_price)
    low_data = convert_input_string(low_price)
    
    # Combine the data for the 4 features
    data = np.array([volume_data, open_data, high_data, low_data]).T  # Transpose to get the right shape
    data = np.reshape(data, (1, n_steps, 4))  # Reshape for prediction
    predictions = model.predict(data)
    
    st.subheader("30-Day Forecast:")
    st.write("Predicted Close Prices for the Next 30 Days:")
    st.write(predictions.flatten())
