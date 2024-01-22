# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 00:06:05 2023

@author: Poorva Khot
"""

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle
import streamlit as st
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow import keras
import math
import tensorflow as tf
import matplotlib.pyplot as plt

st.markdown("""
    <style>
        /* Set background image */
        .stApp {
            background-image: url("https://media.istockphoto.com/id/1436041981/photo/growth-diagram-crude-oil-stock-price-graph-of-energy-market-business-on-gasoline-petroleum.jpg?s=612x612&w=0&k=20&c=KQtz1Ps9QO8_TA1yM4yCeGbEkxyFJCJLNI6D6vcYKBw=");
            background-attachment: fixed;
            background-size: cover;
        }
    </style>""", unsafe_allow_html=True)

model = tf.keras.models.load_model("lstm_model.h5")

# Load the feature and target scalers
scaler_features = joblib.load("feature_scaler.pkl")
scaler_target = joblib.load("target_scaler.pkl")

def main():
    page = st.sidebar.radio("Navigate", ("Introduction","Forecasting", "Visualization", "Next 40 days forecast"))
    
    if page == "Introduction":
        display_introduction_page()
    elif page == "Forecasting":
        forecasting_page()
    elif page == "Visualization":
        if st.session_state.get('Forecasted_close') is not None and not st.session_state['Forecasted_close'].empty:
            display_visualization_page()
        else:
            st.subheader('Please insert a CSV file to see the visualization')
    elif page == "Next 40 days forecast":
        display_next_40days_forecast_page()
        
# def predict(inputs):
#     prediction = forecast_model.predict(inputs)
#     close_prediction = scaler.inverse_transform(prediction)[0][0]
#     return close_prediction
        
def get_file_name():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    file_name = 'Predicted_Data_'+ formatted_datetime + '.csv'
    return file_name

def forecasting_page():
    st.header('Forecasting')
    volume  = st.number_input("Volume",  format="%.2f")
    open_price  = st.number_input("Open Price",  format="%.2f")
    high  = st.number_input("High", format="%.2f")
    low = st.number_input("Low",format="%.2f")
    #user_input = pd.DataFrame({'Volume': [volume], 'Open': [open_price], 'High': [high], 'Low': [low]})
    if st.button('Predict'):
        input_data = np.array([volume, open_price, high, low]).reshape(1, -1)
        # Scale the input data using the feature scaler
        scaled_input_data = scaler_features.transform(input_data)
        # Make a prediction using the trained model
        scaled_prediction = model.predict(scaled_input_data.reshape(1, 4, 1))
        # Inverse transform to get the forecasted Close price
        forecasted_close = scaler_target.inverse_transform(scaled_prediction)[0, 0] 
        st.success(f"Forecasted Close Price: {forecasted_close}") 
    
    st.subheader('Upload CSV')
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read the CSV data using 
        index_names_to_read = ['Volume','Open','High','Low']
        df = pd.read_csv(uploaded_file, usecols= lambda column: column in index_names_to_read)
        predicted_values = []
        for index, row in df.iterrows():
            row_array = row.to_numpy();
            #st.write(row_array)
            input_data = np.array(row_array).reshape(1,-1)
            scaled_input_data = scaler_features.transform(input_data)
            scaled_prediction = model.predict(scaled_input_data.reshape(1, 4, 1))
            # Inverse transform to get the forecasted Close price
            forecasted_close = scaler_target.inverse_transform(scaled_prediction)[0, 0] 
            predicted_values.append(round(forecasted_close, 2))
        # Display the data frame
        df['Forecasted_close'] = predicted_values
        csv_file = df.to_csv(index=False)
        file_name = get_file_name()
        st.session_state['Forecasted_close'] = df;
        st.dataframe(df)
        st.download_button("Download CSV", data=csv_file, file_name = file_name, mime='text/csv')
        
        
def display_introduction_page():
    st.header('Crude Oil Forecasting')
    st.write('This project focuses on forecasting crude oil prices using a Long Short-Term Memory (LSTM) neural network. Crude oil prices have a profound impact on global economies and financial markets. Accurate price predictions are essential for governments, energy companies, and investors.')
    st.write("We'll use daily data from 2018 to 2022, including features like opening prices, trading volumes, and price extremes. Our goal is to predict crude oil closing prices. LSTM networks are chosen for their ability to capture time dependencies.")
    st.write("The project involves data preparation, feature engineering, model building, and evaluation. Additionally, we'll explore directional forecasting to predict 'Up' or 'Down' movements in crude oil prices.")
    st.write("This project aims to provide valuable insights and accurate price forecasts for stakeholders in the energy and financial sectors.")
    
def display_visualization_page():
    st.title("Visualization for all variables")
    data = st.session_state['Forecasted_close']
    open_data = data[['Open']]
    high_data = data[['High']]
    low_data = data[['Low']]
    close_data = data[['Forecasted_close']]
    
    fig, ax = plt.subplots()
   
    ax.plot(open_data, label='Open', marker='o')
    ax.plot(high_data, label='High', marker='o')
    ax.plot(low_data, label='Low', marker='o')
    ax.plot(close_data, label='Forecasted Close', marker='o')

    ax.set_title('Plotting for Open, High, Low and Close')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.legend()

    # Display the plot
    st.pyplot(fig)
    
    volume_data = data[['Volume']]
    fig1, ax1 = plt.subplots()
    ax1.plot(volume_data, label='Volume', marker='o')
    ax1.set_title('Plotting for Volume')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.legend()

    # Display the plot
    st.pyplot(fig1)    

def display_next_40days_forecast_page():
    st.subheader('Visualization for upcoming forecast')
    st.image('image.png', caption='Forecasting of next upcoming 40 days', use_column_width=True)

if __name__=='__main__':
    main()