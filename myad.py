import streamlit as st
import joblib
import pandas as pd
from prophet import Prophet

# Load the Prophet model
model = joblib.load('model_forecast.joblib')

# Function to make predictions and display the results
def make_predictions_and_display(user_input_date):
    # Convert the date input to a DataFrame with 'ds' column
    new_data = pd.DataFrame({'ds': pd.to_datetime([user_input_date])})

    # Make prediction for the user-input date
    forecast_new = model.predict(new_data)

    # Display the prediction results
    st.subheader('Prediction for {}'.format(user_input_date))
    st.write(forecast_new[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Streamlit app
st.title('Prophet Forecasting App')

# Load historical data
df = pd.read_csv('USD_PKR Historical Data (1).csv')  # Replace with your actual file
df = df.rename(columns={'Date': 'ds', 'Price': 'y'})

# Date input for prediction
user_input_date = st.date_input('Enter a date for prediction:', pd.to_datetime('2029-02-04'))

# Call the function with the converted date input
make_predictions_and_display(user_input_date)
