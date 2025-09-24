
# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load('model.joblib')

# Set the title of the web app
st.title('🏡 California House Price Predictor')

st.write("This app predicts the median house value for districts in California.")
st.write("Use the sliders to set the features of the house.")

# Create sliders for user input
med_inc = st.slider('Median Income (in tens of thousands of USD)', 0.5, 15.0, 3.8)
house_age = st.slider('House Age (in years)', 1.0, 52.0, 28.0)
ave_rooms = st.slider('Average Number of Rooms', 1.0, 10.0, 5.4)
ave_bedrms = st.slider('Average Number of Bedrooms', 0.5, 5.0, 1.1)
population = st.slider('District Population', 200.0, 20000.0, 1425.0)

# 'Predict' button
if st.button('Predict Price'):
    # Prepare the input data for the model
    features = [[med_inc, house_age, ave_rooms, ave_bedrms, population]]
    feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population']
    input_df = pd.DataFrame(features, columns=feature_names)

    # Make a prediction
    prediction_raw = model.predict(input_df)
    prediction_usd = prediction_raw[0] * 100000  # Target value is in $100,000s

    # Display the prediction
    st.success(f'Predicted Median House Value: ${prediction_usd:,.2f}')