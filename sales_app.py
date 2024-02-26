import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model
@st.cache_data
def load_model():
    return joblib.load('rf_regressor.joblib')

model = load_model()

# Set background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFB6C1; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Welcome message
st.title('Welcome to the Sales Prediction Platform!')
st.write("This platform is designed to help you predict sales based on your advertising expenditure.")

# Introduction to advertising channels and costs
st.header('Advertising Channels')
st.write("Before we make predictions, let's introduce you to the advertising channels ")
st.write("- TV Advertisment")
st.write("- Radio Advertisment")
st.write("- Newspaper Advertisment")
st.write("Please input the values for advertising expenditure.")

# Input fields for user input
st.sidebar.header('Advertising Channels Parameters')
tv = st.sidebar.text_input('TV Advertising', '150.0')
radio = st.sidebar.text_input('Radio Advertising', '25.0')
newspaper = st.sidebar.text_input('Newspaper Advertising', '50.0')

# Submit button
if st.sidebar.button('Submit'):
    # Convert input values to float
    tv = float(tv)
    radio = float(radio)
    newspaper = float(newspaper)
    
    # Make predictions
    input_data = pd.DataFrame({'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})
    prediction = model.predict(input_data)

    # Display prediction
    st.header('Sales Prediction')
    st.success(f"Predicted Sales: {prediction[0]}")
