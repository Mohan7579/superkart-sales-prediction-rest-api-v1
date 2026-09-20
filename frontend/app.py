import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("SuperKart Prediction")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for product
Product_Weight = st.number_input("Weight", min_value=1, step=1, value=1)
Product_Sugar_Content = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Allocated area", min_value=0.01, value=0.1)
Product_MRP = st.number_input("MRP", min_value=1, step=1, value=1)
Store_Size = st.selectbox("Store size", ["High", "Medium", "Small"])
Store_Location_City_Type = st.selectbox("Store location city type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
Store_Age = st.number_input("Store age", min_value=1, step=1, value=1)
Product_Type_Category = st.selectbox("Product type category", ["Perishables", "Non Perishables"])

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    'Product_Weight': Product_Weight,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_MRP': Product_MRP,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type,
    'Store_Age': Store_Age,
    'Product_Type_Category': Product_Type_Category
}])

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/predict", json=input_data.to_dict(orient='records')[0])  # Send data to Flask API
    if response.status_code == 200:
        prediction = response.json()['Predicted Sales (in dollars)']
        st.success(f"Predicted Sales (in dollars): {prediction}")
    else:
        st.error("Unable to connect to the prediction API.")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": uploaded_file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")
