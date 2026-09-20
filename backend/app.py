# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
sales_predictor_api = Flask("SuperKart Sales Predictor")

# Load the trained machine learning model
import os
import joblib


# MODEL_PATH = os.path.join(
#    os.path.dirname(__file__),
#    "superkart_model.joblib"
#)

MODEL_PATH = os.path.join(
    "/content/backend_files",
    "superkart_model.joblib"
)

model = joblib.load(MODEL_PATH)

# Define a route for the home page (GET request)
@sales_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Predictor API!"

# Define an endpoint for single property prediction (POST request)
@sales_predictor_api.post('/v1/predict')
def predict_sales():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing property details and returns
    the predicted rental price as a JSON response.
    """
    # Get the JSON data from the request body
    product_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': product_data['Product_Weight'],
        'Product_Sugar_Content': product_data['Product_Sugar_Content'],
        'Product_Allocated_Area': product_data['Product_Allocated_Area'],
        'Product_MRP': product_data['Product_MRP'],
        'Store_Size': product_data['Store_Size'],
        'Store_Location_City_Type': product_data['Store_Location_City_Type'],
        'Store_Type': product_data['Store_Type'],
        'Store_Age': product_data['Store_Age'],
        'Product_Type_Category': product_data['Product_Type_Category']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction
    predicted_sales = model.predict(input_data)[0]

    # Convert predicted_price to Python float
    predicted_sales = round(float(predicted_sales), 2)

    # Return the actual price
    return jsonify({'Predicted Price (in dollars)': predicted_sales})


# Define an endpoint for batch prediction (POST request)
@sales_predictor_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    Handles POST requests to '/v1/predictbatch'.
    Expects a CSV file containing batch prediction data.
    """
    file = request.files['file']

    input_data = pd.read_csv(file)

    # Match batch column name to model feature name
    input_data = input_data.rename(
        columns={'Store_Age_Years': 'Store_Age'}
    )

    batch_features = input_data[
        [
            'Product_Weight',
            'Product_Sugar_Content',
            'Product_Allocated_Area',
            'Product_MRP',
            'Store_Size',
            'Store_Location_City_Type',
            'Store_Type',
            'Product_Type_Category',
            'Store_Age'
        ]
    ]

    predicted_sales = model.predict(batch_features).tolist()

    product_ids = input_data['Product_Id_char'].tolist()

    output_dict = dict(zip(product_ids, predicted_sales))

    return jsonify(output_dict)

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    sales_predictor_api.run(debug=True, use_reloader=False)
