from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Union
from utils import encode_categorical_columns

# Load your trained models
model_path = "models/random_forest_trimmed_data_model.joblib"
try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {model_path}: {e}")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data and cast to appropriate types
        data = request.form

        # Define the correct feature order based on the model
        feature_order = [
            'airline', 'fl_number', 'origin_city', 'dest_city',
            'crs_dep_time', 'dep_time', 'dep_delay', 'taxi_out',
            'taxi_in', 'crs_arr_time', 'arr_time', 'arr_delay',
            'distance', 'day', 'month', 'day_of_week'
        ]

        # Create the DataFrame in the correct order
        input_df = pd.DataFrame([[data[field] for field in feature_order]], columns=feature_order)

        # Encode the input data
        encoded_input_df = encode_categorical_columns(input_df, encoding_type='label')

        # Predict the class using the XGBoost model
        prediction = model.predict(encoded_input_df)
        result = "On Time" if prediction[0] == 0 else "Delayed"

        return render_template('result.html', result=result)

    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/predict_api', methods=['POST'])
def predict_api() -> Union[str, jsonify]:
    try:
        # Extract JSON data from the request body
        data = request.get_json()
    
        # Convert the input data to a DataFrame for consistency with model input expectations
        input_df = pd.DataFrame([data])
    
        # Encode the input data
        encoded_input_df = encode_categorical_columns(input_df, encoding_type='label')
    
        # Predict the class using the model
        prediction = model.predict(encoded_input_df)
    
        return jsonify({'result': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':  
    app.run(debug=True)