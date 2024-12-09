from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your trained model
model_path = "models/xgboost_model.joblib"
model = joblib.load(model_path)

app = Flask(__name__)

# Function to encode categorical columns
def encode_categorical_columns(df: pd.DataFrame, encoding_type: str = 'label') -> pd.DataFrame:
    df_encoded = df.copy()
    categorical_columns = df_encoded.select_dtypes(include=['category', 'object']).columns

    if encoding_type == 'label':
        for col in categorical_columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])

    elif encoding_type == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, drop_first=True)

    else:
        raise ValueError("Unsupported encoding_type. Use 'label' or 'onehot'.")

    return df_encoded

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data and cast to appropriate types
        data = request.form
        features = pd.DataFrame({
            'airline': [int(data['airline'])],
            'airline_dot': [int(data['airline_dot'])],
            'airline_code': [int(data['airline_code'])],
            'dot_code': [int(data['dot_code'])],
            'fl_number': [int(data['fl_number'])],
            'origin': [int(data['origin'])],
            'origin_city': [int(data['origin_city'])],
            'dest': [int(data['dest'])],
            'dest_city': [int(data['dest_city'])],
            'crs_dep_time': [int(data['crs_dep_time'])],
            'dep_time': [int(data['dep_time'])],
            'dep_delay': [float(data['dep_delay'])],
            'taxi_out': [float(data['taxi_out'])],
            'wheels_off': [int(data['wheels_off'])],
            'wheels_on': [int(data['wheels_on'])],
            'taxi_in': [float(data['taxi_in'])],
            'crs_arr_time': [int(data['crs_arr_time'])],
            'arr_time': [int(data['arr_time'])],
            'arr_delay': [float(data['arr_delay'])],
            'cancelled': [bool(int(data['cancelled']))],
            'diverted': [bool(int(data['diverted']))],
            'crs_elapsed_time': [float(data['crs_elapsed_time'])],
            'elapsed_time': [float(data['elapsed_time'])],
            'air_time': [float(data['air_time'])],
            'distance': [float(data['distance'])],
            'delay_due_carrier': [float(data['delay_due_carrier'])],
            'delay_due_weather': [float(data['delay_due_weather'])],
            'delay_due_nas': [float(data['delay_due_nas'])],
            'delay_due_security': [float(data['delay_due_security'])],
            'delay_due_late_aircraft': [float(data['delay_due_late_aircraft'])],
            'day': [int(data['day'])],
            'month': [int(data['month'])],
            'day_of_week': [int(data['day_of_week'])],
            'year': [int(data['year'])]
        })

        # Convert the input data to a DataFrame for consistency with model input expectations
        input_df = pd.DataFrame([data])

        # Encode the input data
        encoded_input_df = encode_categorical_columns(input_df, encoding_type='label')

        # Predict the class using the XGBoost model
        prediction = model.predict(encoded_input_df)
        result = "Delayed" if prediction[0] else "On Time"

        return render_template('result.html', result=result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)