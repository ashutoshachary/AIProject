# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv('heart-disease.csv')

# Split data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler for later use
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Load the saved model and scaler
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = [float(x) for x in request.form.values()]
    
    # Convert data to numpy array
    input_data = np.array(data).reshape(1, -1)
    
    # Scale the input data
    scaled_data = loaded_scaler.transform(input_data)
    
    # Make prediction
    prediction = loaded_model.predict(scaled_data)
    output = "Heart Disease Present" if prediction[0] == 1 else "No Heart Disease"
    
    return render_template('index.html', prediction_text=f'Prediction: {output}')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Get JSON data from API request
    data = request.get_json(force=True)
    
    # Convert to numpy array and reshape
    input_data = np.array(list(data.values())).reshape(1, -1)
    
    # Scale the input data
    scaled_data = loaded_scaler.transform(input_data)
    
    # Make prediction
    prediction = loaded_model.predict(scaled_data)
    output = int(prediction[0])
    
    return jsonify({'prediction': output})


if __name__ == '__main__':
    app.run(debug=True)
