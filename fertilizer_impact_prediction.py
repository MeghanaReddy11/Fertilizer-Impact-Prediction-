# Fertilizer Impact Prediction Project
# Author: Naman
# Description: This script analyzes and predicts the impact of different fertilizers on crop growth using machine learning.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,

# Load the dataset
def load_data(filepath):
    """Load crop and fertilizer data from a CSV file."""
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        print("Error: File not found.")
        return None

# Preprocess the data
def preprocess_data(data):
    """Handle missing values and encode categorical variables."""
    data = data.dropna()
    data = pd.get_dummies(data, drop_first=True)  # Encode categorical variables
    return data

# Train the machine learning model
def train_model(data):
    """Train a Random Forest model to predict crop yield based on fertilizer usage."""
    X = data.drop(columns=['Crop_Yield'])  # Features
    y = data['Crop_Yield']  # Target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Evaluation:")
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")
    
    return model

# Predict the impact of fertilizers
def predict_impact(model, new_data):
    """Predict crop yield for new fertilizer combinations."""
    predictions = model.predict(new_data)
    return predictions

# Main function
def main():
    filepath = 'fertilizer_data.csv'  # Replace with your dataset path
    data = load_data(filepath)

    if data is not None:
        print("Data Loaded Successfully.")
        
        data = preprocess_data(data)
        print("Data Preprocessed Successfully.")

        model = train_model(data)
        print("Model Trained Successfully.")

        # Example input for prediction
        new_data = pd.DataFrame({
            'Fertilizer_A': [50],
            'Fertilizer_B': [30],
            'Fertilizer_C': [20]
        })

        predictions = predict_impact(model, new_data)
        print("Predicted Impact on Crop Yield:", predictions)

if __name__ == "__main__":
    main()
