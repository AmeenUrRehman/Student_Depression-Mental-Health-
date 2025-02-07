import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    """Load the trained mental health prediction model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def preprocess_data(new_data):
    """Preprocess the input data using StandardScaler."""
    scaler = StandardScaler()
    return scaler.fit_transform(new_data)

def predict_mental_health(model, new_data):
    """Make predictions using the loaded model."""
    new_data_scaled = preprocess_data(new_data)
    return model.predict(new_data_scaled)

if __name__ == "__main__":
    # Load the model
    model = load_model('student_depression_model.pkl')

    # Updated DataFrame with explicit index
    new_data = pd.DataFrame({
        'Degree': [0],
        'Age': [23],
        'Academic Pressure': [6],
        'cgpa': [3.0],
        'Have you ever had suicidal thoughts?': [0],
    })

    # Predict and print result
    prediction = predict_mental_health(model, new_data)
    print("Predicted mental health condition:", prediction)