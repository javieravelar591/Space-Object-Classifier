import numpy as np
import joblib

# Load model and tools
clf = joblib.load("model/random_forest_model.pkl")
sc = joblib.load("model/scaler.pkl")
le = joblib.load("model/label_encoder.pkl")

# Replace with your actual column names in correct order
columns = [
    'u', 'g', 'r', 'i', 'z',
    'ra', 'dec', 'redshift', 'platequality',
    'solarmass', 'star_metallicity', 'velocitydisp'
]

def predict_class(raw_input: dict):
    features = np.array([raw_input[col] for col in columns])

    scaled_features = sc.transform(features)
    pred = clf.predict(scaled_features)
    predicted_label = le.inverse_transform(pred)
    return predicted_label[0]


sample = {
    'u': 18.2,
    'g': 17.5,
    'r': 17.2,
    'i': 17.0,
    'z': 16.9,
    'ra': 150.1,
    'dec': 2.3,
    'redshift': 0.05,
    'platequality': 1,       # Make sure categorical values are numerically encoded
    'solarmass': 10.5,
    'star_metallicity': -0.2,
    'velocitydisp': 220.0
}

print("Predicted class:", predict_class(sample))