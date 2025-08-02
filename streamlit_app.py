import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model and scaler
model = pickle.load(open('model/breast_cancer_model.pkl', 'rb'))
scaler = model.named_steps['scaler']  # from Pipeline

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.title("🧬 Breast Cancer Detection")
st.markdown("Provide the following 30 features to predict whether the tumor is benign or malignant.")

# Sample data options
sample_options = {
    "Benign Example": [12.45, 15.7, 82.57, 476.5, 0.1278, 0.17, 0.1578, 0.08089, 0.2087, 0.07613,
                        0.3345, 1.106, 2.217, 27.19, 0.007514, 0.02052, 0.03131, 0.01262, 0.01992, 0.002294,
                        14.5, 23.75, 95.14, 634.2, 0.1659, 0.2868, 0.3198, 0.1196, 0.2841, 0.08225],

    "Malignant Example": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                           1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                           25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
}

selected_sample = st.selectbox("⬇️ Load Sample Data (Optional):", ["None"] + list(sample_options.keys()))

if selected_sample != "None":
    st.success(f"Loaded sample: {selected_sample}")
    input_data = sample_options[selected_sample].copy()
else:
    input_data = []

# Feature names
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Collect user input
cols = st.columns(3)
manual_input_data = []
for i, name in enumerate(feature_names):
    default_val = input_data[i] if len(input_data) == 30 else 0.0
    value = cols[i % 3].number_input(f"{name}", step=0.01, value=default_val)
    manual_input_data.append(value)

# Predict button
if st.button("🔍 Predict"):
    arr = np.array(manual_input_data).reshape(1, -1)
    prediction = model.predict(arr)[0]
    confidence = model.predict_proba(arr)[0][prediction]
    label = "Benign" if prediction == 1 else "Malignant"
    color = "green" if label == "Benign" else "red"

    st.markdown(f"### 🧾 Result: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
    st.markdown(f"**Confidence:** {confidence:.2%}")
