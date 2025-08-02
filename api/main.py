from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Load saved model and scaler
model = pickle.load(open('model/breast_cancer_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

class CancerInput(BaseModel):
    features: list[float]  # List of 30 float values

@app.post('/predict')
def predict(data: CancerInput):
    input_data = np.array(data.features).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return {"prediction": "benign" if prediction == 1 else "malignant"}
