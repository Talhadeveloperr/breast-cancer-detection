# etl/extract.py
from sklearn.datasets import load_breast_cancer
import pandas as pd

def extract_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df
