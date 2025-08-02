
# 🧬 Breast Cancer Detection Project

A complete end-to-end machine learning project that detects whether a tumor is **benign** or **malignant** based on 30 medical features. This project demonstrates data engineering (ETL), data preprocessing, model training with GridSearchCV, API deployment, and an interactive Streamlit frontend.

---


🚀 **Live App**: [Click here to try it on Streamlit](https://talhadeveloperr-breast-cancer-detection-streamlit-app-1ltj8g.streamlit.app/)

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://talhadeveloperr-breast-cancer-detection-streamlit-app-1ltj8g.streamlit.app/)

## 📁 Project Structure

```
breast_cancer_project/
├── etl/
│   ├── extract.py           # Load dataset from sklearn
│   ├── transform.py         # Data cleaning
│   └── load.py              # Combine extract and transform
├── model/
│   ├── model_training.py    # Train and save best ML model
│   ├── breast_cancer_model.pkl
│   └── scaler.pkl
├── api/
│   └── main.py              # FastAPI endpoint for prediction
├── streamlit_app.py         # Streamlit UI for user prediction
├── requirements.txt         # Required dependencies
└── README.md
```

---

## 🚀 Features

* ✅ Data extraction from built-in breast cancer dataset
* ✅ ETL pipeline: Extract → Clean → Load
* ✅ Feature scaling with `StandardScaler`
* ✅ GridSearchCV across multiple models:

  * Logistic Regression
  * Support Vector Machine
  * Random Forest
  * Gradient Boosting
  * XGBoost *(optional)*
* ✅ Evaluation:

  * Accuracy, Precision, Recall, F1-score
  * ROC Curve and AUC score
* ✅ FastAPI backend for model deployment
* ✅ Streamlit frontend for real-time prediction

---

## 📦 Installation

```bash
git clone <repo-url>
cd breast_cancer_project
pip install -r requirements.txt
```

---

## 🧪 Run Pipeline and Model Training

```bash
# Run ETL and model training
python model/model_training.py
```

---

## 🌐 Run FastAPI Backend

```bash
uvicorn api.main:app --reload
```

Open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🎯 Run Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

This launches the UI in your browser, allowing predictions using manual input or auto-filled samples.

---

## 📊 Sample Input (Benign)

```json
[12.45, 15.7, 82.57, 476.5, 0.1278, 0.17, 0.1578, 0.08089, 0.2087, 0.07613,
 0.3345, 1.106, 2.217, 27.19, 0.007514, 0.02052, 0.03131, 0.01262, 0.01992, 0.002294,
 14.5, 23.75, 95.14, 634.2, 0.1659, 0.2868, 0.3198, 0.1196, 0.2841, 0.08225]
```

---

## ✅ Requirements

```
scikit-learn
pandas
numpy
matplotlib
seaborn
xgboost
fastapi
uvicorn
streamlit
```

Install with:

```bash
pip install -r requirements.txt
```
