import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from etl.load import load_data

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# ---------------------- Load Cleaned Data ----------------------
df = load_data()
X = df.drop('target', axis=1)
y = df['target'].map({'malignant': 0, 'benign': 1})

# ---------------------- Train/Test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------- Model Dictionary ----------------------
models = {
    'LogisticRegression': (
        LogisticRegression(max_iter=1000),
        {'clf__C': [0.1, 1, 10]}
    ),
    'SVC': (
        SVC(probability=True),
        {'clf__C': [0.1, 1, 10], 'clf__kernel': ['linear', 'rbf']}
    ),
    'RandomForest': (
        RandomForestClassifier(),
        {'clf__n_estimators': [50, 100, 200]}
    ),
    'GradientBoosting': (
        GradientBoostingClassifier(),
        {'clf__n_estimators': [50, 100], 'clf__learning_rate': [0.01, 0.1]}
    )
}

if xgb_available:
    models['XGBoost'] = (
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        {'clf__n_estimators': [50, 100], 'clf__learning_rate': [0.01, 0.1]}
    )

# ---------------------- GridSearch with Pipeline ----------------------
best_model = None
best_score = 0
best_name = ''

print("\n--- Model Training with GridSearchCV ---\n")
for name, (clf, params) in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])

    grid = GridSearchCV(pipe, param_grid=params, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    print(f"{name} best score: {grid.best_score_:.4f} | Best params: {grid.best_params_}")

    if grid.best_score_ > best_score:
        best_model = grid.best_estimator_
        best_score = grid.best_score_
        best_name = name

# ---------------------- Final Evaluation ----------------------
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print(f"\n--- Final Evaluation for Best Model: {best_name} ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# ---------------------- Plot ROC Curve ----------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_score(y_test, y_prob):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_name}')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------- Save Model ----------------------
pickle.dump(best_model, open('model/breast_cancer_model.pkl', 'wb'))
print("\n✅ Best model saved successfully.")
