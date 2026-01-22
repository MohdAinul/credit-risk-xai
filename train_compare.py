import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("data/credit.csv")

X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------- Logistic Regression --------
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000))
])

print("\nTraining Logistic Regression...")
lr_pipeline.fit(X_train, y_train)

y_pred_lr = lr_pipeline.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

joblib.dump(lr_pipeline, "model/logistic_pipeline.joblib")

# -------- EBM --------
print("\nTraining Explainable Boosting Machine (EBM)...")

ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X_train, y_train)

y_pred_ebm = ebm.predict(X_test)

print("\nEBM Results")
print("Accuracy:", accuracy_score(y_test, y_pred_ebm))
print(classification_report(y_test, y_pred_ebm))

joblib.dump(ebm, "model/ebm_model_compare.joblib")