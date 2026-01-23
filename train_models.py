import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1) Load data
csv_path = "data/credit.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found.")

df = pd.read_csv(csv_path)
print("Loaded data:", df.shape)

# 2) Prepare X,y
if "default" not in df.columns:
    raise ValueError("Expected target column named 'default' in data/credit.csv")
X = df.drop(columns=["default"])
y = df["default"].astype(int)

# 3) Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4) Preprocessing
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Numeric columns:", numeric_cols)

# We use the same preprocessor structure for all models for consistency
preprocessor = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="median"), numeric_cols)
], remainder='drop')

# 5) Define Pipelines

# Logistic Regression
lr_pipeline = Pipeline(steps=[
    ("pre", preprocessor),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

# Random Forest
rf_pipeline = Pipeline(steps=[
    ("pre", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

# XGBoost
xgb_pipeline = Pipeline(steps=[
    ("pre", preprocessor),
    ("model", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

models = {
    "Logistic Regression": lr_pipeline,
    "Random Forest": rf_pipeline,
    "XGBoost": xgb_pipeline
}

# 6) Train and Save
os.makedirs("model", exist_ok=True)

for name, pipeline in models.items():
    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {round(acc, 3)}")

    # Generate filename from name
    filename = name.lower().replace(" ", "_")
    if name == "Logistic Regression":
        filename = "lr_pipeline"
    elif name == "Random Forest":
        filename = "rf_pipeline"
    elif name == "XGBoost":
        filename = "xgb_pipeline"

    save_path = f"model/{filename}.joblib"
    joblib.dump(pipeline, save_path)
    print(f"Saved {name} to {save_path}")

print("\nAll models trained and saved.")
