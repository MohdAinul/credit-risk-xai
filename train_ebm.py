# train_ebm.py
import pandas as pd
import joblib, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from interpret.glassbox import ExplainableBoostingClassifier

# 1) Load data
csv_path = "data/credit.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found. Run create_synthetic.py")

df = pd.read_csv(csv_path)
print("Loaded data:", df.shape)

# 2) Prepare X,y
if "default" not in df.columns:
    raise ValueError("Expected target column named 'default' in data/credit.csv")
X = df.drop(columns=["default"])
y = df["default"].astype(int)

# 3) Preprocessing - numeric only (our synthetic data)
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Numeric columns:", numeric_cols)

preprocessor = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="median"), numeric_cols)
], remainder='drop')

# 4) Pipeline with EBM
ebm = ExplainableBoostingClassifier(random_state=42)
pipeline = Pipeline(steps=[("pre", preprocessor), ("ebm", ebm)])

# 5) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 6) Fit
print("Training EBM (this may take ~30-90s)...")
pipeline.fit(X_train, y_train)

# 7) Evaluate
y_pred = pipeline.predict(X_test)
try:
    y_proba = pipeline.predict_proba(X_test)[:,1]
    auc_text = f" ROC AUC: {round((__import__('sklearn').metrics.roc_auc_score(y_test, y_proba)), 3)}"
except Exception:
    auc_text = ""
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3), auc_text)
print(classification_report(y_test, y_pred))

# 8) Save model
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/ebm_pipeline.joblib")
print("Saved model to model/ebm_pipeline.joblib")
