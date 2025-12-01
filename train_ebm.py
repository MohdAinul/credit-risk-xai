import pandas as pd
import joblib, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from interpret.glassbox import ExplainableBoostingClassifier

# Load the dataset
df = pd.read_csv("data/credit.csv")
print("Dataset Loaded:", df.shape)

# Separate input & target
X = df.drop(columns=["default"])
y = df["default"]

# Detect numeric and categorical features
numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", cat_cols)

# Preprocessing
numeric_transformer = SimpleImputer(strategy="median")

if len(cat_cols) > 0:
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )
else:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols)
        ]
    )

# Build Explainable Boosting Model
ebm = ExplainableBoostingClassifier(random_state=42)
model = Pipeline(steps=[
    ("pre", preprocessor),
    ("ebm", ebm)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train the model
print("\nTraining EBM model...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Display results
print("\nModel Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/ebm_pipeline.joblib")
print("\nModel saved at model/ebm_pipeline.joblib")
