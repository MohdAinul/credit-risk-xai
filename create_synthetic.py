# create_synthetic.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os

os.makedirs("data", exist_ok=True)

# Generate synthetic dataset
X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    weights=[0.8, 0.2],
    flip_y=0.02,
    random_state=42
)

df = pd.DataFrame(X, columns=[
    "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3",
    "BILL_AMT1", "BILL_AMT2", "PAY_AMT1", "NUM_OF_CARDS", "ANNUAL_INCOME"
])

# Normalize using numpy
df["LIMIT_BAL"] = (df["LIMIT_BAL"] - df["LIMIT_BAL"].min()) / (df["LIMIT_BAL"].max() - df["LIMIT_BAL"].min()) * 200000 + 10000
df["AGE"] = (df["AGE"] - df["AGE"].min()) / (df["AGE"].max() - df["AGE"].min()) * 50 + 18
df["ANNUAL_INCOME"] = (df["ANNUAL_INCOME"] - df["ANNUAL_INCOME"].min()) / (df["ANNUAL_INCOME"].max() - df["ANNUAL_INCOME"].min()) * 150000 + 5000

# Make NUM_OF_CARDS valid 1–6
df["NUM_OF_CARDS"] = (np.abs(df["NUM_OF_CARDS"]) % 6 + 1).astype(int)

# Add target
df["default"] = y

df.to_csv("data/credit.csv", index=False)
print("Saved synthetic dataset to data/credit.csv -->", df.shape)
