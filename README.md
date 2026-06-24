# Task 4 — The Pre-Processing Protocol

## Overview
This project implements a leak-free, reusable preprocessing pipeline for machine learning.
The pipeline is designed to transform Titanic dataset features consistently across train,
validation, and test splits without introducing data leakage.

## Objective
Define and implement the exact preprocessing protocol applied before modeling.
This includes selecting features, handling missing values, encoding categorical data,
and scaling numeric features in a way that can be reused on unseen data.

## Usage
- Load the dataset and split it into train, validation, and test sets.
- Fit the preprocessing pipeline only on training data.
- Serialize the fitted preprocessor to `pipeline/preprocessor.pkl`.
- Load the saved preprocessor and transform the test set before evaluation.

## Example
```python
import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

data = fetch_openml("titanic", version=1, as_frame=True)
df = data.frame[["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked", "survived"]]
X = df.drop("survived", axis=1)
y = df["survived"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED
)

preprocessor = joblib.load("pipeline/preprocessor.pkl")
X_test_processed = preprocessor.transform(X_test)

print("✅ Loaded saved preprocessor successfully.")
print(f"Test shape: {X_test_processed.shape}")
print("✅ No leakage — pipeline was never fitted on val/test data.")
```
