import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

SEED = 42
np.random.seed(SEED)

from sklearn.datasets import fetch_openml
data = fetch_openml("titanic", version=1, as_frame=True)
df = data.frame

df = df[["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked", "survived"]]
X = df.drop("survived", axis=1)
y = df["survived"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED
)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

numeric_features     = ["age", "sibsp", "parch", "fare"]
categorical_features = ["pclass", "sex", "embarked"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline,     numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

preprocessor.fit(X_train)
print("✅ Preprocessor fitted on training data only.")

X_train_processed = preprocessor.transform(X_train)
X_val_processed   = preprocessor.transform(X_val)
X_test_processed  = preprocessor.transform(X_test)

print(f"Processed train shape : {X_train_processed.shape}")
print(f"Processed val shape   : {X_val_processed.shape}")
print(f"Processed test shape  : {X_test_processed.shape}")

os.makedirs("pipeline", exist_ok=True)
joblib.dump(preprocessor, "pipeline/preprocessor.pkl")
print("✅ Preprocessor saved to pipeline/preprocessor.pkl")