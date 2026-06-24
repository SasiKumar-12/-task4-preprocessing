# Task 4 - The Pre-Processing Protocol

## Overview
This project implements a leak-free, reusable preprocessing pipeline for machine learning, built as part of the PlaceMux Phase 1 Industry Immersion program.

---

## Objective
Define and implement the pre-processing protocol - the exact, reusable feature transformations applied before modelling, ensuring no data leakage between splits.

---

## Project Structure
task4-preprocessing/
+-- data/
+-- pipeline/
¦   +-- preprocessor.pkl
+-- preprocess.py
+-- verify.py
+-- .gitignore
+-- README.md

---

## Dataset
- Titanic dataset from OpenML
- Features used: pclass, age, sibsp, parch, fare, sex, embarked
- Target: survived

---

## Pipeline Details

### Numeric Features
| Feature | Transforms Applied |
|---|---|
| age | Median Imputation ? Standard Scaling |
| fare | Median Imputation ? Standard Scaling |
| sibsp | Median Imputation ? Standard Scaling |
| parch | Median Imputation ? Standard Scaling |

### Categorical Features
| Feature | Transforms Applied |
|---|---|
| sex | Most Frequent Imputation ? One Hot Encoding |
| embarked | Most Frequent Imputation ? One Hot Encoding |
| pclass | Most Frequent Imputation ? One Hot Encoding |

---

## Key Concepts Demonstrated
- No data leakage - pipeline fitted only on training data
- Missing value imputation - inside the pipeline, not before splitting
- Consistent transforms - same pipeline used at train and inference time
- Saved preprocessor - reloadable via joblib for inference reuse

---

## Tools & Stack
- Python 3.x
- scikit-learn (ColumnTransformer, Pipeline)
- pandas
- numpy
- joblib

---

## Author
SasiKumar-12
PlaceMux - Altrodav Technologies Pvt. Ltd. - Phase 1 Industry Immersion
