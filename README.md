💳 Credit Card Fraud Detection with Streamlit

An end-to-end machine learning project that detects fraudulent credit card transactions and provides an interactive web interface for both single and batch predictions.

🚀 Project Overview

This project builds a complete fraud detection system using:

Machine Learning (Logistic Regression)

scikit-learn Pipeline

Streamlit Web Application

Balanced Dataset Handling

The system predicts whether a transaction is fraudulent (1) or legitimate (0) and can also return the probability of fraud.

📊 Dataset

Dataset used: Standard Credit Card Fraud Detection Dataset

Features:

Time – Seconds elapsed between this transaction and the first transaction.

V1 to V28 – Anonymized PCA-transformed features derived from the original confidential transaction data.

Amount – Transaction amount.

Label:

Class

0 → Legitimate

1 → Fraudulent

🔒 Why V1–V28?

For privacy reasons, the original transaction features were transformed using PCA (Principal Component Analysis).
Each V feature is a numerical component derived from real transaction attributes but does not directly correspond to human-readable fields like merchant or location.

⚙️ Model Architecture
1️⃣ Handling Class Imbalance

The dataset is highly imbalanced (very few fraud cases).

To balance the data:

Separate fraud and legitimate transactions.

Randomly sample legitimate transactions equal to the number of fraud cases.

Combine them into a new balanced dataset.

2️⃣ Feature / Label Split
X = new_dataset.drop(columns='Class')
Y = new_dataset['Class']


Features:

30 numeric columns → Time, V1–V28, Amount

Target:

Class

3️⃣ Train-Test Split

test_size = 0.2

stratify = Y (keeps fraud/legit ratio consistent)

4️⃣ Machine Learning Pipeline

The model is built using a scikit-learn Pipeline:

StandardScaler() → Feature scaling

LogisticRegression(max_iter=1000) → Binary classification

Because everything is wrapped inside a Pipeline:

✔ The frontend can send raw feature values
✔ Scaling happens automatically inside the pipeline
✔ Clean and production-ready structure

5️⃣ Model Outputs

The trained pipeline supports:

predict(X) → Returns:

0 → Not Fraud

1 → Fraud

predict_proba(X) → Returns probability scores

🖥️ Streamlit Frontend (app.py)

The web application provides an easy-to-use interface.

🔹 Model Loading

Automatically searches for .pkl or .joblib files

Displays them in a sidebar dropdown

Loads selected model using joblib.load()

🔍 Prediction Modes
1️⃣ Single Transaction (Manual Input)

User enters:

Time

Amount

V1–V28 values

Each V feature includes a tooltip explaining that it is an anonymized PCA feature.

When "Predict fraud" is clicked:

Input is converted into a single-row DataFrame

Passed to the trained pipeline

Displays:

✅ "Transaction is NOT fraudulent"
❌ "Transaction is LIKELY FRAUDULENT"
📊 Fraud probability (if available)

2️⃣ Batch Prediction (CSV Upload)

User uploads a .csv file containing:

Required columns:

Time

V1–V28

Amount

Optional:

Class (true label for comparison)

After upload:

Displays first 10 rows

Validates required columns

Runs prediction on entire dataset

Adds:

is_fraud_prediction

fraud_probability (if available)

Displays:

First 50 prediction results

Summary statistics:

Total transactions

Fraud predictions

Legitimate predictions

🧠 Why Logistic Regression?

Interpretable model

Fast training

Works well for binary classification

Suitable baseline model for fraud detection

🛠️ Tech Stack

Python

pandas

numpy

scikit-learn

joblib

Streamlit

📦 Project Structure
credit-card-fraud-detection/
│
├── app.py
├── model.joblib
├── requirements.txt
├── README.md
└── dataset/

▶️ How to Run Locally

From the project folder:

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
python -m streamlit run app.py


Then open:

http://localhost:8501

📈 Future Improvements

Use advanced models (Random Forest, XGBoost, Neural Networks)

Deploy on cloud (Streamlit Cloud / AWS / Render)

Add model performance dashboard

Add SHAP for feature importance visualization
