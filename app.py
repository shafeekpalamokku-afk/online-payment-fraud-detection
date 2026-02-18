# ===============================
# STREAMLIT FRAUD DETECTION APP
# ===============================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("💳 Online Payment Fraud Detection")


# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("online_payment_fraud_realistic.csv")
    df.columns = df.columns.str.strip()   # Remove hidden spaces
    return df


df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())


# ===============================
# CHECK TARGET COLUMN
# ===============================
target_column = "is_fraud"

if target_column not in df.columns:
    st.error(f"❌ Target column '{target_column}' not found in dataset.")
    st.stop()


# ===============================
# PREPROCESSING
# ===============================
df_model = df.copy()

# Separate features & target
X = df_model.drop(columns=[target_column])
y = df_model[target_column]

# Encode categorical columns
label_encoders = {}
categorical_cols = X.select_dtypes(include="object").columns

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ===============================
# MODEL TRAINING
# ===============================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

st.success("✅ Model Trained Successfully")


# ===============================
# MODEL PERFORMANCE
# ===============================
st.subheader("📈 Model Evaluation")

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

st.dataframe(pd.DataFrame(report).transpose())


# ===============================
# NEW TRANSACTION PREDICTION
# ===============================
st.header("🔍 Predict New Transaction")

input_data = {}

for col in X.columns:
    if col in label_encoders:
        input_data[col] = st.selectbox(
            f"{col}",
            label_encoders[col].classes_
        )
    else:
        input_data[col] = st.number_input(
            f"{col}",
            value=float(X[col].mean())
        )


if st.button("Predict Transaction"):

    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Scale
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected!")
        st.write(f"Fraud Probability: {probability:.2%}")
    else:
        st.success("✅ Legitimate Transaction")
        st.write(f"Fraud Probability: {probability:.2%}")
