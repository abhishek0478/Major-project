
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Model Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Model Performance Dashboard")

# Load saved test data and model
test_file = BASE_DIR / "data" / "processed" / "customer_snapshots.csv"
model = joblib.load(BASE_DIR / "models" / "customer_purchase_model.pkl")

if not test_file.exists():
    st.error("Test dataset not found. Check data/processed/customer_snapshots.csv")
    st.stop()

snapshots = pd.read_csv(test_file, parse_dates=["snapshot_date"])

features = joblib.load(BASE_DIR / "models" / "feature_columns.pkl")

# Use the same final test period as model evaluation
test = snapshots[
    snapshots["snapshot_date"] == pd.Timestamp("2011-09-01")
].copy()

X_test = test[features]
y_test = test["buys_again_90d"]

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.50).astype(int)

# Performance metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.1%}")
col2.metric("Precision", f"{precision:.1%}")
col3.metric("Recall", f"{recall:.1%}")
col4.metric("F1 Score", f"{f1:.3f}")

st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, ax=ax
)
st.pyplot(fig)
plt.close(fig)

st.subheader("Prediction Probability Distribution")

chart_data = pd.DataFrame({
    "Probability": probabilities,
    "Actual Outcome": y_test.to_numpy()
})

st.bar_chart(
    chart_data.groupby("Actual Outcome")["Probability"].mean()
)

st.caption(
    "These metrics use the September 2011 snapshot. "
    "They reflect the saved model and a 0.50 decision threshold."
)
