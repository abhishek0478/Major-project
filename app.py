
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load saved model
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "models" / "customer_purchase_model.pkl"
)
feature_columns = joblib.load(
    BASE_DIR / "models" / "feature_columns.pkl"
)

st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ Customer Purchase Prediction")
st.write(
    "Predict whether a customer is likely to make "
    "another purchase within the next 90 days."
)

st.info("Enter the customer's purchase history below.")

# Customer inputs
st.subheader("Customer Details")

recency_days = st.number_input(
    "Days since last purchase",
    min_value=0,
    value=30
)

total_invoices = st.number_input(
    "Total previous invoices",
    min_value=1,
    value=5
)

frequency_90d = st.number_input(
    "Purchases in the last 90 days",
    min_value=0,
    value=2
)

frequency_180d = st.number_input(
    "Purchases in the last 180 days",
    min_value=0,
    value=4
)

monetary_180d = st.number_input(
    "Total spending in the last 180 days",
    min_value=0.0,
    value=500.0,
    step=100.0
)

if st.button("Predict Purchase", type="primary"):

    customer = pd.DataFrame([{
        "recency_days": recency_days,
        "total_invoices": total_invoices,
        "frequency_90d": frequency_90d,
        "frequency_180d": frequency_180d,
        "monetary_180d": monetary_180d
    }])

    customer = customer[feature_columns]

    probability = model.predict_proba(customer)[0][1]
    prediction = int(probability >= 0.50)

    st.divider()
    st.subheader("Prediction Result")

    st.metric(
        "Estimated probability of another purchase",
        f"{probability:.1%}"
    )

    st.progress(float(probability))

    if prediction == 1:
        st.success(
            "The model predicts that this customer "
            "is likely to purchase again within 90 days."
        )
    else:
        st.warning(
            "The model predicts that this customer "
            "is less likely to purchase again within 90 days."
        )

    st.caption(
        "This is a model estimate, not a guarantee. "
        "The 50% decision threshold is based on the "
        "current validation results."
    )

    with st.expander("View input summary"):
        st.dataframe(customer, use_container_width=True)
