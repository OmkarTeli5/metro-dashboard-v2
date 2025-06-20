
import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing components
model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Metro Station Cost Estimator", layout="wide")
st.title("🚇 Metro Station Civil Cost Estimator")
st.markdown("Estimate metro station civil construction cost using 10 key design parameters.")

# Sidebar for station type selection
st.sidebar.header("📋 Input Parameters")
station_type = st.sidebar.selectbox("Select Station Type:", ["Custom Input", "Regular", "Terminal", "Interchange"])

# Autofill presets
autofill_presets = {
    "Regular": [1, 1, 3, 1.1, 200, 20, 10, 3, 2, 1],
    "Terminal": [2, 2, 4, 1.2, 220, 22, 15, 5, 3, 2],
    "Interchange": [3, 1, 5, 1.3, 250, 25, 20, 4, 4, 3],
    "Custom Input": [0] * 10
}

# Collect user input
user_inputs = {}
for i, feature in enumerate(features):
    default_val = autofill_presets[station_type][i]
    user_inputs[feature] = st.sidebar.number_input(
        label=feature,
        value=float(default_val),
        step=1.0
    )

# Predict cost based on user input
if st.button("💰 Predict Civil Cost"):
    try:
        input_df = pd.DataFrame([user_inputs])
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        prediction = model.predict(input_scaled)[0]
        st.success(f"Estimated Civil Cost: ₹{prediction:,.2f} Cr")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Upload Excel for batch predictions
st.subheader("📁 Batch Prediction via Excel")
uploaded_file = st.file_uploader("Upload Excel file with station parameters", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        if not set(features).issubset(df.columns):
            st.error("Uploaded file is missing one or more required columns.")
        else:
            X_imputed = imputer.transform(df[features])
            X_scaled = scaler.transform(X_imputed)
            predictions = model.predict(X_scaled)
            df["Predicted Civil Cost (₹ Cr)"] = predictions
            st.dataframe(df)
            st.download_button("📥 Download Predictions", df.to_csv(index=False), "predicted_costs.csv", "text/csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")
