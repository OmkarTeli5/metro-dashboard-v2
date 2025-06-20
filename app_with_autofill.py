
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessing objects
model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Metro Station Civil Cost Estimator", layout="wide")
st.title("🚇 Metro Station Civil Cost Estimator")
st.markdown("Estimate metro station civil construction cost using 10 key design parameters.")

# Station type presets
station_type = st.sidebar.selectbox("Select Station Type:", ["Regular", "Terminal", "Interchange", "Custom"])

autofill_presets = {
    "Regular": [1, 1, 3, 100, 150, 20, 15, 5, 2, 1],
    "Terminal": [2, 2, 4, 110, 180, 25, 18, 6, 2, 2],
    "Interchange": [3, 3, 5, 120, 200, 30, 20, 7, 3, 3],
    "Custom": [None] * len(features)
}

# Sidebar input fields
user_inputs = {}
st.sidebar.header("📝 Input Parameters")
for i, feature in enumerate(features):
    default_val = autofill_presets[station_type][i]
    user_inputs[feature] = st.sidebar.number_input(
        label=feature,
        value=float(default_val) if default_val is not None else 0.0,
        step=1.0
    )

# Predict button
if st.button("💰 Predict Civil Cost"):
    try:
        input_df = pd.DataFrame([user_inputs])
        input_df = input_df[features].astype("float64")  # Enforce correct dtype

        if input_df.isnull().values.any():
            st.error("Please fill in all input fields before prediction.")
            st.stop()

        df_imputed = imputer.transform(input_df)
        df_scaled = scaler.transform(df_imputed)
        prediction = model.predict(df_scaled)[0]
        st.success(f"🏗️ Estimated Civil Construction Cost: ₹ {prediction:,.2f} Cr")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Excel batch prediction
st.markdown("### 📂 Batch Prediction via Excel")
uploaded_file = st.file_uploader("Upload Excel file with station parameters", type=["xlsx"])
if uploaded_file:
    try:
        df_excel = pd.read_excel(uploaded_file)
        missing_cols = [col for col in features if col not in df_excel.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            df_excel = df_excel[features].astype("float64")
            df_imputed = imputer.transform(df_excel)
            df_scaled = scaler.transform(df_imputed)
            preds = model.predict(df_scaled)
            df_excel["Predicted Civil Cost (Cr)"] = np.round(preds, 2)
            st.dataframe(df_excel)
            st.download_button("📥 Download Predictions CSV", df_excel.to_csv(index=False), "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {str(e)}")
