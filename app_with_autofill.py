import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load files
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
features = joblib.load("features.pkl")

# Set Streamlit config
st.set_page_config(page_title="Metro Station Civil Cost Estimator", layout="wide")
st.title("🚇 Metro Station Civil Cost Estimator")
st.markdown("Estimate metro station civil construction cost using 10 key design parameters.")

# Station presets
station_type = st.selectbox("Select Station Type:", ["Regular", "Terminal", "Interchange", "Custom"])

autofill_presets = {
    "Regular": ["delhi", "underground", 3, 125, 210, 32, 18, 5, 3, "typ-a"],
    "Terminal": ["mumbai", "elevated", 2, 145, 180, 30, 16, 4, 2, "typ-b"],
    "Interchange": ["chennai", "underground", 4, 160, 240, 36, 20, 6, 4, "typ-c"],
    "Custom": [None] * len(features)
}

# Input form
user_inputs = {}
st.sidebar.header("🧮 Input Parameters")
for i, feature in enumerate(features):
    default_val = autofill_presets[station_type][i]
    if default_val is not None:
        user_inputs[feature] = st.sidebar.text_input(label=feature, value=str(default_val))
    else:
        user_inputs[feature] = st.sidebar.text_input(label=feature)

# Predict
def predict_cost(input_dict):
    df = pd.DataFrame([input_dict])
    df = df.astype({col: "float" if col not in ["city", "metro_type", "station_typology"] else "object" for col in df.columns})
    X = preprocessor.transform(df)
    return round(model.predict(X)[0], 2)

if st.button("💰 Predict Civil Cost"):
    try:
        cost = predict_cost(user_inputs)
        st.success(f"Estimated Civil Cost: ₹{cost} Cr")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Excel Upload
st.markdown("### 📂 Batch Prediction via Excel")
uploaded_file = st.file_uploader("Upload Excel file with station parameters", type=["xlsx"])

if uploaded_file is not None:
    try:
        df_excel = pd.read_excel(uploaded_file)
        missing_cols = [col for col in features if col not in df_excel.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            df_excel = df_excel.astype({col: "float" if col not in ["city", "metro_type", "station_typology"] else "object" for col in df_excel.columns})
            df_prepared = preprocessor.transform(df_excel)
            preds = model.predict(df_prepared)
            df_excel["Predicted Civil Cost (Cr)"] = np.round(preds, 2)
            st.success("Batch prediction successful!")
            st.dataframe(df_excel)
            csv = df_excel.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")
