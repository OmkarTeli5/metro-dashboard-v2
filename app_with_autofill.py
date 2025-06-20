
import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessors
model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# Page config
st.set_page_config(page_title="Metro Station Civil Cost Predictor", layout="wide")
st.title("🚇 Metro Station Civil Cost Predictor (ML-Based)")
st.markdown("Predict estimated civil cost for metro stations using your trained machine learning model.")

# Sidebar for input method
st.sidebar.header("📊 Input Station Parameters")
station_type = st.sidebar.selectbox("Autofill with Station Type:", ["Custom Input", "Regular", "Terminal", "Interchange"])

# Autofill presets
autofill_presets = {
    "Regular": {
        'city': 'Delhi',
        'metro_type': 'Underground',
        'seismic_zone': 4,
        'regional_cost_index': 1.1,
        'station_length_m': 210.0,
        'station_width_m': 9622.0,
        'station_depth_m': 20.0,
        'elevation_height_m': 0.0,
        'levels': 2,
        'station_typology': 'Regular'
    },
    "Terminal": {
        'city': 'Mumbai',
        'metro_type': 'Underground',
        'seismic_zone': 3,
        'regional_cost_index': 1.3,
        'station_length_m': 240.0,
        'station_width_m': 10500.0,
        'station_depth_m': 25.0,
        'elevation_height_m': 0.0,
        'levels': 3,
        'station_typology': 'Terminal'
    },
    "Interchange": {
        'city': 'Bangalore',
        'metro_type': 'Underground',
        'seismic_zone': 2,
        'regional_cost_index': 1.4,
        'station_length_m': 260.0,
        'station_width_m': 12000.0,
        'station_depth_m': 30.0,
        'elevation_height_m': 0.0,
        'levels': 4,
        'station_typology': 'Interchange'
    },
    "Custom Input": {f: "" for f in features}
}

user_inputs = {}
for feature in features:
    default_val = autofill_presets[station_type].get(feature, "")
    if isinstance(default_val, (int, float)):
        user_inputs[feature] = st.sidebar.number_input(label=feature, value=float(default_val) if default_val != "" else 0.0)
    else:
        user_inputs[feature] = st.sidebar.text_input(label=feature, value=str(default_val))

# Predict button
if st.button("📈 Predict Civil Cost"):
    try:
        input_df = pd.DataFrame([user_inputs])
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        prediction = model.predict(input_scaled)[0]
        st.success(f"🧾 Estimated Civil Cost: ₹ {prediction:,.2f} Cr")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Excel Upload
st.subheader("📁 Upload Excel File for Batch Prediction")
uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
if uploaded_file:
    try:
        batch_df = pd.read_excel(uploaded_file)
        batch_imputed = imputer.transform(batch_df[features])
        batch_scaled = scaler.transform(batch_imputed)
        batch_predictions = model.predict(batch_scaled)
        batch_df["Predicted Civil Cost (Cr)"] = batch_predictions
        st.dataframe(batch_df)
        st.download_button("Download Results", data=batch_df.to_csv(index=False), file_name="batch_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
