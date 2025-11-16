import streamlit as st
import pandas as pd
from model_utils import load_data, preprocess, train_models, predict_new
from visuals import render_dashboard

st.set_page_config(layout="wide", page_title="Solar Powerbank Market Dashboard")

st.title("🔆 Solar Powerbank Market Insights Dashboard")

uploaded = st.file_uploader("Upload the dataset (.csv or .xlsx)", type=["csv","xlsx"])

if uploaded:
    df = load_data(uploaded)
    st.success("Dataset loaded successfully!")

    tab1, tab2 = st.tabs(["📊 Insights Dashboard", "🤖 Prediction Tool"])

    with tab1:
        render_dashboard(df)

    with tab2:
        st.subheader("Enter Customer Details for Prediction")
        X, y, encoders = preprocess(df)

        user_input = {}
        for col in X.columns:
            user_input[col] = st.text_input(f"{col}")

        if st.button("Predict"):
            result = predict_new(user_input, df)
            st.success(f"Prediction: {result}")

            pred_df = pd.DataFrame([user_input])
            pred_df["Prediction"] = result
            pred_df.to_csv("/mnt/data/new_prediction.csv", index=False)

            with open("/mnt/data/new_prediction.csv", "rb") as f:
                st.download_button("Download Prediction CSV", f, "prediction.csv")

else:
    st.info("Upload dataset to begin.")
