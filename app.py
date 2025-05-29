import streamlit as st
import pandas as pd

from eda import run_eda
from preprocessing import preprocess_data
from model_training import train_models
from model_tuning import tune_models
from train_evaluate_raw import train_and_evaluate_raw
from kesimpulan_akhir import run_summary
from text_input_classification import run_text_classification # ini bisa dihapus

st.set_page_config(page_title="Obesity Classification", layout="wide")

# Menu di sidebar
menu = st.sidebar.selectbox("Pilih Menu", [
    "EDA",
    "Pre-processing",
    "Pelatihan & Evaluasi Tanpa Pre-processing & Tuning",
    "Pelatihan & Evaluasi Sebelum Tuning",
    "Pelatihan & Evaluasi Setelah Tuning",
    "Kesimpulan Akhir",
    "Klasifikasi Teks Input" # ini juga
])

# Load data
df = pd.read_csv("data/ObesityDataSet.csv")

# Routing menu
if menu == "EDA":
    run_eda(df)

elif menu == "Pre-processing":
    (X_train, X_test, y_train, y_test), label_encoders, target_encoder = preprocess_data(df, show=True)

elif menu == "Pelatihan & Evaluasi Tanpa Pre-processing & Tuning":
    train_and_evaluate_raw()

elif menu == "Pelatihan & Evaluasi Sebelum Tuning":
    (X_train, X_test, y_train, y_test), label_encoders, target_encoder = preprocess_data(df, show=False)
    train_models(X_train, X_test, y_train, y_test, target_encoder)

elif menu == "Pelatihan & Evaluasi Setelah Tuning":
    (X_train, X_test, y_train, y_test), label_encoders, target_encoder = preprocess_data(df, show=False)
    tune_models(X_train, X_test, y_train, y_test, target_encoder)

elif menu == "Kesimpulan Akhir":
    run_summary(df)

elif menu == "Klasifikasi Teks Input":
    run_text_classification() # ini juga