import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from train_evaluate_raw import train_and_evaluate_raw
from model_training import train_models
from model_tuning import tune_models
from preprocessing import preprocess_data

def run_summary(df):
    st.title("📌 Kesimpulan Akhir Evaluasi Model")

    # Dapatkan hasil evaluasi
    df_raw = train_and_evaluate_raw(show=False)
    (X_train, X_test, y_train, y_test), label_encoders, target_encoder = preprocess_data(df, show=False)
    df_pre = train_models(X_train, X_test, y_train, y_test, target_encoder, show=False)
    df_tuned = tune_models(X_train, X_test, y_train, y_test, target_encoder, show=False)

    # Atur nama indeks
    df_raw.index = [f"{i.replace(' (Raw)', '')} (Raw)" for i in df_raw.index]
    df_pre.index = [f"{i} (Preprocessed)" for i in df_pre.index]
    df_tuned.index = [f"{i} (Tuned)" for i in df_tuned.index]

    # Gabungkan dan atur ulang urutan sesuai model
    all_df = pd.concat([df_raw, df_pre, df_tuned])
    model_order = ['Logistic Regression', 'Random Forest', 'KNN']
    tahap_order = ['Raw', 'Preprocessed', 'Tuned']
    ordered_index = [f"{m} ({t})" for m in model_order for t in tahap_order]
    all_df = all_df.reindex(ordered_index)

    # Tabel perbandingan
    st.markdown("## 📋 Tabel Perbandingan Evaluasi Model")
    st.dataframe(all_df.style.format("{:.4f}"))

    # Tampilkan grafik perbandingan
    st.markdown("## 📊 Grafik Perbandingan Evaluasi Model")
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [
    '#FF6F61', '#6B5B95', '#88B04B',  # Raw
    '#FFA500', '#20B2AA', '#9370DB',  # Preprocessed
    '#6495ED', '#FF69B4', '#228B22'   # Tuned
    ]

    all_df.plot(kind='bar', ax=ax, color=colors)
    ax.set_title("Perbandingan Evaluasi Semua Model")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y')
    ax.legend(loc='lower right')
    st.pyplot(fig)

    # Kesimpulan akhir
    st.markdown("### 📝 Kesimpulan Hasil Evaluasi Model Sebelum Pre-Processing, Setelah Pre-Processing (Tidak Tunning) dan Sesudah Melakukan Hyperparameter Tuning")
    st.markdown("""
    #### Logistic Regression:
    - **Raw (Tanpa Preprocessing):** Akurasi hanya sekitar 52%–53%, dengan Precision dan F1-score sangat rendah, terutama pada kelas seperti *Normal Weight* dan *Overweight Level II*.
    - **Setelah Preprocessing:** Akurasi meningkat signifikan menjadi 86%, meskipun beberapa kelas masih menunjukkan ketidakseimbangan skor.
    - **Setelah Tuning:** Akurasi naik hingga 91%, dan metrik lainnya meningkat merata.
    - **Kesimpulan:** Logistic Regression sangat terbantu oleh preprocessing dan tuning.

    #### Random Forest:
    - **Raw (Tanpa Preprocessing):** Akurasi sudah tinggi di 93%.
    - **Setelah Preprocessing:** Naik menjadi 95%, lebih stabil di semua metrik.
    - **Setelah Tuning:** Performa makin optimal.
    - **Kesimpulan:** Random Forest kuat dari awal, tapi tetap membaik dengan preprocessing & tuning.

    #### KNN:
    - **Raw (Tanpa Preprocessing):** Akurasi hanya 75%, F1 rendah terutama untuk *Normal Weight*.
    - **Setelah Preprocessing:** Naik ke 81%, lebih seimbang antar kelas.
    - **Setelah Tuning:** Meningkat signifikan ke 89–90% dengan performa sangat baik secara keseluruhan.
    - **Kesimpulan:** KNN sangat dipengaruhi oleh preprocessing dan hyperparameter tuning.

    #### 💡 Kesimpulan Umum:
    - Preprocessing memberikan dampak besar terutama pada Logistic Regression dan KNN.
    - Hyperparameter tuning meningkatkan performa lebih lanjut, terutama keseimbangan antar kelas.
    - Random Forest paling stabil dan kuat sejak awal.
    - KNN mengalami peningkatan terbesar dari kondisi raw ke tuned.
    """)