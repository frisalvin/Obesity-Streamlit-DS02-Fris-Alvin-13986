import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

def run_eda(df):
    st.subheader("Exploratory Data Analysis (EDA)")

    # Tampilkan 5 data teratas
    st.markdown("### Contoh Isi Dataset")
    st.dataframe(df.head())

    # Informasi DataFrame
    st.markdown("### Informasi DataFrame")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    # Statistik deskriptif
    st.markdown("### Statistik Deskriptif")
    st.dataframe(df.describe())

    # Cek missing values, duplikasi, dan unique values
    st.markdown("### Cek Data Null, Duplikat, dan Unik")
    st.write("Missing Values:")
    st.dataframe(df.isnull().sum())
    st.write(f"Data Duplikat: {df.duplicated().sum()}")
    st.write("Jumlah Data Unik per Kolom:")
    st.dataframe(df.nunique())

    # Distribusi kelas target
    st.markdown("### Distribusi Kelas Obesitas (Target)")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='NObeyesdad', data=df, order=df['NObeyesdad'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title('Distribusi Kelas Obesitas')
    st.pyplot(plt.gcf())  # Menampilkan plot yang aktif

    # Boxplot kolom numerik
    st.markdown("### Deteksi Outlier (Boxplot Kolom Numerik)")
    num_cols = ['Age', 'Height', 'Weight']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(num_cols):
        sns.boxplot(data=df, y=col, ax=axes[i])
        axes[i].set_title(f'Boxplot {col}')
    plt.tight_layout()
    st.pyplot(fig)

    # Visualisasi data kategorikal
    st.markdown("### Visualisasi Kolom Kategorikal")
    cat_cols = ['Gender', 'CALC', 'FAVC', 'SMOKE', 'CAEC', 'MTRANS']
    fig_cat, axes_cat = plt.subplots(2, 3, figsize=(20, 10))
    axes_cat = axes_cat.flatten()
    for i, col in enumerate(cat_cols):
        sns.countplot(data=df, x=col, ax=axes_cat[i])
        axes_cat[i].set_title(f'Distribusi {col}')
        axes_cat[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_cat)

    # Kesimpulan EDA
    st.markdown("### Kesimpulan EDA")
    st.markdown("""
    - Dataset terdiri dari 17 kolom dan 2111 baris.
    - Banyak kolom bertipe object, termasuk beberapa yang seharusnya bertipe numerik.
    - Terdapat missing values dan data duplikat yang perlu dibersihkan.
    - Distribusi kelas target obesitas tidak seimbang, yang dapat memengaruhi performa model klasifikasi.
    - Beberapa kolom numerik seperti Age, Height, dan Weight mengindikasikan adanya outlier yang perlu ditangani.
    - Dari visualisasi kolom kategorikal:
        - Kolom seperti **Gender**, **CALC**, **SMOKE**, dan **MTRANS** memiliki nilai tidak valid seperti tanda tanya ('?') yang harus dibersihkan atau dikategorikan ulang.
        - Distribusi kategori pada beberapa kolom sangat tidak merata, sehingga perlu penanganan lebih lanjut seperti encoding atau teknik balancing sebelum modelling.
    """)