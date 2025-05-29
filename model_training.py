import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time  # 🆕 Tambahan untuk mengukur waktu pelatihan

def train_models(X_train, X_test, y_train, y_test, target_encoder, show=True):
    if show:
        st.subheader("🤖 Training Model")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier()
    }

    results = {}

    for name, model in models.items():
        if show:
            st.markdown(f"### 🔹 {name}")
        
        start = time.time()  # ⏱️ Mulai waktu pelatihan
        model.fit(X_train, y_train)
        end = time.time()    # ⏱️ Selesai

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        training_time = end - start

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "Training Time (s)": training_time  # 🆕 Ditambahkan ke hasil
        }

        if show:
            st.write(f"**Akurasi:** {acc:.4f}")
            st.text(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=target_encoder.classes_,
                        yticklabels=target_encoder.classes_)
            ax.set_title(f"Confusion Matrix: {name}")
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            st.pyplot(fig)

    results_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time (s)']]  # 🆕 kolom waktu

    if show:
        st.markdown("### 📋 Tabel Hasil Evaluasi Model")
        st.dataframe(results_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "Training Time (s)": "{:.2f}"
        }))
        st.markdown("### 📊 Grafik Perbandingan Performa Model")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        results_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(kind='bar', ax=ax_bar, colormap='Set2')
        ax_bar.set_title("Perbandingan Performa Model (Setelah Pre-Processing dan Sebelum Tunning)")
        ax_bar.set_ylabel("Skor")
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_xticklabels(results_df.index, rotation=0)
        ax_bar.legend(loc='lower right')
        ax_bar.grid(axis='y')
        st.pyplot(fig_bar)
        
        st.markdown("### 📝 Kesimpulan Pelatihan Model Setelah Pre-Processing dan Sebelum Tunning")
        st.markdown("""
        - didapatkan setelah melakukan pre-processing dengan benar saat pelatihan model dengan menggunakan algoritma logistic regression, random forest, dan knn. ketiga model memberikan hasil evaluasi cukup bagus dan menandakan model telah berhasil dilatih dengan minim kesalahan untuk dataset obesitas.
        - diperoleh hasil tertinggi masih sama yaitu random forest dengan hasil rata-rata masih diatas 90%, dan setelah dilakukannya pre-processing model ini justru mengalami peningkatan performanya hingga mendekati angka 96% yang dimana menandakan bahwa algoritma ini berhasil mengklasifikasikan hampir semuanya benar untuk kelas targetnya.
        """)

    return results_df