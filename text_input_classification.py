import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from preprocessing import preprocess_data

def run_text_classification():
    st.subheader("📥 Klasifikasi Teks Inputan Manual")

    text_input = st.text_area("Masukkan inputan teks (16 nilai, dipisahkan koma):",
        value="Male,23,1.70,80,yes,yes,2,3,Sometimes,no,2,no,1,1,Sometimes,Public_Transportation")

    if st.button("🔍 Klasifikasikan"):
        if text_input.strip() == "":
            st.warning("Silakan masukkan teks terlebih dahulu.")
            return

        try:
            user_input = text_input.split(",")
            if len(user_input) != 16:
                st.error("❌ Format input harus terdiri dari 16 nilai yang dipisahkan dengan koma.")
                return

            input_dict = {
                'Gender': [user_input[0].strip()],
                'Age': [int(user_input[1].strip())],
                'Height': [float(user_input[2].strip())],
                'Weight': [float(user_input[3].strip())],
                'family_history_with_overweight': [user_input[4].strip()],
                'FAVC': [user_input[5].strip()],
                'FCVC': [float(user_input[6].strip())],
                'NCP': [float(user_input[7].strip())],
                'CAEC': [user_input[8].strip()],
                'SMOKE': [user_input[9].strip()],
                'CH2O': [float(user_input[10].strip())],
                'SCC': [user_input[11].strip()],
                'FAF': [float(user_input[12].strip())],
                'TUE': [float(user_input[13].strip())],
                'CALC': [user_input[14].strip()],
                'MTRANS': [user_input[15].strip()]
            }
            df_input = pd.DataFrame(input_dict)
            df_input_original = df_input.copy()

            # 🔧 Pastikan kolom numerik dalam df_input bertipe numerik
            num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
            for col in num_cols:
                df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
        except Exception as e:
            st.error(f"❌ Gagal parsing input: {e}")
            return

        st.markdown("### 📊 Inputan yang Digunakan untuk Prediksi")
        st.dataframe(df_input)

        # ==== RAW MODEL ====
        st.markdown("## 🔷 Hasil Model Tanpa Pre-processing & Tuning")
        df_raw = pd.read_csv("data/ObesityDataSet.csv")
        df_raw.replace('?', np.nan, inplace=True)
        df_raw.dropna(inplace=True)

        X_raw = df_raw.drop("NObeyesdad", axis=1)
        y_raw = df_raw["NObeyesdad"]
        target_encoder_raw = LabelEncoder().fit(y_raw)
        y_raw_enc = target_encoder_raw.transform(y_raw)

        label_encoders_raw = {}
        categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

        for col in categorical_cols:
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col])
            val = str(df_input[col][0])
            if val not in le.classes_:
                st.error(f"❌ Nilai '{val}' tidak valid untuk kolom '{col}' (Raw Model).")
                return
            df_input[col] = le.transform([val])
            label_encoders_raw[col] = le

        feature_order = X_raw.columns.tolist()
        df_input = df_input[feature_order]

        models_raw = {
            "Logistic Regression (Raw)": LogisticRegression(max_iter=1000),
            "Random Forest (Raw)": RandomForestClassifier(),
            "KNN (Raw)": KNeighborsClassifier()
        }

        for name, model in models_raw.items():
            model.fit(X_raw, y_raw_enc)
            pred = model.predict(df_input)
            st.write(f"**{name}**: {target_encoder_raw.inverse_transform(pred)[0]}")

        # ==== MODEL SETELAH PREPROCESSING ====
        st.markdown("## 🔷 Hasil Model Setelah Pre-processing")
        df_input = df_input_original.copy()

        df = pd.read_csv("data/ObesityDataSet.csv")
        (X_train, X_test, y_train, y_test), label_encoders, target_encoder = preprocess_data(df, show=False)

        for col, le in label_encoders.items():
            val = str(df_input[col][0])
            if val not in le.classes_:
                st.error(f"❌ Nilai '{val}' tidak valid untuk kolom '{col}' (Preprocessed Model).")
                return
            df_input[col] = le.transform([val])

        feature_order = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
                         'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC',
                         'FAF', 'TUE', 'CALC', 'MTRANS']
        df_input = df_input[feature_order]

        scaler = StandardScaler()
        scaler.fit(X_train)
        df_input_scaled = scaler.transform(df_input)

        models_pre = {
            "Logistic Regression (Preprocessed)": LogisticRegression(max_iter=1000),
            "Random Forest (Preprocessed)": RandomForestClassifier(),
            "KNN (Preprocessed)": KNeighborsClassifier()
        }

        for name, model in models_pre.items():
            model.fit(X_train, y_train)
            pred = model.predict(df_input_scaled)
            st.write(f"**{name}**: {target_encoder.inverse_transform(pred)[0]}")

        # ==== MODEL SETELAH TUNING ====
        st.markdown("## 🔷 Hasil Model Setelah Tuning")
        param_grid = {
            'Logistic Regression': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']},
            'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
            'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
        }

        base_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(),
            'KNN': KNeighborsClassifier()
        }

        for name in base_models:
            grid = GridSearchCV(base_models[name], param_grid[name], cv=5, scoring='f1_weighted', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            pred = best_model.predict(df_input_scaled)
            st.write(f"**{name} (Tuned)**: {target_encoder.inverse_transform(pred)[0]}")