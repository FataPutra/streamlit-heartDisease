import streamlit as st
import numpy as np
import pandas as pd
from web_functions import predict, load_data, proses_data, train_model
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def simpan_data_ke_dataset_baru(data, prediction):
    data_array = np.array(data).reshape(1, -1)
    target_array = np.array([prediction]).reshape(1, -1)
    data_array_with_target = np.concatenate((data_array, target_array), axis=1)
    df_baru = pd.DataFrame(data_array_with_target, columns=[
                           'Age', 'Sex', 'CP', 'TrestBps', 'Chol', 'Fbs', 'RestecG', 'Thalac', 'Exang', 'OldPeak', 'Slope', 'CA', 'Thal', 'Target'])
    # Ganti dengan nama file dataset lama yang digunakan
    df_lama = pd.read_csv('dataset_baru.csv')
    df_lama = pd.concat([df_lama, df_baru], ignore_index=True)
    # Ganti dengan nama file dataset lama yang digunakan
    df_lama.to_csv('dataset_baru.csv', index=False)
    st.write("Data berhasil ditambahkan ke dataset lama!")


def tambah_data_ke_dataset_lama(data, prediction):
    data_array = np.array(data).reshape(1, -1)
    target_array = np.array([prediction]).reshape(1, -1)
    data_array_with_target = np.concatenate((data_array, target_array), axis=1)
    # Ganti dengan nama file dataset lama yang digunakan
    df_lama = pd.read_csv('heart1.csv')
    df_baru = pd.DataFrame(data_array_with_target, columns=[
                           'Age', 'Sex', 'CP', 'TrestBps', 'Chol', 'Fbs', 'RestecG', 'Thalac', 'Exang', 'OldPeak', 'Slope', 'CA', 'Thal', 'Target'])
    df_lama = pd.concat([df_lama, df_baru], ignore_index=True)
    # Ganti dengan nama file dataset lama yang digunakan
    df_lama.to_csv('heart1.csv', index=False)
    st.write("Data berhasil ditambahkan ke dataset lama!")


def app(dh, x, y):
    st.title("HALAMAN PREDIKSI")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age (Tahun)')
        sex = st.selectbox(
            'Sex',
            ('0', '1')
        )

        cp = st.selectbox(
            'CP',
            ('0', '1', '2', '3', '4')
        )

        trestbps = st.number_input('TrestBps (mmHg)')
    with col2:
        chol = st.number_input('Chol (mg/dl)')
        fbs = st.selectbox(
            'Fbs',
            ('0', '1')
        )
        restecg = st.selectbox(
            'RestecG',
            ('0', '1', '2')
        )

        thalach = st.number_input('Thalac')
    with col3:
        exang = st.selectbox(
            'Exang',
            ('0', '1')
        )

        oldpeak = st.selectbox(
            'OldPeak',
            ('0', '1', '2')
        )

        slope = st.selectbox(
            'Slope',
            ('0', '1', '2', '3')
        )
        ca = st.selectbox(
            'CA',
            ('0', '1', '2')
        )
        thal = st.selectbox(
            'Thal',
            ('0', '1')
        )

        # Input persentase test_size
        test_size = st.slider('Persentase Data Training (ex 25% = 0.25)', min_value=0.1,
                              max_value=0.9, value=0.25, step=0.05)

        # Input pilihan untuk menyimpan data ke dataset baru atau menambahkannya ke dataset lama
        pilihan_simpan = st.radio(
            'Simpan Data ', ('Dataset Baru', 'Dataset Lama'))

    # Convert input values to numpy array
    features = np.array([age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak, slope, ca, thal])

    dh, x, y = load_data()

    x_train, x_test, y_train, y_test = proses_data(x, y, test_size)

    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)

    classifier = KNeighborsClassifier(n_neighbors=4, metric='euclidean')
    classifier.fit(x_train_scaled, y_train)

    y_pred = classifier.predict(x_test_scaled)

    ac = accuracy_score(y_test, y_pred)

    if col1.button("Prediksi"):
        if any(feature == '' for feature in features):
            st.warning("Mohon lengkapi semua input.")
        else:
            # Konversi nilai atribut dari string menjadi float
            features_float = np.array(features, dtype=float)

            if any(np.isnan(features_float)):
                st.warning(
                    "Terdapat nilai yang tidak valid. Mohon periksa kembali input.")
            else:
                # Skala atribut input menggunakan StandardScaler
                features_scaled = sc.transform(features_float.reshape(1, -1))

                prediction = classifier.predict(features_scaled)

                if prediction == 1:
                    st.warning(
                        "Berdasarkan Prediksi kami menunjukkan rentan terkena Jantung Koroner")
                else:
                    st.success(
                        "Berdasarkan Prediksi kami menunjukkan relatif aman dari Jantung Koroner")

                st.write(
                    "Model yang digunakan memiliki tingkat akurasi ", ac * 100, "%")

                # Simpan data ke dataset baru atau tambahkan ke dataset lama
                if pilihan_simpan == 'Dataset Baru':
                    simpan_data_ke_dataset_baru(features, prediction)
                else:
                    tambah_data_ke_dataset_lama(features, prediction)
