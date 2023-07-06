import streamlit as st
import numpy as np
from web_functions import predict, load_data, proses_data, train_model
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def app(dh, x, y):
    st.title("Halaman Prediksi")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age (Tahun)')
        sex = st.text_input('Sex (Perempuan = 0 , Laki - Laki = 1):')
        cp = st.text_input('CP (0-4):')
        trestbps = st.text_input('TrestBps (mmHg):')
    with col2:
        chol = st.text_input('Chol (mg/dl):')
        fbs = st.text_input(
            'Fbs (0,1):')
        restecg = st.text_input('RestecG (0,1,2):')
        thalach = st.text_input('Thalac :')
    with col3:
        exang = st.text_input('Exang (0,1):')
        oldpeak = st.text_input('OldPeak (0,1,2):')
        slope = st.text_input('Slope (0,1,2,3):')
        ca = st.text_input('CA (0,1,2):')
        thal = st.text_input('Thal (0,1):')

    # Convert input values to numpy array
    features = np.array([age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak, slope, ca, thal])

    dh, x, y = load_data()
    x_train, x_test, y_train, y_test = proses_data(x, y)

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
