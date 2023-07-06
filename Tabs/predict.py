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
        age = st.number_input('Age (Tahun)')
        option = st.selectbox(
            'Sex'
            (0, 1)
        )
        sex = option

        option2 = st.selectbox(
            'CP'
            (0, 1, 2, 3, 4)
        )
        cp = option2

        trestbps = st.number_input('TrestBps (mmHg)')

    with col2:
        chol = st.number_input('Chol (mg/dl)')

        option3 = st.selectbox(
            'Fbs'
            (0, 1)
        )
        fbs = option3

        option4 = st.selectbox(
            'RestecG'
            (0, 1, 2)
        )
        restecg = option4

        thalach = st.number_input('Thalac')
    with col3:
        option5 = st.selectbox(
            'Exang'
            (0, 1)
        )
        exang = option5

        option6 = st.selectbox(
            'OldPeak'
            (0, 1, 2)
        )
        oldpeak = option6

        option7 = st.selectbox(
            'Slope'
            (0, 1, 2, 3)
        )
        slope = option7

        option8 = st.selectbox(
            'CA'
            (0, 1, 2)
        )
        ca = option8

        option9 = st.selectbox(
            'Thal'
            (0, 1)
        )

        thal = option9

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
