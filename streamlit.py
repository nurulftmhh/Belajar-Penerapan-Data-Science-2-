import streamlit as st
import pickle
import numpy as np

# Load model, scaler, dan label encoder
with open('random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('standard_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

# Judul aplikasi
st.title("Prediksi Status Mahasiswa")

# Input dari pengguna
curricular_units_1st_sem_approved = st.number_input("Jumlah Mata Kuliah Semester 1 yang Lulus", min_value=0, max_value=30, value=0)
curricular_units_1st_sem_grade = st.number_input("Rata-rata Nilai Semester 1", min_value=0, max_value=20, value=0)
curricular_units_2nd_sem_approved = st.number_input("Jumlah Mata Kuliah Semester 2 yang Lulus", min_value=0, max_value=30, value=0)
curricular_units_2nd_sem_grade = st.number_input("Rata-rata Nilai Semester 2", min_value=0, max_value=20, value=0)
tuition_fees_up_to_date = st.selectbox("Apakah Biaya Kuliah Sudah Lunas?", ['Ya', 'Tidak'])
scholarship_holder = st.selectbox("Apakah Menerima Beasiswa?", ['Ya', 'Tidak'])
curricular_units_2nd_sem_enrolled = st.number_input("Jumlah Mata Kuliah Semester 2 yang Diambil",  min_value=0, max_value=30, value=0)
curricular_units_1st_sem_enrolled = st.number_input("Jumlah Mata Kuliah Semester 1 yang Diambil", min_value=0, max_value=30, value=0)
admission_grade = st.number_input("Nilai Masuk", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
displaced = st.selectbox("Apakah Mahasiswa Pindahan?", ['Ya', 'Tidak'])

# Mapping nilai kategorikal ke numerik
tuition_fees_up_to_date = 1 if tuition_fees_up_to_date == 'Ya' else 0
scholarship_holder = 1 if scholarship_holder == 'Ya' else 0
displaced = 1 if displaced == 'Ya' else 0

# Membuat array fitur
features = np.array([[curricular_units_2nd_sem_approved,
                      curricular_units_2nd_sem_grade,
                      curricular_units_1st_sem_approved,
                      curricular_units_1st_sem_grade,
                      tuition_fees_up_to_date,
                      scholarship_holder,
                      curricular_units_2nd_sem_enrolled,
                      curricular_units_1st_sem_enrolled,
                      admission_grade,
                      displaced]])

# Prediksi
if st.button("Prediksi Status"):
    features_scaled = scaler.transform(features)
    prediction = rf_model.predict(features_scaled)
    status = le.inverse_transform(prediction)[0]
    
    status_map = {
        0: "Dropout",
        1: "Enrolled",
        2: "Graduate"
    }

    st.success(f"Status Mahasiswa: {status_map[prediction[0]]}")
