# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan memiliki reputasi baik dalam mencetak lulusan berkualitas. Namun, institusi ini menghadapi tantangan serius berupa tingginya angka siswa yang dropout (tidak menyelesaikan pendidikan). Masalah ini tidak hanya memengaruhi reputasi institusi, tetapi juga efisiensi operasional, perencanaan akademik, serta tingkat keberhasilan program pendidikan yang dijalankan. Oleh karena itu, perlu dianalisis faktor-faktor yang mempengaruhi tingkat keluar/dropout dari siswa Jaya Jaya Institut.  

## Permasalahan Bisnis
Permasalahan utama yang dihadapi yaitu tingginya angka siswa yang dropout, namun belum terdapat sistem yang dapat mengidentifikasi siswa yang beresiko dropout sehingga akan kesulitan untuk memberikan bimbingan khusus.

## Cakupan Proyek
1. Melakukan analisis data untuk mengidentifikasi faktor-faktor yang mempengaruhi status siswa.
2. Membuat dashboard untuk mempermudah analisis secara visual.
3. Membangun model machine learning yang dapat digunakan user dalam bentuk prototype untuk memprediksi status siswa berdasarkan inputan yang diberikan.

## Persiapan
Sumber Data: [Dataset Student Performance](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)

Setup environment:
**Install Dependencies**

pip install -r requirements.txt

### Buka dan Jalankan file .ipynb
1. Pastikan dependensi yang ada dalam daftar requirements.txt telah terinstall.
2. Buka dan jalankan seluruh isi notebokk_sub2.ipynb menggunakan Google Colab
3. Temukan visualisasi dan insight yang diperoleh untuk menemukan pola-pola yang relevan.

### Menjalankan Dashboard 
Dashboard dibuat dengan Tableau, sehingga untuk melihatnya dapat melalui URL dashboard yang dicantumkan.
- [Dashboard 1](https://public.tableau.com/app/profile/nurul.fatimah4077/viz/Dropout_17490550307390/Dashboard1)
- [Dashboard 2](https://public.tableau.com/app/profile/nurul.fatimah4077/viz/Dropout_17490550307390/Dashboard2)

## Menjalankan Sistem Machine Learning
Sistem prediksi dibuat dengan menggunakan streamlit dengan memasukkan data siswa dan akan mendapatkan hasil prediksinya. Untuk menjalankan protoype secara lokal jalankan perintah berikut di terminal: streamlit run app.py . Selain itu juga dapat diakses melalui url ini: [Prototype Streamlit](https://predictstatusmhs.streamlit.app/)

## Conclusion
Dari dashboard dan visualisasi yang dihasilkan dapat ditemukan beberapa faktor yang mempengaruhi status droput siswa diantaranya:
1. Terlambat membayar UKT 
2. Nilai akademik rendah (semester 1 & 2)
3. Jumlah mata kuliah yang lulus rendah
4. Jurusan tertentu dengan beban atau seleksi berbeda
5. Usia masuk yang lebih tua

Analisis korelasi dan feature importance yang dilakukan menenjukkan bahwa terdapat beberapa faktor yang paling berpengaruh terhadap status mahasiswa, yaitu 'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade', 'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade', 'Tuition_fees_up_to_date', 'Scholarship_holder', 'Curricular_units_2nd_sem_enrolled', 'Curricular_units_1st_sem_enrolled', 'Admission_grade', 'Displaced'. Selain itu, model machine learning yang dibangun menggunakan Random Forest berhasil mencapai akurasi ~77%. 

## Rekomendasi Action Items
1. Implementasi Sistem Early Warning
    Terapkan sistem prediksi dropout berbasis machine learning untuk mengidentifikasi mahasiswa dengan risiko tinggi sejak awal semester.

2. Tingkatkan Program untuk Dukungan Akademik maupun Finansial
Bisa dengan menambahkan program tutoring atau mentoring untuk mahasiswa baru. Selain itu juga dapat menghubungkan sistem keuangan dengan sistem early warning agar mahasiswa yang telat bayar bisa langsung dibantu.

3. Evaluasi dan Penyesuaian Kurikulum
Lakukan evaluasi terkait kurikulum maupun proses pembelajaran terhadap jurusan yang memiliki resiko dropout tinggi. 
