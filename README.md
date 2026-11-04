# KLASIFIKASI KELULUSAN MAHASISWA MENGGUNAKAN MACHINE LEARNING

## 1. Deskripsi Project

Proyek ini merupakan implementasi Machine Learning untuk melakukan klasifikasi terhadap status kelulusan mahasiswa berdasarkan beberapa karakteristik dan aktivitas akademik. Klasifikasi dilakukan dengan memanfaatkan data mahasiswa yang berisi informasi demografi, latar belakang wilayah, keterlibatan dalam pembelajaran, serta tingkat kehadiran.

Tujuan utama dari proyek ini adalah membangun model Machine Learning yang mampu mempelajari pola dari data mahasiswa sebelumnya untuk mengklasifikasikan status mahasiswa ke dalam dua kategori, yaitu **Lulus (0)** dan **Gagal (1)**.

Dalam proses pengerjaannya, digunakan beberapa algoritma klasifikasi untuk membandingkan performa model sehingga dapat dipilih model yang memberikan hasil terbaik berdasarkan nilai F1-Score Cross-Validation.

## 2. Tujuan Proyek

Tujuan dari pembuatan proyek ini adalah:

1. Memahami karakteristik dataset mahasiswa melalui proses Data Understanding dan Exploratory Data Analysis (EDA).
2. Melakukan pembersihan dan persiapan data sebelum digunakan dalam proses pemodelan.
3. Membangun dan membandingkan beberapa algoritma Machine Learning untuk klasifikasi kelulusan mahasiswa.
4. Mengevaluasi performa model menggunakan beberapa metrik evaluasi.
5. Menggunakan model terbaik untuk melakukan prediksi terhadap data mahasiswa baru.

## 3. Dataset

Dataset yang digunakan adalah `original.csv`, yang berisi informasi mengenai mahasiswa dengan beberapa kelompok fitur.

Dataset memiliki 80 baris data dan 20 kolom, yang terdiri dari 19 fitur (variabel input) dan 1 kolom label (target).

Kelompok fitur yang digunakan meliputi:

| Kelompok Fitur | Keterangan |
|---|---|
| Demografi | Gender dan Age |
| Latar Belakang Wilayah | Informasi kuintil wilayah dan latar belakang pendidikan |
| Keterlibatan Belajar | Aktivitas login, durasi belajar, dan akses pembelajaran |
| Kehadiran | Jumlah kehadiran, ketidakhadiran, dan persentase kehadiran |

**Variabel Target (Label):**
- 0 = Lulus (Pass)
- 1 = Gagal (Fail)

Distribusi label pada dataset:
- Lulus: 68 mahasiswa (85%).
- Gagal: 12 mahasiswa (15%).

Data tersebut menunjukkan bahwa jumlah mahasiswa yang lulus lebih banyak dibandingkan mahasiswa yang gagal. Oleh karena itu, ketidakseimbangan kelas perlu diperhatikan selama proses pemodelan.

## 4. Tahapan Pengerjaan

Proyek ini dikerjakan melalui empat tahapan utama, yaitu Data Understanding, Exploratory Data Analysis (EDA), Data Preprocessing, dan Modelling.

### A. Data Understanding

Tahap Data Understanding dilakukan untuk memahami struktur dan karakteristik dataset sebelum dilakukan analisis lebih lanjut.

Beberapa proses yang dilakukan:

1. Mengimpor dataset menggunakan library Pandas.
2. Memeriksa jumlah baris dan kolom menggunakan `shape`.
3. Memeriksa tipe data dan informasi setiap kolom menggunakan `info()`.
4. Menampilkan lima data pertama menggunakan `head()`.
5. Mengidentifikasi variabel target atau label klasifikasi.
6. Mengelompokkan fitur berdasarkan demografi, latar belakang wilayah, keterlibatan belajar, dan kehadiran.
7. Mengidentifikasi distribusi jumlah mahasiswa yang lulus dan gagal.

Hasil tahap ini menunjukkan bahwa dataset memiliki 80 baris, 20 kolom, 19 fitur, dan satu variabel target.

### B. Exploratory Data Analysis (EDA)

Tahap Exploratory Data Analysis (EDA) bertujuan untuk mengetahui pola, hubungan, distribusi, serta karakteristik data melalui analisis statistik dan visualisasi.

Beberapa analisis yang dilakukan:

**1. Distribusi Label**

Menggunakan diagram batang untuk mengetahui perbandingan jumlah mahasiswa yang lulus dan gagal.

**2. Histogram**

Digunakan untuk mengetahui distribusi nilai pada fitur numerik, seperti umur, jarak ke universitas, jumlah login, durasi belajar, dan persentase kehadiran.

**3. Boxplot**

Digunakan untuk mengidentifikasi persebaran data dan kemungkinan adanya outlier pada fitur numerik.

**4. Perbandingan Fitur Berdasarkan Label**

Membandingkan karakteristik mahasiswa yang lulus dan gagal berdasarkan persentase kehadiran, jumlah ketidakhadiran, aktivitas login, durasi belajar, dan jarak ke universitas.

**5. Heatmap Korelasi**

Digunakan untuk melihat hubungan korelasi antarfitur numerik dan hubungan fitur terhadap variabel target.

**6. Analisis Outlier dan Nilai Tidak Wajar**

Pemeriksaan dilakukan menggunakan metode Interquartile Range (IQR), pengecekan nilai kuintil, serta pemeriksaan rentang logis pada persentase kehadiran dan jarak ke universitas.

Hasil analisis menunjukkan bahwa rata-rata persentase kehadiran mahasiswa yang lulus adalah 78,55%, sedangkan mahasiswa yang gagal adalah 50,69%. Rata-rata durasi belajar pada mahasiswa yang lulus juga lebih tinggi dibandingkan mahasiswa yang gagal.

### C. Data Preprocessing

Tahap Data Preprocessing merupakan proses mempersiapkan data agar dapat digunakan oleh algoritma Machine Learning.

Proses yang dilakukan meliputi:

1. **Membersihkan nama kolom:** Menghilangkan spasi yang tidak diperlukan pada nama kolom.
2. **Menghapus data kosong:** Menghapus baris dan kolom yang seluruh nilainya kosong.
3. **Menangani missing value:** Mengisi nilai kosong pada kolom `# of Absence` berdasarkan jumlah pertemuan dikurangi jumlah kehadiran.
4. **Menghapus duplikasi:** Menghilangkan baris data yang memiliki isi sama.
5. **Menghapus label kosong:** Memastikan data yang digunakan memiliki target klasifikasi.
6. **Transformasi Logaritma:** Menggunakan `np.log1p()` pada fitur yang memiliki distribusi miring ke kanan.
7. **Menangani nilai kuintil:** Mengubah nilai 0 pada kolom kuintil menjadi nilai kosong (NaN).
8. **Seleksi fitur:** Menghapus fitur yang redundan, duplikat, atau sensitif berdasarkan pertimbangan etika dan analisis data.
9. **Pembagian dataset:** Membagi data menjadi 80% data latih dan 20% data uji menggunakan `train_test_split()` dengan stratifikasi label.
10. **Pemeriksaan korelasi:** Mengidentifikasi pasangan fitur dengan korelasi absolut lebih dari 0,90 dan menghapus salah satu fitur yang memiliki korelasi tinggi.
11. **Imputasi dan standardisasi:** Menggunakan `SimpleImputer` untuk menangani nilai kosong dan `StandardScaler` untuk menyamakan skala fitur di dalam pipeline model.

Setelah preprocessing dan seleksi fitur, data memiliki 14 fitur akhir yang digunakan dalam proses pemodelan.

### D. Modelling

Tahap Modelling merupakan proses membangun model Machine Learning berdasarkan data yang telah dipersiapkan.

Pada proyek ini digunakan empat algoritma klasifikasi:

1. **Logistic Regression:** Algoritma klasifikasi yang digunakan untuk memperkirakan probabilitas suatu data masuk ke dalam kelas tertentu.
2. **Decision Tree:** Algoritma yang menggunakan struktur percabangan berdasarkan kondisi fitur untuk menentukan hasil klasifikasi.
3. **Random Forest:** Algoritma yang menggabungkan beberapa Decision Tree untuk menghasilkan prediksi klasifikasi.
4. **Support Vector Machine (SVM):** Algoritma yang mencari batas pemisah terbaik antara kelas menggunakan kernel RBF.

Setiap algoritma diterapkan menggunakan Pipeline yang menggabungkan proses imputasi nilai kosong, standardisasi data, dan pemodelan.

Untuk mengatasi ketidakseimbangan kelas, beberapa model menggunakan pengaturan `class_weight='balanced'` atau `class_weight='balanced_subsample'`.

**Cross-Validation**

Evaluasi awal model menggunakan `RepeatedStratifiedKFold` dengan:
- 5 fold atau lipatan.
- 3 kali pengulangan.
- Total 15 proses evaluasi untuk setiap model.

Metrik yang digunakan adalah Accuracy, Precision, Recall, F1-Score, dan ROC-AUC.

**Hyperparameter Tuning**

Algoritma Random Forest juga dioptimalkan menggunakan `GridSearchCV` untuk mencari kombinasi parameter terbaik, seperti:
- `n_estimators`
- `max_depth`
- `min_samples_leaf`
- `max_features`

Proses tuning menggunakan F1-Score sebagai dasar pemilihan parameter terbaik.

## 5. Evaluasi Model

Evaluasi dilakukan untuk mengetahui kemampuan model dalam mengklasifikasikan mahasiswa yang lulus maupun gagal.

Metrik evaluasi yang digunakan:

| Metrik | Fungsi |
|---|---|
| Accuracy | Mengukur persentase keseluruhan prediksi yang benar. |
| Precision | Mengukur ketepatan prediksi pada kelas positif (Gagal). |
| Recall | Mengukur kemampuan model menemukan data yang benar-benar termasuk kelas Gagal. |
| F1-Score | Menggabungkan Precision dan Recall dalam satu nilai. |
| ROC-AUC | Mengukur kemampuan model membedakan dua kelas berdasarkan skor probabilitas. |

Hasil evaluasi pada data uji:

| Algoritma | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.875 | 0.500 | 0.500 | 0.500 | 0.857 |
| Decision Tree | 0.875 | 0.500 | 0.500 | 0.500 | 0.732 |
| Random Forest | 0.938 | 1.000 | 0.500 | 0.667 | 0.750 |
| SVM (RBF) | 0.938 | 1.000 | 0.500 | 0.667 | 0.821 |
| Random Forest (Tuned) | 0.938 | 1.000 | 0.500 | 0.667 | 0.679 |

**Pemilihan Model Terbaik**

Berdasarkan hasil Cross-Validation, Logistic Regression dipilih sebagai model terbaik karena memperoleh nilai F1-Score Cross-Validation tertinggi, yaitu 0.503.

Pemilihan model terbaik dilakukan berdasarkan hasil Cross-Validation pada data latih, bukan berdasarkan hasil data uji, untuk menghindari pemilihan model berdasarkan data pengujian.

## 6. Analisis Fitur Penting

Setelah model terbaik diperoleh, dilakukan analisis fitur menggunakan metode `Permutation Importance`.

Metode ini digunakan untuk mengetahui seberapa besar penurunan performa model ketika nilai suatu fitur diacak.

Lima fitur dengan pengaruh terbesar berdasarkan hasil analisis adalah:

1. `% of Average Hours in Module Area` = 0.2114
2. `attending from home?` = 0.0482
3. `POLAR4 Quintile` = 0.0433
4. `POLAR3 Quintile` = 0.0320
5. `Gaps GCSE Quintile` = 0.0283

Hasil tersebut menunjukkan bahwa persentase rata-rata jam aktivitas mahasiswa pada Module Area merupakan fitur dengan pengaruh terbesar terhadap performa model dalam analisis ini.

## 7. Prediksi Mahasiswa Baru

Tahap terakhir adalah melakukan pengujian prediksi terhadap lima data mahasiswa baru.

Karena belum menggunakan data mahasiswa baru yang berasal dari pengumpulan data aktual, program membangkitkan lima data contoh menggunakan NumPy dengan rentang nilai berdasarkan dataset yang tersedia.

Selanjutnya, data tersebut diproses menggunakan fungsi `siapkan_data()` agar memiliki tahapan preprocessing yang sama dengan data latih.

Model Logistic Regression kemudian digunakan untuk menghasilkan:
- Probabilitas kelulusan.
- Probabilitas kegagalan.
- Hasil prediksi status mahasiswa.

**Hasil prediksi pada notebook:**

| Mahasiswa | Kehadiran | Prediksi |
|---|---:|---|
| Mahasiswa baru 1 | 91,67% | LULUS |
| Mahasiswa baru 2 | 8,33% | LULUS |
| Mahasiswa baru 3 | 58,33% | LULUS |
| Mahasiswa baru 4 | 50,00% | LULUS |
| Mahasiswa baru 5 | 100,00% | LULUS |

Seluruh data contoh pada pengujian tersebut diprediksi masuk ke kategori LULUS oleh model.

Perlu diperhatikan bahwa data mahasiswa baru tersebut merupakan data simulasi, bukan data aktual. Oleh karena itu, hasil prediksi hanya digunakan sebagai demonstrasi cara kerja model dan belum dapat dijadikan dasar untuk menentukan kelulusan mahasiswa sebenarnya.

## 8. Kesimpulan

Berdasarkan seluruh tahapan yang telah dilakukan, dapat disimpulkan bahwa proses klasifikasi kelulusan mahasiswa berhasil dibangun melalui tahapan Data Understanding, Exploratory Data Analysis, Data Preprocessing, dan Modelling.

Empat algoritma klasifikasi telah dibandingkan, yaitu Logistic Regression, Decision Tree, Random Forest, dan SVM (RBF). Selain itu, dilakukan optimasi parameter pada Random Forest menggunakan GridSearchCV.

Berdasarkan hasil Cross-Validation, Logistic Regression memperoleh F1-Score tertinggi sebesar 0,503 sehingga dipilih sebagai model terbaik dalam proyek ini.

Hasil analisis juga menunjukkan bahwa fitur `% of Average Hours in Module Area` memiliki pengaruh terbesar berdasarkan metode Permutation Importance.

Meskipun demikian, dataset yang digunakan memiliki jumlah data relatif kecil dan distribusi kelas yang tidak seimbang. Oleh karena itu, model masih memerlukan pengujian lebih lanjut menggunakan dataset yang lebih besar dan data mahasiswa aktual agar kemampuan generalisasinya dapat diketahui dengan lebih baik.

## 9. Tools dan Library

Proyek ini menggunakan bahasa pemrograman Python dengan beberapa library berikut:

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## 10. Struktur File Repository

Struktur repository GitHub yang disarankan:

```text
Klasifikasi-Kelulusan-Mahasiswa/
│
├── UTS01(1).ipynb
├── original.csv
└── README.md
```

Keterangan:
- `UTS01(1).ipynb`: Berisi seluruh kode program, proses analisis, preprocessing, pemodelan, evaluasi, dan prediksi.
- `original.csv`: Dataset yang digunakan dalam proses analisis dan pelatihan model.
- `README.md`: Dokumentasi yang menjelaskan tujuan, tahapan pengerjaan, hasil, dan kesimpulan proyek.

## 11. Cara Menjalankan Program

1. Pastikan Python dan Jupyter Notebook sudah terinstal.
2. Instal library yang diperlukan:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

3. Download atau clone repository GitHub.
4. Pastikan file `original.csv` dan notebook berada dalam folder yang sama.
5. Buka terminal pada folder proyek.
6. Jalankan Jupyter Notebook:

```bash
jupyter notebook
```

7. Buka file `UTS01(1).ipynb`.
8. Jalankan seluruh cell secara berurutan menggunakan menu **Run All** atau tombol `Shift + Enter`.

---

**Catatan:** Dataset `original.csv` harus tersedia agar notebook dapat dijalankan tanpa mengalami kesalahan `FileNotFoundError`.
