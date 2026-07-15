# MeiSkinGlow - Smart Skincare Recommendation Powered by AI

MeiSkinGlow adalah aplikasi kecerdasan buatan (AI) sederhana berbasis web untuk tugas mata kuliah **Pengantar Kecerdasan Buatan**. Aplikasi ini menganalisis foto wajah pengguna, mengidentifikasi jenis kulit wajah (Acne, Oily, Normal, Dry, Sensitive), dan memberikan rekomendasi produk serta kandungan skincare yang sesuai.

---

## 🌟 Fitur Utama

1. **Landing Page**: Berisi pengenalan aplikasi, ringkasan alur kerja AI, dan tombol mulai analisis.
2. **Kamera & Unggah Foto**: Pengguna dapat memilih mengunggah foto wajah lokal atau menggunakan kamera perangkat langsung (HTML5 `getUserMedia` API) dengan pratinjau instan.
3. **Deteksi Wajah (OpenCV)**: Melakukan verifikasi wajah menggunakan OpenCV Haar Cascade. Jika tidak terdeteksi wajah manusia, sistem menampilkan pesan error dan membatalkan klasifikasi.
4. **Klasifikasi Jenis Kulit (CNN)**: Melakukan klasifikasi 5 jenis kulit (Acne, Oily, Normal, Dry, Sensitive) menggunakan model Convolutional Neural Network (CNN) berbasis TensorFlow & Keras.
5. **Ambangan Keyakinan (Confidence Threshold)**: Jika keyakinan prediksi AI di bawah 60%, sistem menolak memberikan rekomendasi skincare demi alasan keamanan data pengguna.
6. **Hasil Rekomendasi Detail**:
   - Persentase kecocokan/keyakinan AI (confidence score) dengan circular progress ring.
   - Penjelasan detail mengenai kondisi jenis kulit.
   - Kandungan skincare yang direkomendasikan.
   - Kandungan skincare yang wajib dihindari.
   - Contoh merek produk skincare yang beredar di pasaran.
   - Disclaimer medis edukatif.
7. **Tentang AI Page**: Halaman edukasi yang menguraikan teori Computer Vision, detail operasi lapisan CNN, dan visualisasi alur pemrosesan gambar wajah.
8. **Statistik Dataset Page**: Menyajikan visualisasi sebaran data pelatihan menggunakan diagram batang dinamis yang dibuat dengan **Matplotlib**.

---

## 📂 Struktur Proyek

```text
MeiSkinGlow/
│
├── app.py                  # Server Utama Flask (Inference, OpenCV Face Detection & Recommendation)
├── train_model.py          # Script Pembuat Dataset Tiruan & Pelatihan Model CNN Keras
├── requirements.txt        # Daftar Dependensi Python
├── README.md               # Dokumentasi Proyek
│
├── models/
│   └── skin_classifier_model.h5   # File Model CNN yang Sudah Terlatih (Hasil train_model.py)
│
├── dataset/                # Dataset Gambar Lokal per Kategori (Hasil train_model.py)
│   ├── acne/
│   ├── dry/
│   ├── normal/
│   ├── oily/
│   └── sensitive/
│
├── templates/              # File HTML Template (Flask Jinja2)
│   ├── index.html          # Landing Page
│   ├── analyze.html        # Halaman Input Kamera/Upload & Pemindaian
│   ├── result.html         # Halaman Output Hasil Klasifikasi & Skincare
│   ├── about_ai.html       # Halaman Edukasi Teori AI & Arsitektur CNN
│   └── dataset.html        # Halaman Visualisasi Distribusi Dataset
│
└── static/                 # Aset Statis
    ├── css/
    │   └── style.css       # Custom Glassmorphism Pink-Lavender Styling
    ├── js/
    │   └── main.js         # JavaScript Eksternal Tambahan (Cadangan)
    ├── uploads/            # Direktori Penyimpanan Foto Wajah Sementara
    └── images/
        └── dataset_chart.png  # Visualisasi Bar Chart Matplotlib (Hasil train_model.py)
```

---

## 🛠️ Cara Menjalankan Aplikasi

Ikuti langkah-langkah berikut secara berurutan untuk memasang dependensi, melatih model neural network, dan menjalankan server web lokal:

### 1. Pasang Dependensi Python
Pastikan Python 3.8+ sudah terpasang. Jalankan perintah berikut di terminal/command prompt Anda untuk menginstal semua dependensi yang dideklarasikan:
```bash
pip install -r requirements.txt
```

### 2. Buat Dataset & Latih Model CNN
Jalankan file `train_model.py` untuk menghasilkan 500 gambar skin tiruan lokal, menyimpan chart analisis sebaran dataset Matplotlib, dan melatih model CNN Anda:
```bash
python train_model.py
```
*Proses ini memakan waktu beberapa menit tergantung CPU/GPU Anda. Setelah selesai, model akan disimpan di `models/skin_classifier_model.h5`.*

### 3. Jalankan Aplikasi Web Flask
Setelah model berhasil disimpan, luncurkan server web utama dengan perintah:
```bash
python app.py
```

### 4. Akses Web
Buka browser favorit Anda dan kunjungi alamat lokal berikut:
```text
http://127.0.0.1:5000/
```

---

## 🧠 Penjelasan Model CNN (Convolutional Neural Network)

Model CNN yang dirancang menggunakan Keras Sequential API terdiri dari:
- **Rescaling Layer**: Mengubah piksel gambar dari rentang [0-255] ke [0-1] untuk mempercepat konvergensi gradien.
- **Convolutional Layer**: Memfilter visual gambar (Convolution 2D dengan kernel 3x3) guna memetakan fitur tepi, sudut, bercak jerawat, atau pantulan sebum.
- **Max Pooling Layer**: Melakukan downsampling spasial (2x2) untuk meminimalkan beban komputasi dan mencegah overfitting.
- **Fully Connected (Dense) Layer**: Menyatukan (Flatten) fitur multi-dimensi menjadi array satu dimensi, kemudian memetakan relasi ke dalam 128 neuron tersembunyi dengan fungsi aktivasi ReLU.
- **Dropout Layer**: Menghilangkan 50% neuron secara acak selama training untuk memaksa model belajar secara umum (anti-overfitting).
- **Softmax Output Layer**: Menghasilkan probabilitas dari 5 kelas target: `acne`, `dry`, `normal`, `oily`, dan `sensitive` dengan total probabilitas bernilai 1.
