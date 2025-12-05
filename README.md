# Sistem Deteksi Cacat Kain Otomatis (Fabric Defect Detection)

Proyek ini adalah implementasi sistem Computer Vision untuk mendeteksi cacat pada kain tenun menggunakan analisis tekstur dan pola geometris.

## 🚀 Fitur Utama

1.  **Analisis Tekstur (GLCM):** Mendeteksi cacat tekstur kasar seperti *Neps* (bintik) dan *Slubs*.
2.  **Analisis Densitas (Background Subtraction):** Mendeteksi cacat *Wadding* (gumpalan) yang memiliki perbedaan intensitas warna.
3.  **Analisis Pola (Hough Transform):** Mendeteksi cacat struktur anyaman (*Skewing* / Pola Miring).
4.  **Smart GUI:** Antarmuka responsif yang otomatis menyesuaikan ukuran gambar (*Smart Fit*) tanpa distorsi.

## 🛠️ Teknologi yang Digunakan

* **Bahasa:** Python 3.13.8
* **GUI:** Tkinter
* **Computer Vision:** OpenCV (`cv2`), Scikit-Image (`skimage`)
* **Matematika:** NumPy

## 📦 Cara Menjalankan

1.  **Clone repository ini:**
    ```bash
    git clone [https://github.com/UsernameAnda/Fabric-Defect-Detection.git](https://github.com/SteavenJ/Fabric-Defect-Detection.git)
    cd Fabric-Defect-Detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan aplikasi:**
    ```bash
    python main.py
    ```

## 📊 Metodologi

Sistem bekerja dengan tiga tahapan filter:
1.  **Preprocessing:** Normalisasi citra dan koreksi pencahayaan.
2.  **Sliding Window:** Memindai citra per blok (64x64 piksel) untuk mencari anomali lokal.
3.  **Decision Fusion:** Menggabungkan hasil analisis tekstur dan geometri untuk keputusan akhir (Diterima/Ditolak).
