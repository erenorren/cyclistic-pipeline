# 🚲 Cyclistic Bike-Share Data Engineering & Analytics Pipeline

## Deskripsi Proyek
Proyek ini merupakan **Data Engineering & Machine Learning Pipeline** *end-to-end* untuk menganalisis pola perjalanan serta perilaku pengguna layanan berbagi sepeda (*bike-sharing*) **Cyclistic / Divvy Chicago** (periode April 2021 – Maret 2022, mencakup **5.723.532 baris** transaksi).

Pipeline dirancang mengikuti arsitektur **Medallion (Bronze -> Silver -> Gold)**, diorkestrasi secara otomatis menggunakan **Apache Airflow**, diproses secara terdistribusi menggunakan **Apache Spark (PySpark)**, dan bermuara pada **PostgreSQL** (`cyclistic_dw`) untuk divisualisasikan menggunakan **Power BI Desktop**.

---

## 🏗️ Arsitektur Data
- **Ingestion (Bronze Layer):** Pembacaan dan validasi skema 12 file CSV bulanan mentah (`YYYYMM-divvy-tripdata.csv`) dari zona *Raw* tanpa mengubah data sumber asli.
- **Transformation (Silver Layer):** Standardisasi skema ID stasiun (`string`), pembersihan anomali (durasi negatif & perjalanan operasional gudang/perawatan), penandaan *outlier* (`< 1 menit` dan `> 24 jam`), serta *feature engineering* (durasi perjalanan, jarak lurus *Haversine displacement*, penanda akhir pekan, dan jam sibuk) menggunakan **PySpark** dalam format Parquet terpartisi bulanan (`tahun_bulan`).
- **Load & Serving (Gold Layer):** Pembentukan skema *Star Schema* (tabel fakta teragregasi dan tabel dimensi stasiun, waktu, jenis sepeda, serta tipe pengguna) yang dimuat ke **PostgreSQL** agar ringan dibaca oleh dasbor BI.
- **Orchestration:** **Apache Airflow** digunakan untuk menjadwalkan pemrosesan *batch* bulanan secara berurutan (`max_active_runs=1`) dengan fitur `catchup=True` yang dibatasi pada rentang April 2021 – Maret 2022.
- **Analytics & Machine Learning:** Pengujian hipotesis statistik serta pemodelan klasifikasi (`scikit-learn`) untuk mengidentifikasi ciri perjalanan pengguna kasual (*casual*) yang berpola mirip pengguna komuter (*member*) sebagai dasar rekomendasi kampanye pemasaran.

---

## 📂 Struktur Repositori
- `dags/` : Definisi alur kerja (*DAG*) pemrosesan *batch* bulanan Apache Airflow.
- `src/` : Kumpulan skrip ETL PySpark dan Python (`ingest/`, `transform/`, `load/`).
- `logs/` : Log eksekusi otomatis dari Apache Airflow. **(TIDAK DI-COMMIT)**.
- `data/` : Folder lokal pengganti *Data Lake* (`raw/`, `silver/`, `gold/`). **(TIDAK DI-COMMIT)**.

---

## Tim Pengembang
1. Kharizma Rizkiah (H1101241050) - Project Lead, Data Architect, & Marketing Analyst
2. Regisha Sheren (H1101241036) - Data Engineer (Ingestion, PySpark Pipeline, Airflow Orchestration, Data Warehouse)
3. Febrianti Khumairoh (H1101241006) - Data Scientist & BI Analyst (EDA, Statistical Testing, Machine Learning, Dashboard)
