-- ============================================================================
-- DDL Star Schema Data Warehouse: Cyclistic Bike-Share (cyclistic_dw)
-- ============================================================================

-- 1. Dimensi Tanggal & Musim
CREATE TABLE IF NOT EXISTS dim_date (
    date_key INT PRIMARY KEY,
    full_date DATE NOT NULL UNIQUE,
    year INT NOT NULL,
    month_num INT NOT NULL,
    month_name VARCHAR(15) NOT NULL,
    year_month VARCHAR(7) NOT NULL,
    day_of_week_num INT NOT NULL,
    day_name VARCHAR(15) NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    season VARCHAR(15) NOT NULL
);

-- 2. Dimensi Jam & Waktu Sibuk
CREATE TABLE IF NOT EXISTS dim_time (
    hour_key INT PRIMARY KEY,
    time_period VARCHAR(25) NOT NULL,
    is_rush_hour BOOLEAN NOT NULL
);

-- 3. Dimensi Stasiun & Koordinat Peta
CREATE TABLE IF NOT EXISTS dim_stations (
    station_key SERIAL PRIMARY KEY,
    station_id VARCHAR(50) NOT NULL UNIQUE,
    station_name VARCHAR(150) NOT NULL,
    latitude DOUBLE PRECISION NULL,
    longitude DOUBLE PRECISION NULL,
    is_station_missing BOOLEAN NOT NULL
);

-- 4. Dimensi Armada Sepeda
CREATE TABLE IF NOT EXISTS dim_bike_type (
    bike_type_key INT PRIMARY KEY,
    rideable_type VARCHAR(30) NOT NULL UNIQUE,
    bike_category_label VARCHAR(30) NOT NULL
);

-- 5. Dimensi Tipe Pengguna
CREATE TABLE IF NOT EXISTS dim_user_type (
    user_type_key INT PRIMARY KEY,
    member_casual VARCHAR(20) NOT NULL UNIQUE,
    user_category_label VARCHAR(30) NOT NULL
);

-- 6. Tabel Fakta Utama (Perjalanan Teragregasi)
CREATE TABLE IF NOT EXISTS fact_trips_agg (
    fact_id BIGSERIAL PRIMARY KEY,
    date_key INT NOT NULL REFERENCES dim_date(date_key),
    hour_key INT NOT NULL REFERENCES dim_time(hour_key),
    start_station_key INT NOT NULL REFERENCES dim_stations(station_key),
    end_station_key INT NOT NULL REFERENCES dim_stations(station_key),
    bike_type_key INT NOT NULL REFERENCES dim_bike_type(bike_type_key),
    user_type_key INT NOT NULL REFERENCES dim_user_type(user_type_key),
    is_round_trip BOOLEAN NOT NULL,
    is_false_start BOOLEAN NOT NULL,
    is_outlier_24h BOOLEAN NOT NULL,
    total_trips INT NOT NULL,
    valid_distance_trips INT NOT NULL,
    total_duration_minutes DOUBLE PRECISION NOT NULL,
    avg_duration_minutes DOUBLE PRECISION NOT NULL,
    total_distance_km DOUBLE PRECISION NOT NULL,
    avg_distance_km DOUBLE PRECISION NULL
);

-- 7. Tabel Output Machine Learning (Target Konversi Kampanye)
CREATE TABLE IF NOT EXISTS ml_conversion_targets (
    target_id SERIAL PRIMARY KEY,
    start_station_key INT NOT NULL REFERENCES dim_stations(station_key),
    day_name VARCHAR(15) NOT NULL,
    hour_key INT NOT NULL REFERENCES dim_time(hour_key),
    bike_type_key INT NOT NULL REFERENCES dim_bike_type(bike_type_key),
    total_casual_trips INT NOT NULL,
    predicted_member_like_trips INT NOT NULL,
    conversion_propensity_score DOUBLE PRECISION NOT NULL,
    priority_tier VARCHAR(20) NOT NULL
);

-- Indeks untuk mempercepat query filter & relasi di Power BI
CREATE INDEX IF NOT EXISTS idx_fact_date_key ON fact_trips_agg(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_hour_key ON fact_trips_agg(hour_key);
CREATE INDEX IF NOT EXISTS idx_fact_start_station ON fact_trips_agg(start_station_key);
CREATE INDEX IF NOT EXISTS idx_fact_end_station ON fact_trips_agg(end_station_key);
CREATE INDEX IF NOT EXISTS idx_fact_user_bike ON fact_trips_agg(user_type_key, bike_type_key);