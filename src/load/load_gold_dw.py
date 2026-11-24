import argparse
from datetime import date, timedelta
import io
import os
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def get_pg_connection():
    """Membuka koneksi ke PostgreSQL Data Warehouse di dalam jaringan Docker."""
    return psycopg2.connect(
        host=os.getenv("DW_HOST", "postgres-warehouse"),
        port=int(os.getenv("DW_PORT_INTERNAL", "5432")),
        dbname=os.getenv("DW_DB", "cyclistic_dw"),
        user=os.getenv("DW_USER", "cyclistic_user"),
        password=os.getenv("DW_PASSWORD", "cyclistic_pass"),
    )


def create_spark_session(app_name: str) -> SparkSession:
    """Inisialisasi SparkSession yang dioptimalkan untuk Docker (RAM 4 GB)."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.session.timeZone", "America/Chicago")
        .getOrCreate()
    )


def init_schema_and_static_dims(conn, ddl_path: str) -> None:
    """Menjalankan DDL tabel dan mengisi dimensi statis (date, time, bike, user, NON_DOCKING)."""
    with open(ddl_path, "r", encoding="utf-8") as f:
        ddl_sql = f.read()

    with conn.cursor() as cur:
        cur.execute(ddl_sql)

        # 1. Seed dim_date (2021-04-01 s.d. 2022-03-31)
        start_dt = date(2021, 4, 1)
        end_dt = date(2022, 3, 31)
        date_rows = []
        curr = start_dt
        while curr <= end_dt:
            date_key = int(curr.strftime("%Y%m%d"))
            month_num = curr.month
            if month_num in (3, 4, 5):
                season = "Spring"
            elif month_num in (6, 7, 8):
                season = "Summer"
            elif month_num in (9, 10, 11):
                season = "Fall"
            else:
                season = "Winter"

            iso_dow = curr.isoweekday()  # 1=Senin s.d. 7=Minggu
            date_rows.append(
                (
                    date_key,
                    curr,
                    curr.year,
                    month_num,
                    curr.strftime("%B"),
                    curr.strftime("%Y-%m"),
                    iso_dow,
                    curr.strftime("%A"),
                    iso_dow in (6, 7),
                    season,
                )
            )
            curr += timedelta(days=1)

        execute_values(
            cur,
            """
            INSERT INTO dim_date (
                date_key, full_date, year, month_num, month_name,
                year_month, day_of_week_num, day_name, is_weekend, season
            ) VALUES %s
            ON CONFLICT (date_key) DO NOTHING;
            """,
            date_rows,
        )

        # 2. Seed dim_time (0..23)
        time_rows = []
        for h in range(24):
            if 6 <= h <= 10:
                period = "Pagi (06-10)"
            elif 11 <= h <= 14:
                period = "Siang (11-14)"
            elif 15 <= h <= 19:
                period = "Sore (15-19)"
            else:
                period = "Malam (20-05)"

            is_rush = (7 <= h <= 9) or (16 <= h <= 18)
            time_rows.append((h, period, is_rush))

        execute_values(
            cur,
            """
            INSERT INTO dim_time (hour_key, time_period, is_rush_hour)
            VALUES %s
            ON CONFLICT (hour_key) DO NOTHING;
            """,
            time_rows,
        )

        # 3. Seed dim_bike_type
        bike_rows = [
            (1, "classic_bike", "Sepeda Klasik"),
            (2, "electric_bike", "Sepeda Listrik"),
            (3, "docked_bike", "Sepeda Dock"),
        ]
        execute_values(
            cur,
            """
            INSERT INTO dim_bike_type (bike_type_key, rideable_type, bike_category_label)
            VALUES %s
            ON CONFLICT (bike_type_key) DO NOTHING;
            """,
            bike_rows,
        )

        # 4. Seed dim_user_type
        user_rows = [
            (1, "casual", "Pengguna Kasual"),
            (2, "member", "Anggota Tahunan"),
        ]
        execute_values(
            cur,
            """
            INSERT INTO dim_user_type (user_type_key, member_casual, user_category_label)
            VALUES %s
            ON CONFLICT (user_type_key) DO NOTHING;
            """,
            user_rows,
        )

        # 5. Seed baris khusus NON_DOCKING di dim_stations (lat/lng NULL agar tidak muncul titik palsu di peta)
        cur.execute(
            """
            INSERT INTO dim_stations (station_id, station_name, latitude, longitude, is_station_missing)
            VALUES ('NON_DOCKING', 'Non-Docking / Unknown', NULL, NULL, TRUE)
            ON CONFLICT (station_id) DO NOTHING;
            """
        )

    conn.commit()


def load_month_to_gold(month_str: str, base_data_dir: str = "/opt/airflow/data") -> None:
    """Membaca partisi Silver bulanan, membentuk agregat Gold, dan memuat ke PostgreSQL."""
    year_month = f"{month_str[:4]}-{month_str[4:]}"
    silver_partition = os.path.join(base_data_dir, "silver", f"tahun_bulan={year_month}")
    gold_fact_dir = os.path.join(base_data_dir, "gold", "fact_trips_agg")
    ddl_path = "/opt/airflow/src/load/schema_ddl.sql"

    if not os.path.exists(silver_partition):
        raise FileNotFoundError(f"Partisi Silver tidak ditemukan: {silver_partition}")

    conn = get_pg_connection()
    spark = create_spark_session(f"Cyclistic_Gold_Load_{month_str}")
    spark.sparkContext.setLogLevel("WARN")

    try:
        # 1. Pastikan struktur tabel & dimensi statis tersedia
        init_schema_and_static_dims(conn, ddl_path)

        # 2. Baca partisi Silver bulan terkait & bersihkan ID '.0', stasiun uji coba, serta koordinat luar Chicago
        df_silver = spark.read.parquet(silver_partition)
        bad_station_regex = r"(?i)(WAREHOUSE|REPAIR|TEST|HQ QR|CHARGING STATION)"

        df_clean = (
            df_silver.filter(
                ~(
                    F.col("start_station_name").rlike(bad_station_regex)
                    | F.col("end_station_name").rlike(bad_station_regex)
                )
            )
            .filter(
                (F.col("start_lat").isNull() | F.col("start_lat").between(41.5, 42.2))
                & (F.col("start_lng").isNull() | F.col("start_lng").between(-88.1, -87.4))
                & (F.col("end_lat").isNull() | F.col("end_lat").between(41.5, 42.2))
                & (F.col("end_lng").isNull() | F.col("end_lng").between(-88.1, -87.4))
            )
            .withColumn(
                "start_station_id_clean",
                F.regexp_replace(F.trim(F.col("start_station_id")), r"\.0$", ""),
            )
            .withColumn(
                "end_station_id_clean",
                F.regexp_replace(F.trim(F.col("end_station_id")), r"\.0$", ""),
            )
            .withColumn(
                "start_station_id_norm",
                F.when(
                    F.col("start_station_id_clean") == "UNKNOWN", F.lit("NON_DOCKING")
                ).otherwise(F.col("start_station_id_clean")),
            )
            .withColumn(
                "end_station_id_norm",
                F.when(
                    F.col("end_station_id_clean") == "UNKNOWN", F.lit("NON_DOCKING")
                ).otherwise(F.col("end_station_id_clean")),
            )
        )
        
        # 3. Ekstrak stasiun fisik unik dari bulan ini & upsert ke dim_stations
        start_st = df_clean.filter(F.col("start_station_id_norm") != "NON_DOCKING").select(
            F.col("start_station_id_norm").alias("station_id"),
            F.col("start_station_name").alias("station_name"),
            F.col("start_lat").alias("lat"),
            F.col("start_lng").alias("lng"),
        )
        end_st = df_clean.filter(F.col("end_station_id_norm") != "NON_DOCKING").select(
            F.col("end_station_id_norm").alias("station_id"),
            F.col("end_station_name").alias("station_name"),
            F.col("end_lat").alias("lat"),
            F.col("end_lng").alias("lng"),
        )

        all_st = start_st.unionByName(end_st)

        # Pilih nama stasiun paling dominan per station_id untuk mencegah duplikasi
        name_counts = all_st.groupBy("station_id", "station_name").count()
        w_name = Window.partitionBy("station_id").orderBy(
            F.desc("count"), F.asc("station_name")
        )
        dominant_names = (
            name_counts.withColumn("rn", F.row_number().over(w_name))
            .filter(F.col("rn") == 1)
            .select("station_id", "station_name")
        )

        station_coords = all_st.groupBy("station_id").agg(
            F.round(F.avg("lat"), 6).alias("latitude"),
            F.round(F.avg("lng"), 6).alias("longitude"),
        )

        df_stations_month = (
            dominant_names.join(station_coords, on="station_id", how="inner")
            .withColumn("is_station_missing", F.lit(False))
            .toPandas()
        )

        if not df_stations_month.empty:
            station_records = [
                (
                    row["station_id"],
                    row["station_name"],
                    None if pd.isna(row["latitude"]) else float(row["latitude"]),
                    None if pd.isna(row["longitude"]) else float(row["longitude"]),
                    False,
                )
                for _, row in df_stations_month.iterrows()
            ]
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO dim_stations (
                        station_id, station_name, latitude, longitude, is_station_missing
                    ) VALUES %s
                    ON CONFLICT (station_id) DO NOTHING;
                    """,
                    station_records,
                )
            conn.commit()

        # 4. Ambil lookup key dari PostgreSQL agar surrogate key 100% sinkron
        with conn.cursor() as cur:
            cur.execute("SELECT station_id, station_key FROM dim_stations;")
            station_map = dict(cur.fetchall())

            cur.execute("SELECT rideable_type, bike_type_key FROM dim_bike_type;")
            bike_map = dict(cur.fetchall())

            cur.execute("SELECT member_casual, user_type_key FROM dim_user_type;")
            user_map = dict(cur.fetchall())

        # 5. Agregasi fact_trips_agg sesuai granularitas Star Schema
        df_prepped = (
            df_clean.withColumn(
                "date_key", F.date_format(F.col("ride_date"), "yyyyMMdd").cast("int")
            )
            .withColumn("hour_key", F.col("start_hour").cast("int"))
            .withColumn("tahun_bulan", F.lit(year_month))
        )

        df_agg = df_prepped.groupBy(
            "tahun_bulan",
            "date_key",
            "hour_key",
            "start_station_id_norm",
            "end_station_id_norm",
            "rideable_type",
            "member_casual",
            "is_round_trip",
            "is_false_start",
            "is_outlier_24h",
        ).agg(
            F.count("ride_id").cast("int").alias("total_trips"),
            F.count("displacement_km").cast("int").alias("valid_distance_trips"),
            F.round(F.sum("trip_duration_minutes"), 2).alias("total_duration_minutes"),
            F.round(F.avg("trip_duration_minutes"), 2).alias("avg_duration_minutes"),
            F.round(F.coalesce(F.sum("displacement_km"), F.lit(0.0)), 3).alias(
                "total_distance_km"
            ),
            F.round(F.avg("displacement_km"), 3).alias("avg_distance_km"),
        )

        # Simpan salinan Gold Parquet terpartisi di data/gold/
        (
            df_agg.write.mode("overwrite")
            .partitionBy("tahun_bulan")
            .parquet(gold_fact_dir)
        )

        # Konversi ke Pandas untuk pemetaan Foreign Key & streaming COPY ke PostgreSQL
        pdf_fact = df_agg.drop("tahun_bulan").toPandas()
        pdf_fact["start_station_key"] = pdf_fact["start_station_id_norm"].map(station_map)
        pdf_fact["end_station_key"] = pdf_fact["end_station_id_norm"].map(station_map)
        pdf_fact["bike_type_key"] = pdf_fact["rideable_type"].map(bike_map)
        pdf_fact["user_type_key"] = pdf_fact["member_casual"].map(user_map)

        target_cols = [
            "date_key",
            "hour_key",
            "start_station_key",
            "end_station_key",
            "bike_type_key",
            "user_type_key",
            "is_round_trip",
            "is_false_start",
            "is_outlier_24h",
            "total_trips",
            "valid_distance_trips",
            "total_duration_minutes",
            "avg_duration_minutes",
            "total_distance_km",
            "avg_distance_km",
        ]
        pdf_final = pdf_fact[target_cols]

        # 6. Muat ke PostgreSQL secara idempoten (hapus data bulan sama jika di-rerun)
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM fact_trips_agg
                WHERE date_key IN (
                    SELECT date_key FROM dim_date WHERE year_month = %s
                );
                """,
                (year_month,),
            )

            csv_buffer = io.StringIO()
            pdf_final.to_csv(csv_buffer, index=False, header=False, na_rep="\\N")
            csv_buffer.seek(0)

            copy_sql = f"""
                COPY fact_trips_agg ({', '.join(target_cols)})
                FROM STDIN WITH (FORMAT CSV, NULL '\\N');
            """
            cur.copy_expert(copy_sql, csv_buffer)

            cur.execute("SELECT COUNT(*) FROM dim_stations;")
            total_stations_db = cur.fetchone()[0]

        conn.commit()

        raw_trips_sum = int(pdf_final["total_trips"].sum())
        valid_dist_sum = int(pdf_final["valid_distance_trips"].sum())
        agg_rows_count = len(pdf_final)

        print("=" * 60)
        print(f"AUDIT GOLD LAYER & POSTGRESQL - PERIODE {month_str}")
        print("=" * 60)
        print(f"Baris Agregat Dimuat (fact_trips_agg) : {agg_rows_count:,}")
        print(f"Total Perjalanan Terwakili           : {raw_trips_sum:,}")
        print(f"Perjalanan dengan Jarak Valid        : {valid_dist_sum:,}")
        print(f"Total Stasiun Terdaftar (dim_stations): {total_stations_db:,}")
        print("=" * 60)

    finally:
        spark.stop()
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Gold Star Schema ke PostgreSQL")
    parser.add_argument(
        "--month",
        required=True,
        help="Periode bulan dalam format YYYYMM (contoh: 202104)",
    )
    args = parser.parse_args()

    if len(args.month) != 6 or not args.month.isdigit():
        print("Error: Parameter --month harus berformat YYYYMM (contoh: 202104)")
        sys.exit(1)

    load_month_to_gold(args.month)