import argparse
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    TimestampType,
    DoubleType,
)


def create_spark_session(app_name: str) -> SparkSession:
    """Inisialisasi SparkSession yang dioptimalkan untuk Docker (RAM 4 GB)."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.session.timeZone", "America/Chicago")
        .getOrCreate()
    )


def get_raw_schema() -> StructType:
    """Definisi skema eksplisit 13 kolom CSV Cyclistic agar tipe ID stasiun konsisten."""
    return StructType(
        [
            StructField("ride_id", StringType(), False),
            StructField("rideable_type", StringType(), True),
            StructField("started_at", TimestampType(), True),
            StructField("ended_at", TimestampType(), True),
            StructField("start_station_name", StringType(), True),
            StructField("start_station_id", StringType(), True),
            StructField("end_station_name", StringType(), True),
            StructField("end_station_id", StringType(), True),
            StructField("start_lat", DoubleType(), True),
            StructField("start_lng", DoubleType(), True),
            StructField("end_lat", DoubleType(), True),
            StructField("end_lng", DoubleType(), True),
            StructField("member_casual", StringType(), True),
        ]
    )


def clean_and_transform(month_str: str, base_data_dir: str = "/opt/airflow/data") -> None:
    """
    Membersihkan 1 file CSV bulanan dari layer Raw dan menyimpannya ke layer Silver
    dalam format Parquet terpartisi berdasarkan kolom `tahun_bulan`.
    """
    raw_file = os.path.join(base_data_dir, "raw", f"{month_str}-divvy-tripdata.csv")
    silver_dir = os.path.join(base_data_dir, "silver")

    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"File mentah tidak ditemukan: {raw_file}")

    spark = create_spark_session(f"Cyclistic_Silver_Cleaning_{month_str}")
    spark.sparkContext.setLogLevel("WARN")

    try:
        # 1. Baca data mentah dengan skema eksplisit
        df_raw = (
            spark.read.option("header", "true")
            .schema(get_raw_schema())
            .csv(raw_file)
        )

        raw_count = df_raw.count()

        # 2. Bersihkan spasi berlebih pada kolom teks utama
        df_trimmed = (
            df_raw.withColumn("ride_id", F.trim(F.col("ride_id")))
            .withColumn("rideable_type", F.trim(F.col("rideable_type")))
            .withColumn("start_station_name", F.trim(F.col("start_station_name")))
            .withColumn("start_station_id", F.trim(F.col("start_station_id")))
            .withColumn("end_station_name", F.trim(F.col("end_station_name")))
            .withColumn("end_station_id", F.trim(F.col("end_station_id")))
            .withColumn("member_casual", F.trim(F.col("member_casual")))
        )

        # Pastikan kolom wajib tidak null & buang duplikat ride_id jika ada
        df_valid_core = df_trimmed.dropna(
            subset=["ride_id", "rideable_type", "started_at", "ended_at", "member_casual"]
        ).dropDuplicates(["ride_id"])

        # 3. Buang anomali durasi negatif (ended_at < started_at)
        df_non_negative = df_valid_core.filter(F.col("ended_at") >= F.col("started_at"))
        after_neg_count = df_non_negative.count()
        dropped_negative = raw_count - after_neg_count

        # 4. Buang perjalanan operasional gudang & stasiun perbaikan
        operational_stations = [
            "BASE - 2132 W HUBBARD WAREHOUSE",
            "DIVVY CASSETTE REPAIR MOBILE STATION",
        ]
        df_customer_trips = df_non_negative.filter(
            ~(
                F.upper(F.coalesce(F.col("start_station_name"), F.lit(""))).isin(
                    operational_stations
                )
                | F.upper(F.coalesce(F.col("end_station_name"), F.lit(""))).isin(
                    operational_stations
                )
            )
        )
        clean_count = df_customer_trips.count()
        dropped_operational = after_neg_count - clean_count

        # 5. Hitung durasi perjalanan (menit) & flag outlier (< 1 menit / > 24 jam)
        df_duration = df_customer_trips.withColumn(
            "trip_duration_minutes",
            F.round(
                (F.unix_timestamp(F.col("ended_at")) - F.unix_timestamp(F.col("started_at")))
                / 60.0,
                2,
            ),
        ).withColumn(
            "is_false_start", F.col("trip_duration_minutes") < 1.0
        ).withColumn(
            "is_outlier_24h", F.col("trip_duration_minutes") > 1440.0
        )

        # 6. Tangani stasiun kosong (umum pada electric_bike dockless)
        df_stations = (
            df_duration.withColumn(
                "is_station_missing",
                F.col("start_station_name").isNull()
                | (F.col("start_station_name") == "")
                | F.col("end_station_name").isNull()
                | (F.col("end_station_name") == ""),
            )
            .withColumn(
                "start_station_name",
                F.when(
                    F.col("start_station_name").isNull() | (F.col("start_station_name") == ""),
                    F.lit("Non-Docking / Unknown"),
                ).otherwise(F.col("start_station_name")),
            )
            .withColumn(
                "end_station_name",
                F.when(
                    F.col("end_station_name").isNull() | (F.col("end_station_name") == ""),
                    F.lit("Non-Docking / Unknown"),
                ).otherwise(F.col("end_station_name")),
            )
            .withColumn(
                "start_station_id",
                F.when(
                    F.col("start_station_id").isNull() | (F.col("start_station_id") == ""),
                    F.lit("UNKNOWN"),
                ).otherwise(F.col("start_station_id")),
            )
            .withColumn(
                "end_station_id",
                F.when(
                    F.col("end_station_id").isNull() | (F.col("end_station_id") == ""),
                    F.lit("UNKNOWN"),
                ).otherwise(F.col("end_station_id")),
            )
        )

        # 7. Hitung jarak lurus Haversine (km) dengan proteksi koordinat NULL
        has_valid_coords = (
            F.col("start_lat").isNotNull()
            & F.col("start_lng").isNotNull()
            & F.col("end_lat").isNotNull()
            & F.col("end_lng").isNotNull()
        )

        dlat = F.radians(F.col("end_lat") - F.col("start_lat"))
        dlon = F.radians(F.col("end_lng") - F.col("start_lng"))
        lat1 = F.radians(F.col("start_lat"))
        lat2 = F.radians(F.col("end_lat"))

        a_raw = F.pow(F.sin(dlat / 2.0), 2) + F.cos(lat1) * F.cos(lat2) * F.pow(
            F.sin(dlon / 2.0), 2
        )
        # Batasi nilai a pada [0, 1] untuk mencegah error presisi floating-point pada sqrt
        a_clamped = F.greatest(F.lit(0.0), F.least(F.lit(1.0), a_raw))
        c = 2.0 * F.atan2(F.sqrt(a_clamped), F.sqrt(1.0 - a_clamped))
        haversine_km = F.round(F.lit(6371.0) * c, 3)

        df_geo = df_stations.withColumn(
            "displacement_km",
            F.when(has_valid_coords, haversine_km).otherwise(F.lit(None).cast("double")),
        ).withColumn(
            "is_round_trip",
            (F.col("start_station_id") != "UNKNOWN")
            & (F.col("start_station_id") == F.col("end_station_id")),
        )

        # 8. Ekstraksi fitur temporal & kolom partisi tahun_bulan (YYYY-MM)
        # Spark dayofweek: 1 = Minggu (Sunday), 7 = Sabtu (Saturday)
        df_silver = (
            df_geo.withColumn("tahun_bulan", F.date_format(F.col("started_at"), "yyyy-MM"))
            .withColumn("ride_date", F.to_date(F.col("started_at")))
            .withColumn("day_of_week", F.dayofweek(F.col("started_at")))
            .withColumn("day_name", F.date_format(F.col("started_at"), "EEEE"))
            .withColumn("start_hour", F.hour(F.col("started_at")))
            .withColumn("is_weekend", F.col("day_of_week").isin([1, 7]))
            .withColumn(
                "is_rush_hour",
                (~F.col("is_weekend"))
                & (F.col("start_hour").between(7, 9) | F.col("start_hour").between(16, 18)),
            )
        )

        # 9. Simpan ke layer Silver (Parquet terpartisi per tahun_bulan)
        (
            df_silver.write.mode("overwrite")
            .partitionBy("tahun_bulan")
            .parquet(silver_dir)
        )

        # 10. Cetak ringkasan audit kualitas data untuk dokumentasi laporan
        stats = df_silver.agg(
            F.sum(F.when(F.col("is_false_start"), 1).otherwise(0)).alias("false_starts"),
            F.sum(F.when(F.col("is_outlier_24h"), 1).otherwise(0)).alias("outliers_24h"),
            F.sum(F.when(F.col("is_station_missing"), 1).otherwise(0)).alias("missing_stations"),
            F.sum(F.when(F.col("displacement_km").isNull(), 1).otherwise(0)).alias("missing_coords"),
        ).collect()[0]

        print("=" * 60)
        print(f"AUDIT SILVER LAYER - PERIODE {month_str}")
        print("=" * 60)
        print(f"Total Baris Mentah (Raw)         : {raw_count:,}")
        print(f"Dibuang - Durasi Negatif         : {dropped_negative:,}")
        print(f"Dibuang - Gudang/Repair Station  : {dropped_operational:,}")
        print(f"Total Baris Bersih (Silver)      : {clean_count:,}")
        print("-" * 60)
        print(f"Flagged - False Start (< 1 mnt)  : {stats['false_starts']:,}")
        print(f"Flagged - Outlier (> 24 jam)     : {stats['outliers_24h']:,}")
        print(f"Flagged - Stasiun Kosong         : {stats['missing_stations']:,}")
        print(f"Flagged - Koordinat Akhir Kosong : {stats['missing_coords']:,}")
        print(f"Output Tersimpan di              : {silver_dir}/tahun_bulan={month_str[:4]}-{month_str[4:]}")
        print("=" * 60)

    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PySpark Silver Layer Cleaning untuk Cyclistic")
    parser.add_argument(
        "--month",
        required=True,
        help="Periode bulan dalam format YYYYMM (contoh: 202104)",
    )
    args = parser.parse_args()

    if len(args.month) != 6 or not args.month.isdigit():
        print("Error: Parameter --month harus berformat YYYYMM (contoh: 202104)")
        sys.exit(1)

    clean_and_transform(args.month)