import io
import os
import pandas as pd
import psycopg2


def get_pg_connection():
    """Membuka koneksi ke PostgreSQL Data Warehouse di dalam jaringan Docker."""
    return psycopg2.connect(
        host=os.getenv("DW_HOST", "postgres-warehouse"),
        port=int(os.getenv("DW_PORT_INTERNAL", "5432")),
        dbname=os.getenv("DW_DB", "cyclistic_dw"),
        user=os.getenv("DW_USER", "cyclistic_user"),
        password=os.getenv("DW_PASSWORD", "cyclistic_pass"),
    )


def assign_priority_tier(row: pd.Series) -> str:
    """Menjaga konsistensi aturan klasifikasi tier setelah penggabungan duplikat."""
    trips = row["total_casual_trips"]
    score = row["conversion_propensity_score"]
    if trips >= 30 and score >= 1.0:
        return "Prioritas Tinggi"
    if trips >= 30 and score > 0.4:
        return "Prioritas Sedang"
    return "Rendah"


def load_ml_targets(csv_path: str = "/opt/airflow/data/gold/ml_targets.csv") -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File output ML tidak ditemukan: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"start_station_id": str})

    # 1. Normalisasi ID stasiun & buang stasiun uji coba luar Chicago
    df["start_station_id_norm"] = (
        df["start_station_id"]
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"UNKNOWN": "NON_DOCKING"})
    )
    bad_station_regex = r"(?i)(?:WAREHOUSE|REPAIR|TEST|HQ QR|CHARGING STATION)"
    df = df[~df["start_station_id_norm"].str.contains(bad_station_regex, regex=True)].copy()

    # 2. Gabungkan 62 pasang baris ganda pada stasiun 331
    key_cols = ["start_station_id_norm", "day_name", "hour_key", "rideable_type"]
    if df.duplicated(subset=key_cols).any():
        df_grouped = df.groupby(key_cols, as_index=False).agg(
            total_casual_trips=("total_casual_trips", "sum"),
            predicted_member_like_trips=("predicted_member_like_trips", "sum"),
            priority_tier=("priority_tier", "first"),
        )
        df_grouped["conversion_propensity_score"] = (
            df_grouped["predicted_member_like_trips"] / df_grouped["total_casual_trips"]
        ).round(4)
        df_grouped["priority_tier"] = df_grouped.apply(assign_priority_tier, axis=1)
    else:
        df_grouped = df.copy()

    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            # 3. Ambil lookup surrogate key dari tabel dimensi PostgreSQL
            cur.execute("SELECT station_id, station_key FROM dim_stations;")
            station_map = dict(cur.fetchall())

            cur.execute("SELECT rideable_type, bike_type_key FROM dim_bike_type;")
            bike_map = dict(cur.fetchall())

            df_grouped["start_station_key"] = df_grouped["start_station_id_norm"].map(station_map)
            df_grouped["bike_type_key"] = df_grouped["rideable_type"].map(bike_map)

            unmapped = df_grouped[df_grouped["start_station_key"].isna()]["start_station_id_norm"].unique()
            if len(unmapped) > 0:
                raise ValueError(f"Terdapat station_id yang belum ada di dim_stations: {unmapped}")

            df_grouped["start_station_key"] = df_grouped["start_station_key"].astype(int)
            df_grouped["bike_type_key"] = df_grouped["bike_type_key"].astype(int)
            df_grouped["hour_key"] = df_grouped["hour_key"].astype(int)
            df_grouped["total_casual_trips"] = df_grouped["total_casual_trips"].astype(int)
            df_grouped["predicted_member_like_trips"] = df_grouped["predicted_member_like_trips"].astype(int)

            target_cols = [
                "start_station_key",
                "day_name",
                "hour_key",
                "bike_type_key",
                "total_casual_trips",
                "predicted_member_like_trips",
                "conversion_propensity_score",
                "priority_tier",
            ]
            df_final = df_grouped[target_cols]

            # 4. Muat ke tabel ml_conversion_targets
            cur.execute("TRUNCATE TABLE ml_conversion_targets RESTART IDENTITY;")

            csv_buffer = io.StringIO()
            df_final.to_csv(csv_buffer, index=False, header=False)
            csv_buffer.seek(0)

            copy_sql = f"""
                COPY ml_conversion_targets ({', '.join(target_cols)})
                FROM STDIN WITH (FORMAT CSV);
            """
            cur.copy_expert(copy_sql, csv_buffer)

        conn.commit()

        print("=" * 60)
        print("AUDIT TABEL ML_CONVERSION_TARGETS - SELESAI")
        print("=" * 60)
        print(f"Total Baris Dimuat             : {len(df_final):,}")
        print(f"Total Perjalanan Casual        : {int(df_final['total_casual_trips'].sum()):,}")
        print(f"Prediksi Mirip Member          : {int(df_final['predicted_member_like_trips'].sum()):,}")
        print(f"Distribusi Priority Tier       : {df_final['priority_tier'].value_counts().to_dict()}")
        print("=" * 60)

    finally:
        conn.close()


if __name__ == "__main__":
    load_ml_targets()