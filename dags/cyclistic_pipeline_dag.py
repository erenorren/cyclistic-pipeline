from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "regisha_sheren",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def check_raw_csv(month_str: str) -> None:
    """Memastikan file CSV mentah bulan terkait tersedia sebelum diproses PySpark."""
    raw_path = f"/opt/airflow/data/raw/{month_str}-divvy-tripdata.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File mentah tidak ditemukan: {raw_path}")

    size_mb = os.path.getsize(raw_path) / (1024 * 1024)
    print(f"[OK] File ditemukan: {raw_path} ({size_mb:.2f} MB)")


with DAG(
    dag_id="cyclistic_medallion_pipeline",
    description="Pipeline ETL bulanan Cyclistic (April 2021 - Maret 2022) ke Silver Parquet & Gold DW",
    default_args=default_args,
    start_date=datetime(2021, 4, 1),
    end_date=datetime(2022, 3, 1),
    schedule="@monthly",
    catchup=True,
    max_active_runs=1,  # Wajib 1 agar tidak memproses banyak bulan bersamaan (aman untuk RAM 4 GB)
    tags=["cyclistic", "pyspark", "medallion"],
) as dag:

    # Task 1: Validasi keberadaan file mentah di Bronze/Raw
    validate_raw = PythonOperator(
        task_id="validate_raw_csv",
        python_callable=check_raw_csv,
        op_kwargs={"month_str": "{{ data_interval_start.strftime('%Y%m') }}"},
    )

    # Task 2: Cleaning & Feature Engineering menggunakan PySpark ke Silver Layer
    transform_silver = BashOperator(
        task_id="transform_to_silver",
        bash_command=(
            "python /opt/airflow/src/transform/clean_trips.py "
            "--month {{ data_interval_start.strftime('%Y%m') }}"
        ),
    )

    # Catatan: Task 3 (transform & load ke Gold Layer PostgreSQL) akan disambungkan di sini
    # setelah skema Star Schema final dikonfirmasi oleh PM.
    validate_raw >> transform_silver