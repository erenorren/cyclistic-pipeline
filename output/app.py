from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cyclistic Marketing & Target Dashboard",
    layout="wide",
)

SILVER_PATH = Path("data/silver")
ML_TARGETS_PATH = Path("output/ml_targets.csv")
CHICAGO_BOUNDS = {
    "lat": (41.5, 42.2),
    "lng": (-88.1, -87.4),
}

COLUMN_LABELS = {
    "member_casual": "Tipe Pengguna",
    "rideable_type": "Jenis Sepeda",
    "month_label": "Bulan",
    "day_name": "Hari",
    "hour_key": "Jam",
    "start_station_id": "ID Stasiun Awal",
    "start_station_name": "Nama Stasiun Awal",
    "total_trips": "Total Perjalanan",
    "casual_trips": "Perjalanan Casual",
    "member_trips": "Perjalanan Member",
    "avg_duration_min": "Rata-Rata Durasi (Menit)",
    "avg_distance_km": "Rata-Rata Jarak (Km)",
    "total_casual_trips": "Total Casual Trips",
    "predicted_member_like_trips": "Predicted Member-Like Trips",
    "conversion_propensity_score": "Skor Propensi Konversi",
    "priority_tier": "Tingkat Prioritas",
}


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        col: COLUMN_LABELS.get(col, col.replace("_", " ").title())
        for col in df.columns
    }
    return df.rename(columns=renamed)


def assign_priority_tier(row: pd.Series) -> str:
    trips = row.get("total_casual_trips", 0)
    score = row.get("conversion_propensity_score", 0.0)
    if pd.notna(trips) and pd.notna(score) and trips >= 30 and score >= 1.0:
        return "Prioritas Tinggi"
    if pd.notna(trips) and pd.notna(score) and trips >= 30 and score > 0.4:
        return "Prioritas Sedang"
    return "Rendah"


@st.cache_data
def load_silver_data(silver_path: Path) -> pd.DataFrame:
    if not silver_path.exists():
        return pd.DataFrame()

    if silver_path.is_dir():
        files = sorted(silver_path.rglob("*.parquet"))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df = pd.read_parquet(silver_path)

    for col in ("is_false_start", "is_outlier_24h"):
        if col in df.columns:
            df = df[~df[col].fillna(False).astype(bool)].copy()

    if "rideable_type" in df.columns:
        df = df[df["rideable_type"] != "docked_bike"].copy()

    if "started_at" in df.columns:
        started = pd.to_datetime(df["started_at"], errors="coerce")
        df["month_num"] = started.dt.month
        df["month_label"] = started.dt.strftime("%Y-%m")
        df["day_name"] = started.dt.day_name()
        df["hour_key"] = started.dt.hour
    else:
        if "ride_date" in df.columns:
            rdate = pd.to_datetime(df["ride_date"], errors="coerce")
            df["month_num"] = rdate.dt.month
            df["month_label"] = rdate.dt.strftime("%Y-%m")
        if "start_hour" in df.columns and "hour_key" not in df.columns:
            df["hour_key"] = pd.to_numeric(df["start_hour"], errors="coerce")

    if "trip_duration_minutes" not in df.columns and "ended_at" in df.columns and "started_at" in df.columns:
        ended = pd.to_datetime(df["ended_at"], errors="coerce")
        started = pd.to_datetime(df["started_at"], errors="coerce")
        df["trip_duration_minutes"] = (ended - started).dt.total_seconds() / 60.0

    for col in ("trip_duration_minutes", "displacement_km", "start_lat", "start_lng", "hour_key"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "start_station_name" not in df.columns and "start_station_id" in df.columns:
        df["start_station_name"] = df["start_station_id"].astype(str)

    return df


@st.cache_data
def load_ml_targets(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path, dtype={"start_station_id": str}, low_memory=False)
    for col, default in [
        ("total_casual_trips", 0),
        ("predicted_member_like_trips", 0),
        ("conversion_propensity_score", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    df["priority_tier"] = df.apply(assign_priority_tier, axis=1)
    return df


st.title("🚲 Cyclistic Bike-Share Marketing & Target Dashboard")

silver_df = load_silver_data(SILVER_PATH)
ml_df = load_ml_targets(ML_TARGETS_PATH)

if silver_df.empty and ml_df.empty:
    st.error("Data Silver maupun ml_targets.csv belum ditemukan.")
    st.stop()

# Sidebar Filter
st.sidebar.header("Filter Dashboard")

filtered_silver = silver_df.copy()
if not filtered_silver.empty:
    if "month_label" in filtered_silver.columns:
        months = sorted(filtered_silver["month_label"].dropna().unique().tolist())
        sel_months = st.sidebar.multiselect("Bulan", options=months, default=months)
        filtered_silver = filtered_silver[filtered_silver["month_label"].isin(sel_months)]

    if "member_casual" in filtered_silver.columns:
        users = sorted(filtered_silver["member_casual"].dropna().unique().tolist())
        sel_users = st.sidebar.multiselect("Tipe Pengguna", options=users, default=users)
        filtered_silver = filtered_silver[filtered_silver["member_casual"].isin(sel_users)]

    if "rideable_type" in filtered_silver.columns:
        bikes = sorted(filtered_silver["rideable_type"].dropna().unique().tolist())
        sel_bikes = st.sidebar.multiselect("Jenis Sepeda", options=bikes, default=bikes)
        filtered_silver = filtered_silver[filtered_silver["rideable_type"].isin(sel_bikes)]

filtered_ml = ml_df.copy()
if not filtered_ml.empty:
    tier_opts = ["Prioritas Tinggi", "Prioritas Sedang", "Rendah"]
    avail_tiers = [t for t in tier_opts if t in filtered_ml["priority_tier"].unique()]
    sel_tiers = st.sidebar.multiselect("Tingkat Prioritas Target", options=avail_tiers, default=avail_tiers)
    min_trips = st.sidebar.slider(
        "Minimum Casual Trips per Sel",
        min_value=0,
        max_value=max(int(filtered_ml["total_casual_trips"].max()), 1),
        value=0,
        step=5,
    )
    filtered_ml = filtered_ml[
        filtered_ml["priority_tier"].isin(sel_tiers)
        & (filtered_ml["total_casual_trips"] >= min_trips)
    ]

# Bagian 1: Ringkasan Angka Penting
with st.container(border=True):
    st.subheader("1. Ringkasan")
    c1, c2, c3, c4 = st.columns(4)

    total_trips = len(filtered_silver) if not filtered_silver.empty else int(filtered_ml["total_casual_trips"].sum())
    with c1:
        with st.container(border=True):
            st.metric("Total Perjalanan", f"{total_trips:,}")

    with c2:
        with st.container(border=True):
            if not filtered_silver.empty and "member_casual" in filtered_silver.columns and len(filtered_silver) > 0:
                casual_pct = (filtered_silver["member_casual"] == "casual").mean() * 100
                member_pct = 100.0 - casual_pct
                st.metric("Member vs Casual", f"{member_pct:.1f}% / {casual_pct:.1f}%")
            else:
                st.metric("Member vs Casual", "-")

    with c3:
        with st.container(border=True):
            if not filtered_silver.empty and "member_casual" in filtered_silver.columns and "trip_duration_minutes" in filtered_silver.columns:
                dur_user = filtered_silver.groupby("member_casual")["trip_duration_minutes"].mean()
                avg_c = dur_user.get("casual", 0.0)
                avg_m = dur_user.get("member", 0.0)
                st.metric("Rata-Rata Durasi (Casual / Member)", f"{avg_c:.1f}m / {avg_m:.1f}m")
            else:
                st.metric("Rata-Rata Durasi (Casual / Member)", "-")

    with c4:
        with st.container(border=True):
            if not filtered_ml.empty and filtered_ml["total_casual_trips"].sum() > 0:
                commuter_pct = (
                    filtered_ml["predicted_member_like_trips"].sum()
                    / filtered_ml["total_casual_trips"].sum()
                    * 100
                )
                st.metric("Casual Berpola Mirip Member", f"{commuter_pct:.1f}%")
            else:
                st.metric("Casual Berpola Mirip Member", "-")

# Bagian 2: Grafik Waktu Bersepeda
with st.container(border=True):
    st.subheader("2. Grafik Waktu Bersepeda")
    if not filtered_silver.empty and "member_casual" in filtered_silver.columns:
        t1, t2, t3 = st.columns(3)

        with t1:
            st.markdown("**Grafik Bulanan**")
            if "month_label" in filtered_silver.columns:
                m_df = (
                    filtered_silver.groupby(["month_label", "member_casual"])
                    .size()
                    .unstack(fill_value=0)
                    .sort_index()
                )
                st.bar_chart(clean_labels(m_df))

        with t2:
            st.markdown("**Grafik Hari (Senin – Minggu)**")
            if "day_name" in filtered_silver.columns:
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                d_df = (
                    filtered_silver.groupby(["day_name", "member_casual"])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(day_order)
                    .dropna(how="all")
                )
                st.bar_chart(clean_labels(d_df))

        with t3:
            st.markdown("**Grafik Jam (0 – 23)**")
            if "hour_key" in filtered_silver.columns:
                h_df = (
                    filtered_silver.groupby(["hour_key", "member_casual"])
                    .size()
                    .unstack(fill_value=0)
                    .sort_index()
                )
                st.bar_chart(clean_labels(h_df))
    else:
        st.info("Data Silver belum tersedia untuk grafik waktu.")

# Bagian 3: Peta & Daftar Stasiun Paling Ramai
with st.container(border=True):
    st.subheader("3. Peta & Daftar Stasiun Paling Ramai (Casual)")
    if not filtered_silver.empty and "member_casual" in filtered_silver.columns:
        casual_silver = filtered_silver[filtered_silver["member_casual"] == "casual"].copy()
        left_map, right_top = st.columns([1.2, 1])

        with left_map:
            st.markdown("**Peta Kota Chicago (Perjalanan Casual)**")
            if {"start_lat", "start_lng"}.issubset(casual_silver.columns):
                map_df = casual_silver.dropna(subset=["start_lat", "start_lng"])
                map_df = map_df[
                    map_df["start_lat"].between(*CHICAGO_BOUNDS["lat"])
                    & map_df["start_lng"].between(*CHICAGO_BOUNDS["lng"])
                ]
                if not map_df.empty:
                    st.map(map_df.rename(columns={"start_lat": "lat", "start_lng": "lon"})[["lat", "lon"]])
                else:
                    st.info("Tidak ada koordinat stasiun Chicago pada filter aktif.")

        with right_top:
            st.markdown("**10 Stasiun Awal Terpadat (Casual)**")
            if "start_station_name" in casual_silver.columns:
                top_st = (
                    casual_silver.groupby("start_station_name", dropna=False)
                    .size()
                    .reset_index(name="total_trips")
                    .sort_values("total_trips", ascending=False)
                    .head(10)
                )
                st.dataframe(clean_labels(top_st), use_container_width=True, hide_index=True)
    else:
        st.info("Data Silver belum tersedia untuk peta dan daftar stasiun.")

# Bagian 4: Grafik Jenis Sepeda & Ciri Perjalanan Target Promo
with st.container(border=True):
    st.subheader("4. Jenis Sepeda & Ciri Perjalanan Target Promo")
    b_col, ml_col = st.columns(2)

    with b_col:
        st.markdown("**Pilihan Jenis Sepeda (Perjalanan & Rata-Rata Jarak)**")
        if not filtered_silver.empty and "rideable_type" in filtered_silver.columns:
            agg_spec = {"total_trips": ("rideable_type", "count")}
            if "displacement_km" in filtered_silver.columns:
                agg_spec["avg_distance_km"] = ("displacement_km", "mean")
            bike_df = filtered_silver.groupby("rideable_type", dropna=False).agg(**agg_spec).reset_index()
            if "avg_distance_km" in bike_df.columns:
                bike_df["avg_distance_km"] = bike_df["avg_distance_km"].round(2)
            st.dataframe(clean_labels(bike_df), use_container_width=True, hide_index=True)
            st.bar_chart(bike_df.set_index("rideable_type")["total_trips"].rename("Total Perjalanan"))
        else:
            st.info("Data jenis sepeda belum tersedia.")

    with ml_col:
        st.markdown("**Ciri Perjalanan Target Promo (Hasil Model ML)**")
        if not filtered_ml.empty:
            top_promo = filtered_ml.sort_values(
                by=["conversion_propensity_score", "total_casual_trips"],
                ascending=[False, False],
            ).head(25)
            st.dataframe(clean_labels(top_promo), use_container_width=True, hide_index=True)
        else:
            st.info("Data ml_targets.csv belum tersedia.")