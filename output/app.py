"""
Cyclistic: Dasbor Pemasaran (Casual vs Member)

Jalankan dari folder cyclistic-pipeline:
    python -m streamlit run app.py

Sumber data:
  - data/silver/            -> Parquet per trip (data bersih, dari Data Engineer)
  - output/ml_targets.csv   -> hasil model (kemiripan pola trip casual dengan member)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
SILVER_PATH = Path("data/silver")
ML_TARGETS_CANDIDATES = [Path("output/ml/ml_targets.csv"), Path("output/ml_targets.csv")]

# Aturan tingkat prioritas yang dipakai saat memuat tabel ke database (src/load/load_ml_targets.py).
# Samakan dengan skrip Data Engineer bila aturannya berubah.
TIER_MIN_TRIPS = 30
TIER_HIGH_SCORE = 1.0   # skor >= ini -> Prioritas Tinggi
TIER_MID_SCORE = 0.4    # skor > ini  -> Prioritas Sedang

DAY_LABELS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
EN_TO_ID = dict(zip(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], DAY_LABELS))
COLORS = {"casual": "#E4572E", "member": "#2E86AB"}
MIN_TRIPS_SHARE_TABLE = 500

# Nama kolom yang mungkin ada di Silver (kolom pertama yang cocok dipakai).
CANDIDATES = {
    "user": ["member_casual", "user_type"],
    "bike": ["rideable_type", "bike_type"],
    "started_at": ["started_at", "start_time"],
    "ended_at": ["ended_at", "end_time"],
    "duration": ["trip_duration_minutes", "duration_min", "ride_length_min"],
    "distance": ["displacement_km", "distance_km"],
    "start_id": ["start_station_id"],
    "start_name": ["start_station_name"],
    "end_id": ["end_station_id"],
    "end_name": ["end_station_name"],
    "lat": ["start_lat"],
    "lng": ["start_lng"],
    "false_start": ["is_false_start"],
    "outlier": ["is_outlier_24h"],
}

st.set_page_config(page_title="Cyclistic: Dasbor Pemasaran", layout="wide")


# ----------------------------------------------------------------------------
# Memuat data
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat data Silver (sekali saja)...")
def load_trips():
    """Baca Silver, simpan hanya kolom yang dipakai dengan tipe hemat memori."""
    if not SILVER_PATH.exists():
        return None, None
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(SILVER_PATH), format="parquet")
    available = set(dataset.schema.names)
    m = {k: next((c for c in v if c in available), None) for k, v in CANDIDATES.items()}
    for must in ("user", "bike", "started_at"):
        if m[must] is None:
            raise KeyError(f"Kolom wajib untuk '{must}' tidak ditemukan. Kolom yang ada: {sorted(available)}")

    cols = sorted({c for c in m.values() if c})
    table = dataset.to_table(columns=cols)
    for i, field in enumerate(table.schema):  # string -> dictionary agar hemat memori
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            table = table.set_column(i, field.name, pc.dictionary_encode(table[field.name]))
    raw = table.to_pandas()

    started = pd.to_datetime(raw[m["started_at"]])
    d = pd.DataFrame({
        "user": raw[m["user"]].astype("category"),
        "bike": raw[m["bike"]].astype("category"),
        "month": (started.dt.year * 100 + started.dt.month).astype("int32"),
        "dow": started.dt.dayofweek.astype("int8"),
        "hour": started.dt.hour.astype("int8"),
    })

    # Buang trip yang ditandai kotor di Silver
    keep = pd.Series(True, index=raw.index)
    for key in ("false_start", "outlier"):
        if m[key]:
            keep &= ~raw[m[key]].fillna(False).astype(bool)

    if m["duration"]:
        d["duration"] = pd.to_numeric(raw[m["duration"]], errors="coerce").astype("float32")
    elif m["ended_at"]:
        d["duration"] = ((pd.to_datetime(raw[m["ended_at"]]) - started).dt.total_seconds() / 60).astype("float32")
    if m["distance"]:
        d["distance"] = pd.to_numeric(raw[m["distance"]], errors="coerce").astype("float32")
    if m["lat"] and m["lng"]:
        d["lat"] = pd.to_numeric(raw[m["lat"]], errors="coerce").astype("float32")
        d["lng"] = pd.to_numeric(raw[m["lng"]], errors="coerce").astype("float32")

    for side in ("start", "end"):
        id_col, name_col = m[f"{side}_id"], m[f"{side}_name"]
        known = pd.Series(True, index=raw.index)
        if id_col:
            sid = raw[id_col].astype("category")
            known &= ~(sid.isna() | (sid == "UNKNOWN"))
            d[f"{side}_id"] = sid
        if name_col:
            nm = raw[name_col].astype("category")
            known &= ~(nm.isna() | (nm == "UNKNOWN"))
            d[f"{side}_name"] = nm
        if id_col or name_col:
            d[f"{side}_known"] = known.to_numpy()
            d[f"{side}_label"] = (d[f"{side}_name"] if name_col else d[f"{side}_id"])

    d = d[keep.to_numpy()].reset_index(drop=True)
    info = {
        "start": started[keep.to_numpy()].min(),
        "end": started[keep.to_numpy()].max(),
        "dropped": int((~keep).sum()),
        "has_duration": "duration" in d.columns,
        "has_distance": "distance" in d.columns,
        "has_start": "start_label" in d.columns,
        "has_end": "end_label" in d.columns,
        "has_geo": "lat" in d.columns,
    }
    return d, info


@st.cache_data
def load_ml(path_str: str, name_map: dict | None):
    df = pd.read_csv(path_str, dtype={"start_station_id": str}, low_memory=False)
    for col in ("total_casual_trips", "predicted_member_like_trips", "conversion_propensity_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    def tier(row):
        if row["total_casual_trips"] >= TIER_MIN_TRIPS and row["conversion_propensity_score"] >= TIER_HIGH_SCORE:
            return "Prioritas Tinggi"
        if row["total_casual_trips"] >= TIER_MIN_TRIPS and row["conversion_propensity_score"] > TIER_MID_SCORE:
            return "Prioritas Sedang"
        return "Rendah"

    df["priority_tier"] = df.apply(tier, axis=1)
    df["day_label"] = df["day_name"].map(EN_TO_ID)
    names = name_map or {}
    df["station"] = df["start_station_id"].map(names).fillna(df["start_station_id"])
    return df


def month_label(m: int) -> str:
    return f"{m // 100}-{m % 100:02d}"


def share_within_user(df: pd.DataFrame, key: str) -> pd.DataFrame:
    c = df.groupby([key, "user"], observed=True).size().rename("n").reset_index()
    c["persen"] = c["n"] / c.groupby("user", observed=True)["n"].transform("sum") * 100
    return c


def show(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend_title_text="")
    st.plotly_chart(fig)




# ----------------------------------------------------------------------------
# Halaman
# ----------------------------------------------------------------------------
def page_summary(df, info):
    c1, c2, c3, c4 = st.columns(4)
    n = len(df)
    c1.metric("Total perjalanan (filter aktif)", f"{n:,}")
    share = df["user"].value_counts(normalize=True) * 100
    c2.metric("Casual / Member", f"{share.get('casual', 0):.1f}% / {share.get('member', 0):.1f}%")
    if info["has_duration"]:
        med = df.groupby("user", observed=True)["duration"].median()
        c3.metric("Median durasi: casual", f"{med.get('casual', float('nan')):.1f} menit")
        c4.metric("Median durasi: member", f"{med.get('member', float('nan')):.1f} menit")


def page_when(df):
    c1, c2 = st.columns(2)

    h = share_within_user(df, "hour")
    fig = px.line(h, x="hour", y="persen", color="user", color_discrete_map=COLORS, markers=True,
                  labels={"hour": "Jam mulai", "persen": "% trip dalam kelompok", "user": ""}, title="Per jam")
    with c1:
        show(fig)

    d = share_within_user(df, "dow")
    d["hari"] = d["dow"].map(dict(enumerate(DAY_LABELS)))
    fig = px.bar(d, x="hari", y="persen", color="user", barmode="group", color_discrete_map=COLORS,
                 category_orders={"hari": DAY_LABELS}, labels={"hari": "", "persen": "% trip dalam kelompok"}, title="Per hari")
    with c2:
        show(fig)

    c3, c4 = st.columns(2)
    mth = df.groupby(["month", "user"], observed=True).size().unstack(fill_value=0)
    mth.index = [month_label(i) for i in mth.index]
    cnt = mth.rename_axis("bulan").reset_index().melt(id_vars="bulan", var_name="user", value_name="trip")
    fig = px.line(cnt, x="bulan", y="trip", color="user", color_discrete_map=COLORS, markers=True,
                  labels={"bulan": "", "trip": "Jumlah trip"}, title="Jumlah trip per bulan")
    with c3:
        show(fig)

    if {"casual", "member"}.issubset(mth.columns):
        pct = (mth["casual"] / (mth["casual"] + mth["member"]) * 100).rename("persen").rename_axis("bulan").reset_index()
        fig = px.bar(pct, x="bulan", y="persen", labels={"bulan": "", "persen": "% casual dari semua trip"},
                     title="Porsi casual per bulan", color_discrete_sequence=[COLORS["casual"]])
        with c4:
            show(fig)



def page_duration_bike(df, info):
    if info["has_duration"]:
        st.subheader("Durasi perjalanan")
        s = df.groupby("user", observed=True)["duration"].agg(Median="median", Rata_rata="mean", Persentil_90=lambda x: x.quantile(0.9))
        st.dataframe(s.round(1).rename(columns={"Rata_rata": "Rata-rata", "Persentil_90": "Persentil 90"}).rename_axis("Tipe"))
        sample = df.sample(n=min(300_000, len(df)), random_state=1)
        sample = sample[sample["duration"].between(0, 60)]
        fig = px.histogram(sample, x="duration", color="user", histnorm="percent", barmode="overlay", nbins=60,
                           opacity=0.6, color_discrete_map=COLORS,
                           labels={"duration": "Durasi (menit, dipotong 60)", "user": ""}, title="Sebaran durasi (sampel)")
        show(fig)
    else:
        st.info("Kolom durasi tidak ditemukan di Silver.")

    if info["has_distance"]:
        st.subheader("Jarak garis lurus titik awal ke titik akhir (km)")
        dist = df.groupby("user", observed=True)["distance"].agg(Median="median", Rata_rata="mean").round(2)
        st.dataframe(dist.rename(columns={"Rata_rata": "Rata-rata"}).rename_axis("Tipe"))

    st.subheader("Jenis sepeda")
    b = share_within_user(df, "bike")
    fig = px.bar(b, x="bike", y="persen", color="user", barmode="group", color_discrete_map=COLORS,
                 labels={"bike": "", "persen": "% trip dalam kelompok"}, title="Pilihan jenis sepeda per kelompok")
    show(fig)
    tbl = df.groupby(["bike", "user"], observed=True).size().unstack(fill_value=0)
    st.dataframe(tbl.rename_axis("Jenis sepeda"))


def top_stations(df, side, user, n=10):
    sub = df[(df["user"] == user) & df[f"{side}_known"]]
    return sub.groupby(f"{side}_label", observed=True).size().sort_values(ascending=False).head(n).rename("Jumlah trip").reset_index() \
              .rename(columns={f"{side}_label": "Stasiun"})


def page_where(df, info):
    if not (info["has_start"] or info["has_end"]):
        st.info("Kolom stasiun tidak ditemukan di Silver.")
        return
    unk_start = (~df["start_known"]).mean() * 100 if info["has_start"] else 0

    for side, title, ok in (("start", "Stasiun awal", info["has_start"]), ("end", "Stasiun akhir", info["has_end"])):
        if not ok:
            continue
        st.subheader(f"10 {title} terpadat")
        c1, c2 = st.columns(2)
        for col, u in zip((c1, c2), ("casual", "member")):
            with col:
                st.markdown(f"**{u.capitalize()}**")
                st.dataframe(top_stations(df, side, u), hide_index=True)

    if info["has_start"] and info["has_geo"]:
        st.subheader("Peta stasiun awal")
        who = st.radio("Tampilkan perjalanan", ["casual", "member"], horizontal=True)
        sub = df[(df["user"] == who) & df["start_known"]]
        g = sub.groupby("start_label", observed=True).agg(lat=("lat", "mean"), lon=("lng", "mean"), trips=("lat", "size")).dropna().reset_index()
        g = g[g["lat"].between(41.5, 42.2) & g["lon"].between(-88.1, -87.4)]
        g["size"] = 20 + np.sqrt(g["trips"]) * 4
        g[["lat", "lon", "size"]] = g[["lat", "lon", "size"]].astype("float64")
        st.map(g, latitude="lat", longitude="lon", size="size")

        st.subheader(f"Stasiun dengan porsi casual tertinggi (minimal {MIN_TRIPS_SHARE_TABLE} trip)")
        allst = df[df["start_known"]].groupby(["start_label", "user"], observed=True).size().unstack(fill_value=0)
        if {"casual", "member"}.issubset(allst.columns):
            allst["Total trip"] = allst["casual"] + allst["member"]
            allst = allst[allst["Total trip"] >= MIN_TRIPS_SHARE_TABLE]
            allst["Porsi casual (%)"] = (allst["casual"] / allst["Total trip"] * 100).round(1)
            st.dataframe(allst.sort_values("Porsi casual (%)", ascending=False).head(15)[["Total trip", "Porsi casual (%)"]].rename_axis("Stasiun"))


def page_model(ml, info):
    if ml is None:
        st.info("File ml_targets.csv belum ditemukan di folder output.")
        return
    tot, mirip = ml["total_casual_trips"].sum(), ml["predicted_member_like_trips"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Trip casual yang dinilai model", f"{int(tot):,}")
    c2.metric("Trip casual dengan pola mirip member", f"{int(mirip):,}")
    c3.metric("% trip casual dengan pola mirip member", f"{mirip / tot * 100:.1f}%" if tot else "-")

    tiers = ml["priority_tier"].value_counts()
    t1, t2, t3 = st.columns(3)
    t1.metric("Sel Prioritas Tinggi", f"{int(tiers.get('Prioritas Tinggi', 0)):,}")
    t2.metric("Sel Prioritas Sedang", f"{int(tiers.get('Prioritas Sedang', 0)):,}")
    t3.metric("Sel Rendah", f"{int(tiers.get('Rendah', 0)):,}")

    def ratio(key):
        g = ml.groupby(key).agg(total=("total_casual_trips", "sum"), mirip=("predicted_member_like_trips", "sum")).reset_index()
        g["persen"] = g["mirip"] / g["total"] * 100
        return g

    c1, c2 = st.columns(2)
    with c1:
        show(px.line(ratio("hour_key"), x="hour_key", y="persen", markers=True,
                     labels={"hour_key": "Jam", "persen": "% trip casual mirip member"}, title="Per jam"))
    with c2:
        gd = ratio("day_label")
        show(px.bar(gd, x="day_label", y="persen", category_orders={"day_label": DAY_LABELS},
                    labels={"day_label": "", "persen": "% trip casual mirip member"}, title="Per hari"))
    show(px.bar(ratio("rideable_type"), x="rideable_type", y="persen",
                labels={"rideable_type": "", "persen": "% trip casual mirip member"}, title="Per jenis sepeda"))

    st.subheader("Sel dengan prioritas tertinggi")
    f1, f2 = st.columns(2)
    avail = [t for t in ["Prioritas Tinggi", "Prioritas Sedang", "Rendah"] if t in tiers.index]
    sel = f1.multiselect("Tingkat prioritas", avail, default=avail[:2] or avail)
    min_t = f2.slider("Minimal trip casual per sel", 0, int(ml["total_casual_trips"].max()), TIER_MIN_TRIPS)
    view = ml[ml["priority_tier"].isin(sel) & (ml["total_casual_trips"] >= min_t)]
    view = view.sort_values(["conversion_propensity_score", "total_casual_trips"], ascending=False).head(50)
    out = view[["station", "day_label", "hour_key", "rideable_type", "total_casual_trips",
                "predicted_member_like_trips", "conversion_propensity_score", "priority_tier"]]
    out.columns = ["Stasiun", "Hari", "Jam", "Jenis sepeda", "Trip casual", "Trip mirip member", "Skor kemiripan pola", "Tingkat prioritas"]
    st.dataframe(out, hide_index=True)




# ----------------------------------------------------------------------------
# Aplikasi
# ----------------------------------------------------------------------------
st.title("🚲 Cyclistic: Casual vs Member")

trips, info = load_trips()

name_map = None
if trips is not None and info["has_start"] and "start_id" in trips.columns and "start_name" in trips.columns:
    pairs = trips.loc[trips["start_known"], ["start_id", "start_name"]].drop_duplicates()
    name_map = dict(zip(pairs["start_id"].astype(str), pairs["start_name"].astype(str)))

ml_path = next((p for p in ML_TARGETS_CANDIDATES if p.exists()), None)
ml = load_ml(str(ml_path), name_map) if ml_path else None

if trips is None and ml is None:
    st.error("Data Silver (data/silver) dan ml_targets.csv belum ditemukan. Letakkan keduanya sesuai konfigurasi di bagian atas app.py.")
    st.stop()

if trips is None:
    st.warning("Data Silver belum ditemukan, hanya halaman hasil model yang ditampilkan.")
    page_model(ml, None)
    st.stop()

# Filter
st.sidebar.header("Filter")
months = sorted(trips["month"].unique().tolist())
sel_m = st.sidebar.multiselect("Bulan", months, default=months, format_func=month_label)
bikes = sorted(trips["bike"].dropna().unique().tolist())
sel_b = st.sidebar.multiselect("Jenis sepeda", bikes, default=bikes)
users = sorted(trips["user"].dropna().unique().tolist())
sel_u = st.sidebar.multiselect("Tipe pengguna", users, default=users)
daytype = st.sidebar.radio("Hari", ["Semua", "Hari kerja", "Akhir pekan"])

mask = trips["month"].isin(sel_m) & trips["bike"].isin(sel_b) & trips["user"].isin(sel_u)
if daytype == "Hari kerja":
    mask &= trips["dow"] < 5
elif daytype == "Akhir pekan":
    mask &= trips["dow"] >= 5
df = trips[mask]
if df.empty:
    st.warning("Tidak ada data untuk kombinasi filter ini.")
    st.stop()

tabs = st.tabs(["Ringkasan", "Kapan", "Durasi & sepeda", "Di mana", "Target promosi"])
with tabs[0]:
    page_summary(df, info)
with tabs[1]:
    page_when(df)
with tabs[2]:
    page_duration_bike(df, info)
with tabs[3]:
    page_where(df, info)
with tabs[4]:
    page_model(ml, info)