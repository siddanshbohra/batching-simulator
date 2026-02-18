"""
Logistics Batching Optimizer & Simulation Dashboard
====================================================
FirstClub Quick Commerce — Rider Productivity & Breach Simulation

Uses actual delivery lat/lng from "Orders Raw" sheet — zero synthetic data.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Any

# ─── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="FirstClub — Batching Optimizer",
    page_icon="🚚",
)

SERVICE_TIME_PENALTY_MIN = 8
BREACH_REDISTRIBUTION_EFF = 0.35
EARTH_RADIUS_KM = 6371.0
DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "Batching Analysis.xlsx"

st.markdown("""<style>
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    section[data-testid="stSidebar"] { width: 320px !important; }
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading data from Excel…")
def load_data(_file):
    xls = pd.ExcelFile(_file)

    orders = pd.read_excel(xls, sheet_name="orders")
    breach = pd.read_excel(xls, sheet_name="breach")
    rider = pd.read_excel(xls, sheet_name="Rider Availability")
    distance = pd.read_excel(xls, sheet_name="Distance")
    buckets = pd.read_excel(xls, sheet_name="Distance Buckets")
    wh = pd.read_excel(xls, sheet_name="WH Coordinates")
    raw = pd.read_excel(xls, sheet_name="Orders Raw")

    for df, col in [
        (orders, "order_date_ist"), (orders, "order_created_at_ist"),
        (breach, "order_date_ist"), (rider, "shift_date_ist"),
        (distance, "order_date_ist"), (buckets, "shift_date_ist"),
        (raw, "order_date_ist"), (raw, "order_created_at_ist"),
    ]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    wh = wh.rename(columns={"Warehouse ID": "ch_id", "Name": "wh_name"})
    orders["order_hour_ist"] = orders["order_hour_ist"].astype(float)
    raw["order_hour_ist"] = pd.to_numeric(raw["order_hour_ist"], errors="coerce")

    breach_map = orders.set_index("order_id")[["breach_status", "delay_minutes"]].to_dict("index")
    raw["breach_status"] = raw["order_id"].map(
        lambda x: breach_map.get(x, {}).get("breach_status", None)
    )
    raw["delay_minutes"] = raw["order_id"].map(
        lambda x: breach_map.get(x, {}).get("delay_minutes", None)
    )

    raw = raw.dropna(subset=["delivery_lat", "delivery_lng", "order_created_at_ist"])

    breached_orders = orders[orders["breach_status"] == "BREACHED"]["delay_minutes"].dropna()
    global_breach_delay = float(breached_orders.mean()) if len(breached_orders) >= 5 else 5.0

    return orders, breach, rider, distance, buckets, wh, raw, global_breach_delay


# ═══════════════════════════════════════════════════════════════════════════════
#  VECTORISED HAVERSINE
# ═══════════════════════════════════════════════════════════════════════════════

def haversine_matrix(lats: np.ndarray, lngs: np.ndarray) -> np.ndarray:
    """Pairwise Haversine distance matrix (km). Fully vectorised with numpy."""
    lat_r = np.radians(lats)
    lng_r = np.radians(lngs)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlng = lng_r[:, None] - lng_r[None, :]
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) *
         np.sin(dlng / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ═══════════════════════════════════════════════════════════════════════════════
#  BATCHING ENGINE — Greedy Proximity (deterministic, actual coordinates)
# ═══════════════════════════════════════════════════════════════════════════════

def run_batching(orders_sub: pd.DataFrame, R_km: float, T_min: float,
                 S_max: int, active_riders: float) -> Dict[str, Any]:
    """
    Greedy Proximity Batching using actual delivery lat/lng.

    Groups orders if ALL conditions met:
      - Haversine distance between delivery points <= R_km
      - Timestamp gap from first order in batch  <= T_min
      - Batch size <= S_max
    Total trips capped at active_riders.
    """
    empty = {
        "total_orders": 0, "batches": [], "singles": [],
        "total_trips": 0, "batch_trips": 0, "single_trips": 0,
        "batched_count": 0, "capped": False,
        "dist_mat_stats": {}, "time_mat_stats": {},
        "constraint_breakdown": {},
    }
    if orders_sub.empty:
        return empty

    df = orders_sub.sort_values("order_created_at_ist").reset_index(drop=True)
    n = len(df)

    dist_mat = haversine_matrix(df["delivery_lat"].values, df["delivery_lng"].values)
    times = df["order_created_at_ist"].values.astype("int64")
    time_mat = np.abs(times[:, None] - times[None, :]) / 6e10  # ns -> min

    triu_idx = np.triu_indices(n, k=1)
    upper_dists = dist_mat[triu_idx]
    upper_times = time_mat[triu_idx]
    n_pairs = len(upper_dists)

    pairs_within_R = int((upper_dists <= R_km).sum())
    pairs_within_T = int((upper_times <= T_min).sum())
    pairs_within_both = int(((upper_dists <= R_km) & (upper_times <= T_min)).sum())

    assigned = np.zeros(n, dtype=bool)
    batches: List[List[int]] = []

    for i in range(n):
        if assigned[i]:
            continue
        batch = [i]
        assigned[i] = True

        for j in range(i + 1, n):
            if assigned[j] or len(batch) >= S_max:
                continue
            if time_mat[batch[0], j] > T_min:
                continue
            if all(dist_mat[b, j] <= R_km for b in batch):
                batch.append(j)
                assigned[j] = True

        batches.append(batch)

    capped = False
    if active_riders > 0 and len(batches) > active_riders:
        batches = batches[: int(active_riders)]
        capped = True

    multi = [b for b in batches if len(b) > 1]
    singles = [b for b in batches if len(b) == 1]

    return {
        "total_orders": n,
        "batches": multi,
        "singles": singles,
        "all_trips": batches,
        "total_trips": len(batches),
        "batch_trips": len(multi),
        "single_trips": len(singles),
        "batched_count": sum(len(b) for b in multi),
        "capped": capped,
        "dist_mat_stats": {
            "n_pairs": n_pairs,
            "min": float(upper_dists.min()) if n_pairs else 0,
            "mean": float(upper_dists.mean()) if n_pairs else 0,
            "median": float(np.median(upper_dists)) if n_pairs else 0,
            "max": float(upper_dists.max()) if n_pairs else 0,
            "pairs_within_R": pairs_within_R,
            "pct_within_R": pairs_within_R / n_pairs * 100 if n_pairs else 0,
        },
        "time_mat_stats": {
            "pairs_within_T": pairs_within_T,
            "pct_within_T": pairs_within_T / n_pairs * 100 if n_pairs else 0,
        },
        "constraint_breakdown": {
            "pairs_within_R": pairs_within_R,
            "pairs_within_T": pairs_within_T,
            "pairs_within_both": pairs_within_both,
            "pct_within_both": pairs_within_both / n_pairs * 100 if n_pairs else 0,
            "n_pairs": n_pairs,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  L1 / L2 METRIC ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_l1_l2(result: Dict, breach_count: float, breach_rate: float,
                  avg_o2r: float, avg_promise: float,
                  total_orders: int,
                  avg_breach_delay_est: float = 5.0) -> Dict[str, Any]:
    actual_trips = total_orders
    sim_trips = result["total_trips"]
    batched = result["batched_count"]
    n_batch_trips = result["batch_trips"]

    actual_prod = total_orders / actual_trips if actual_trips else 0
    sim_prod = total_orders / sim_trips if sim_trips else 0
    prod_delta = sim_prod - actual_prod
    prod_delta_pct = (prod_delta / actual_prod * 100) if actual_prod else 0

    trips_saved = actual_trips - sim_trips
    trip_reduction_pct = (trips_saved / actual_trips * 100) if actual_trips else 0

    extra_stops = max(batched - n_batch_trips, 0)
    service_penalty = extra_stops * SERVICE_TIME_PENALTY_MIN
    gross_time_saved = trips_saved * avg_o2r
    net_time_saved = max(gross_time_saved - service_penalty, 0)

    avg_breach_delay = avg_breach_delay_est if breach_count > 0 else 0
    effective_time = net_time_saved * BREACH_REDISTRIBUTION_EFF
    if breach_count > 0 and effective_time > 0:
        breaches_prevented = min(
            int(breach_count),
            int(effective_time / max(avg_breach_delay, 1)),
        )
    else:
        breaches_prevented = 0

    if breach_count > 0:
        new_breach_rate = max(0, breach_rate * (1 - breaches_prevented / breach_count))
    else:
        new_breach_rate = breach_rate

    batching_pct = (batched / total_orders * 100) if total_orders else 0
    avg_batch_size = (batched / n_batch_trips) if n_batch_trips else 0

    return {
        "l1": {
            "actual_prod": actual_prod, "sim_prod": sim_prod,
            "prod_delta": prod_delta, "prod_delta_pct": prod_delta_pct,
            "actual_trips": actual_trips, "sim_trips": sim_trips,
            "trips_saved": trips_saved, "trip_reduction_pct": trip_reduction_pct,
            "actual_breach_rate": breach_rate, "sim_breach_rate": new_breach_rate,
            "breaches_prevented": breaches_prevented, "breach_count": breach_count,
        },
        "l2": {
            "batching_pct": batching_pct, "avg_batch_size": avg_batch_size,
            "gross_time_saved": gross_time_saved,
            "service_penalty": service_penalty,
            "net_time_saved": net_time_saved,
            "effective_time": effective_time,
            "extra_stops": extra_stops,
            "avg_breach_delay": avg_breach_delay,
            "capped": result["capped"],
        },
        "inputs": {
            "avg_o2r": avg_o2r, "avg_promise": avg_promise,
            "n_batch_trips": n_batch_trips,
            "total_batched": batched, "total_orders": total_orders,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — DESCRIPTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def render_analytics(breach: pd.DataFrame, rider: pd.DataFrame,
                     distance: pd.DataFrame, buckets: pd.DataFrame,
                     wh: pd.DataFrame):
    st.header("Descriptive Analytics")

    all_dates = sorted(breach["order_date_ist"].dropna().dt.date.unique())
    if not all_dates:
        st.warning("No breach data available.")
        return

    date_range = st.date_input(
        "Date Range", value=(all_dates[0], all_dates[-1]),
        min_value=all_dates[0], max_value=all_dates[-1], key="a_dates",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d_start, d_end = date_range
    else:
        d_start = d_end = date_range[0] if isinstance(date_range, tuple) else date_range

    bf = breach[(breach["order_date_ist"].dt.date >= d_start) &
                (breach["order_date_ist"].dt.date <= d_end)]
    rf = rider[(rider["shift_date_ist"].dt.date >= d_start) &
               (rider["shift_date_ist"].dt.date <= d_end)]
    bk = buckets[(buckets["shift_date_ist"].dt.date >= d_start) &
                 (buckets["shift_date_ist"].dt.date <= d_end)]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Orders", f"{bf['order_count'].sum():,.0f}")
    k2.metric("Avg Breach Rate", f"{bf['breach_rate_pct'].mean():.1f}%")
    k3.metric("Avg Rider Utilisation", f"{rf['utilization_rate_pct'].mean():.1f}%")
    k4.metric("Active Hubs", f"{bf['ch_id'].nunique()}")
    st.divider()

    c_map, c_eff = st.columns(2)
    with c_map:
        st.subheader("Warehouse Breach Heatmap")
        hub_agg = (
            bf.groupby("ch_id")
            .agg(avg_breach=("breach_rate_pct", "mean"),
                 total_orders=("order_count", "sum"),
                 total_breaches=("breach_count", "sum"))
            .reset_index().merge(wh, on="ch_id", how="left")
        )
        fig_map = px.scatter_mapbox(
            hub_agg, lat="Lat", lon="Lng",
            size="avg_breach", color="avg_breach",
            color_continuous_scale="RdYlGn_r", size_max=35, zoom=11,
            hover_name="wh_name",
            hover_data={"avg_breach": ":.1f", "total_orders": ":,.0f",
                        "total_breaches": ":,.0f", "Lat": False, "Lng": False},
            labels={"avg_breach": "Breach %"}, mapbox_style="open-street-map",
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=420)
        st.plotly_chart(fig_map, use_container_width=True)

    with c_eff:
        st.subheader("Rider Utilisation vs Hour of Day")
        hourly = rf.groupby("shift_hour_ist")["utilization_rate_pct"].mean().reset_index()
        fig_eff = px.area(
            hourly, x="shift_hour_ist", y="utilization_rate_pct",
            labels={"shift_hour_ist": "Hour (IST)", "utilization_rate_pct": "Utilisation %"},
            color_discrete_sequence=["#6366f1"],
        )
        fig_eff.add_hline(y=100, line_dash="dash", line_color="red",
                          annotation_text="100 % Capacity")
        fig_eff.update_layout(height=420, xaxis_dtick=1,
                              yaxis_range=[0, max(110, hourly["utilization_rate_pct"].max() + 10)])
        st.plotly_chart(fig_eff, use_container_width=True)

    st.divider()
    st.subheader("Order Volume by Distance Bucket")
    if not bk.empty:
        bk_agg = (
            bk.groupby("distance_bucket")
            .agg(total_orders=("bucket_orders", "sum"), avg_dist=("avg_distance_km", "mean"))
            .reset_index().sort_values("distance_bucket")
        )
        c1, c2 = st.columns([3, 1])
        with c1:
            fig_b = px.bar(bk_agg, x="distance_bucket", y="total_orders",
                           color="avg_dist", color_continuous_scale="Blues",
                           labels={"distance_bucket": "Distance Bucket",
                                   "total_orders": "Orders", "avg_dist": "Avg km"})
            fig_b.update_layout(height=350)
            st.plotly_chart(fig_b, use_container_width=True)
        with c2:
            st.dataframe(
                bk_agg.rename(columns={"distance_bucket": "Bucket",
                                       "total_orders": "Orders", "avg_dist": "Avg km"})
                .style.format({"Orders": "{:,.0f}", "Avg km": "{:.2f}"}),
                use_container_width=True, height=350)

    st.divider()
    st.subheader("Hourly Breach Rate by Hub")
    hub_hr = bf.groupby(["ch_id", "order_hour_ist"])["breach_rate_pct"].mean().reset_index()
    name_map = wh.set_index("ch_id")["wh_name"].to_dict()
    hub_hr["hub"] = hub_hr["ch_id"].map(name_map)
    pivot = hub_hr.pivot_table(index="hub", columns="order_hour_ist",
                               values="breach_rate_pct", aggfunc="mean")
    fig_ht = px.imshow(pivot, color_continuous_scale="RdYlGn_r", aspect="auto",
                       labels=dict(x="Hour (IST)", y="Hub", color="Breach %"))
    fig_ht.update_layout(height=450)
    st.plotly_chart(fig_ht, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — BATCHING SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

def render_simulator(raw: pd.DataFrame, rider: pd.DataFrame,
                     breach: pd.DataFrame, wh: pd.DataFrame,
                     global_breach_delay: float = 5.0):
    st.header("Batching Simulator")
    st.caption("Uses actual delivery lat/lng from Orders Raw — zero synthetic data.")
    hub_names = wh.set_index("ch_id")["wh_name"].to_dict()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Simulation Parameters")

        avail_dates = sorted(raw["order_date_ist"].dropna().dt.date.unique())
        sel_date = st.selectbox("Date", avail_dates,
                                index=len(avail_dates) - 1, key="sim_date")

        date_mask = raw["order_date_ist"].dt.date == sel_date
        hubs = sorted(raw.loc[date_mask, "ch_id"].unique())
        sel_hub = st.selectbox("Hub", hubs,
                               format_func=lambda x: f"{hub_names.get(x, x)} ({x})",
                               key="sim_hub")

        hub_mask = date_mask & (raw["ch_id"] == sel_hub)
        hours = sorted(raw.loc[hub_mask, "order_hour_ist"].dropna().unique())
        if not hours:
            hours = [0.0]
        min_h, max_h = int(hours[0]), int(hours[-1])
        default_lo = max(min_h, min(10, max_h))
        default_hi = min(max_h, max(default_lo, 12))
        hour_range = st.slider(
            "Time Range (IST hours)", min_value=min_h, max_value=max_h,
            value=(default_lo, default_hi), key="sim_hours",
        )

        st.divider()
        st.subheader("Batching Logic")
        R_km = st.slider("Batching Radius R (km)", 0.5, 5.0, 2.0, 0.25)
        T_min = st.slider("Time Window T (min)", 3, 20, 10, 1)
        S_max = st.slider("Max Batch Size S", 2, 5, 3, 1)

        st.divider()
        run_btn = st.button("🚀 Run Simulation", type="primary",
                            use_container_width=True)

    # ── Filter orders by time range ──────────────────────────────────────────
    sim_orders = raw.loc[
        hub_mask &
        (raw["order_hour_ist"] >= hour_range[0]) &
        (raw["order_hour_ist"] <= hour_range[1])
    ].copy()

    # ── Aggregate rider data across selected hours ───────────────────────────
    rider_rows = rider[
        (rider["shift_date_ist"].dt.date == sel_date) &
        (rider["shift_hour_ist"] >= hour_range[0]) &
        (rider["shift_hour_ist"] <= hour_range[1]) &
        (rider["ch_id"] == sel_hub)
    ]
    avg_active = float(rider_rows["active_riders"].mean()) if not rider_rows.empty else 0
    avg_avail = float(rider_rows["available_riders"].mean()) if not rider_rows.empty else 0
    n_hours = hour_range[1] - hour_range[0] + 1
    rider_cap = avg_avail * n_hours

    # ── Aggregate breach data across selected hours ──────────────────────────
    breach_rows = breach[
        (breach["order_date_ist"].dt.date == sel_date) &
        (breach["order_hour_ist"] >= hour_range[0]) &
        (breach["order_hour_ist"] <= hour_range[1]) &
        (breach["ch_id"] == sel_hub)
    ]
    total_breach_sheet_orders = float(breach_rows["order_count"].sum()) if not breach_rows.empty else 0
    breach_count = float(breach_rows["breach_count"].sum()) if not breach_rows.empty else 0
    breach_rate = (breach_count / total_breach_sheet_orders * 100) if total_breach_sheet_orders > 0 else 0
    avg_promise = float(breach_rows["avg_promise_minutes"].mean()) if not breach_rows.empty else 15

    o2r_vals = breach_rows["o2r_avg"].dropna() if not breach_rows.empty else pd.Series(dtype=float)
    avg_o2r = float(o2r_vals.mean()) if len(o2r_vals) > 0 else 12.0

    if avg_o2r == 12.0 and "actual_wh_to_cust_min" in sim_orders.columns:
        order_o2r = sim_orders["actual_wh_to_cust_min"].dropna()
        if not order_o2r.empty:
            avg_o2r = float(order_o2r.mean())

    # ── Context Cards ────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Orders", len(sim_orders))
    m2.metric("Avg Available Riders/hr", f"{avg_avail:.0f}" if avg_avail else "—")
    m3.metric("Rider Cap (range)", f"{rider_cap:.0f}" if rider_cap else "—",
              help=f"Avg available ({avg_avail:.0f}) x {n_hours} hours")
    m4.metric("Breach Rate", f"{breach_rate:.1f}%")
    m5.metric("Avg O2R", f"{avg_o2r:.1f} min")

    if not run_btn:
        st.info("Configure parameters in the sidebar and click **Run Simulation**.")
        return

    if sim_orders.empty:
        st.warning("No orders with delivery coordinates for this Date / Hub / Time Range.")
        return

    total_n = len(sim_orders)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1 — DATA SCOPE & DISTANCE PROFILE
    # ══════════════════════════════════════════════════════════════════════════

    st.divider()
    st.subheader("Step 1 — Data Scope & Delivery Spread")

    dc1, dc2 = st.columns([1, 2])
    with dc1:
        wh_dists = sim_orders["pickup_to_delivery_distance_km"].dropna()
        if not wh_dists.empty:
            st.markdown(f"""
| Stat | Value |
|---|---|
| Orders | **{total_n}** |
| Min WH→Cust | **{wh_dists.min():.2f} km** |
| Mean WH→Cust | **{wh_dists.mean():.2f} km** |
| Median WH→Cust | **{wh_dists.median():.2f} km** |
| Max WH→Cust | **{wh_dists.max():.2f} km** |
""")
        if "distance_bucket" in sim_orders.columns:
            bucket_dist = sim_orders["distance_bucket"].value_counts().sort_index()
            st.markdown("**Distance bucket split:**")
            st.dataframe(
                pd.DataFrame({"Bucket": bucket_dist.index,
                              "Orders": bucket_dist.values,
                              "Share": (bucket_dist.values / total_n * 100).round(1)}),
                use_container_width=True, hide_index=True,
            )

    with dc2:
        if not wh_dists.empty:
            fig_dist = px.histogram(
                sim_orders, x="pickup_to_delivery_distance_km", nbins=20,
                color_discrete_sequence=["#6366f1"],
                labels={"pickup_to_delivery_distance_km": "Distance from WH (km)"},
                title="Delivery Distance Distribution (actual)",
            )
            fig_dist.update_layout(height=300, margin=dict(t=30, b=0))
            st.plotly_chart(fig_dist, use_container_width=True)

    with st.expander("Data source details", expanded=False):
        st.markdown(f"""
**Source:** `Orders Raw` sheet from the uploaded Excel file.

Each order has **actual** `delivery_lat` / `delivery_lng` from the logistics system.
No coordinates are generated or estimated.

- Hub: **{hub_names.get(sel_hub, sel_hub)}** ({sel_hub})
- Date: **{sel_date}**, Time range: **{hour_range[0]}:00 – {hour_range[1]}:59 IST**
- Orders with valid coordinates: **{total_n}**
- Also available per order: `pickup_to_delivery_distance_km`, `distance_bucket`, `promise_time_min`
""")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2 — GREEDY PROXIMITY BATCHING (deterministic)
    # ══════════════════════════════════════════════════════════════════════════

    with st.spinner("Computing Haversine distance matrix & batching…"):
        result = run_batching(sim_orders, R_km, T_min, S_max, rider_cap)

    st.divider()
    st.subheader("Step 2 — Greedy Proximity Batching")

    ds = result["dist_mat_stats"]
    cb = result["constraint_breakdown"]

    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("Order pairs", f"{cb['n_pairs']:,}")
    bc2.metric(f"Within R={R_km}km", f"{cb['pairs_within_R']:,}",
               help=f"{ds['pct_within_R']:.1f}% of all pairs")
    bc3.metric(f"Within T={T_min}min", f"{cb['pairs_within_T']:,}")
    bc4.metric("Both constraints", f"{cb['pairs_within_both']:,}",
               help=f"{cb['pct_within_both']:.1f}% of all pairs")

    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.metric("Multi-order batches", result["batch_trips"])
    ec2.metric("Orders batched", result["batched_count"])
    ec3.metric("Single trips", result["single_trips"])
    ec4.metric("Total trips", result["total_trips"])

    with st.expander("Reasoning: How the Greedy Proximity algorithm works", expanded=False):
        st.markdown(f"""
**Input:** {total_n} orders with actual delivery coordinates, sorted by `order_created_at_ist`.

**Pre-computation:**
- Haversine distance matrix: {total_n}x{total_n} = **{cb['n_pairs']:,}** unique pairs
- Inter-delivery distance: min **{ds['min']:.2f} km**, mean **{ds['mean']:.2f} km**, median **{ds['median']:.2f} km**, max **{ds['max']:.2f} km**

**Constraint filter (on {cb['n_pairs']:,} pairs):**

| Constraint | Pairs passing | % |
|---|---|---|
| Distance <= {R_km} km | {cb['pairs_within_R']:,} | {ds['pct_within_R']:.1f}% |
| Time <= {T_min} min | {cb['pairs_within_T']:,} | {result['time_mat_stats']['pct_within_T']:.1f}% |
| **Both** | **{cb['pairs_within_both']:,}** | **{cb['pct_within_both']:.1f}%** |

**Greedy scan:**
1. Start with order #1, try to add the next closest order that meets ALL 3 constraints (distance <= R, time <= T, batch size <= S={S_max}).
2. Once no more orders fit, close the batch and move to the next unassigned order.
3. Repeat until all orders are assigned.

**Result:** {result['batch_trips']} multi-order batches ({result['batched_count']} orders) + {result['single_trips']} singles = **{result['total_trips']} trips**
{'⚠️ **Rider cap applied:** Capped at ' + str(int(rider_cap)) + ' available riders (' + str(int(avg_avail)) + '/hr x ' + str(n_hours) + ' hrs).' if result['capped'] else ''}
""")

    if result["batches"]:
        sorted_df = sim_orders.sort_values("order_created_at_ist").reset_index(drop=True)
        batch_rows = []
        for i, batch in enumerate(result["batches"]):
            oids = [str(sorted_df.iloc[idx]["order_id"])[:16] for idx in batch]
            dists_in_batch = []
            for a in range(len(batch)):
                for b in range(a + 1, len(batch)):
                    d = haversine_matrix(
                        np.array([sorted_df.iloc[batch[a]]["delivery_lat"],
                                  sorted_df.iloc[batch[b]]["delivery_lat"]]),
                        np.array([sorted_df.iloc[batch[a]]["delivery_lng"],
                                  sorted_df.iloc[batch[b]]["delivery_lng"]]),
                    )[0, 1]
                    dists_in_batch.append(d)
            max_d = max(dists_in_batch) if dists_in_batch else 0
            t0 = sorted_df.iloc[batch[0]]["order_created_at_ist"]
            t1 = sorted_df.iloc[batch[-1]]["order_created_at_ist"]
            span = (t1 - t0).total_seconds() / 60

            batch_rows.append({
                "Batch": f"#{i+1}",
                "Orders": len(batch),
                "Max inter-dist (km)": round(max_d, 2),
                "Time span (min)": round(span, 1),
                "Order IDs": ", ".join(oids),
            })

        with st.expander(f"Batch detail table ({len(result['batches'])} batches)", expanded=False):
            st.dataframe(pd.DataFrame(batch_rows), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3 — L1 / L2 METRICS WITH REASONING
    # ══════════════════════════════════════════════════════════════════════════

    sim_breached = sim_orders[sim_orders["breach_status"] == "BREACHED"]["delay_minutes"].dropna()
    if len(sim_breached) >= 2:
        avg_breach_delay_est = float(sim_breached.mean())
    elif global_breach_delay > 0:
        avg_breach_delay_est = global_breach_delay
    else:
        avg_breach_delay_est = max(avg_o2r - avg_promise, 5)

    metrics = compute_l1_l2(result, breach_count, breach_rate,
                            avg_o2r, avg_promise, total_n,
                            avg_breach_delay_est)
    l1 = metrics["l1"]
    l2 = metrics["l2"]
    inp = metrics["inputs"]

    st.divider()
    st.subheader("L1 Metrics — North-Star Impact")

    lc1, lc2, lc3 = st.columns(3)

    with lc1:
        st.markdown("**Rider Productivity**")
        st.metric("Actual → Simulated",
                  f"{l1['sim_prod']:.2f} orders/trip",
                  delta=f"+{l1['prod_delta']:.2f} (+{l1['prod_delta_pct']:.1f}%)")
        with st.expander("Reasoning"):
            st.markdown(f"""
**Formula:** Total Orders / Total Trips

| Scenario | Trips | Productivity |
|---|---|---|
| Actual (no batching) | {l1['actual_trips']} | {l1['actual_trips']} / {l1['actual_trips']} = **{l1['actual_prod']:.2f}** |
| Simulated | {l1['sim_trips']} | {l1['actual_trips']} / {l1['sim_trips']} = **{l1['sim_prod']:.2f}** |

**Why:** {inp['total_batched']} orders grouped into {inp['n_batch_trips']} multi-order trips.
Trips reduced from {l1['actual_trips']} -> {l1['sim_trips']}, so each trip now serves more orders.
""")

    with lc2:
        st.markdown("**SLA Breach Rate**")
        breach_pp_change = l1['actual_breach_rate'] - l1['sim_breach_rate']
        st.metric("Actual → Simulated",
                  f"{l1['sim_breach_rate']:.1f}%",
                  delta=f"−{breach_pp_change:.1f}pp" if breach_pp_change > 0 else "0pp",
                  delta_color="inverse")
        with st.expander("Reasoning"):
            st.markdown(f"""
**Time-saving chain:**

| Step | Calculation | Result |
|---|---|---|
| Trips eliminated | {l1['actual_trips']} - {l1['sim_trips']} | **{l1['trips_saved']}** |
| Gross time saved | {l1['trips_saved']} x {inp['avg_o2r']:.1f} min | **{l2['gross_time_saved']:.0f} min** |
| Service penalty | {l2['extra_stops']} extra stops x {SERVICE_TIME_PENALTY_MIN} min | **-{l2['service_penalty']:.0f} min** |
| Net time saved | {l2['gross_time_saved']:.0f} - {l2['service_penalty']:.0f} | **{l2['net_time_saved']:.0f} min** |
| Effective for breach (x{BREACH_REDISTRIBUTION_EFF:.0%}) | {l2['net_time_saved']:.0f} x {BREACH_REDISTRIBUTION_EFF} | **{l2['effective_time']:.0f} min** |

**Breach recovery:**
- Current breaches: **{int(l1['breach_count'])}** ({l1['actual_breach_rate']:.1f}%)
- Avg breach delay: **{l2['avg_breach_delay']:.1f} min** (from per-order delay_minutes data)
- Effective time available: **{l2['effective_time']:.0f} min** (35% redistribution efficiency — not all freed time reaches breached orders)
- Recoverable: {l2['effective_time']:.0f} / {l2['avg_breach_delay']:.1f} = **{l1['breaches_prevented']}** breaches
- New rate: {l1['actual_breach_rate']:.1f}% x (1 - {l1['breaches_prevented']}/{max(int(l1['breach_count']), 1)}) = **{l1['sim_breach_rate']:.1f}%**

**Why 35% redistribution?** Freed rider time is system-wide — only a fraction
reaches the specific orders at risk of breaching. The 35% factor accounts for
queue position, rider assignment, and geographic mismatch.
""")

    with lc3:
        st.markdown("**Total Trips Required**")
        st.metric("Actual → Simulated",
                  f"{l1['sim_trips']}",
                  delta=f"−{l1['trips_saved']} (−{l1['trip_reduction_pct']:.1f}%)",
                  delta_color="inverse")
        with st.expander("Reasoning"):
            st.markdown(f"""
| Component | Count |
|---|---|
| Multi-order batch trips | {inp['n_batch_trips']} |
| Single-order trips | {result['single_trips']} |
| **Total simulated** | **{l1['sim_trips']}** |
| Baseline (1 per order) | {l1['actual_trips']} |
| **Trips saved** | **{l1['trips_saved']}** |

**Operational impact:**
- {l1['trip_reduction_pct']:.1f}% fewer rider dispatches
- Rider-minutes freed: {l1['trips_saved']} x {inp['avg_o2r']:.0f} = **{l1['trips_saved'] * inp['avg_o2r']:.0f} min** ({l1['trips_saved'] * inp['avg_o2r'] / 60:.1f} hrs)
{'- Rider ceiling applied: ' + str(int(rider_cap)) + ' available riders (' + str(int(avg_avail)) + '/hr x ' + str(n_hours) + ' hrs)' if result['capped'] else ''}
""")

    st.divider()

    st.subheader("L2 Metrics — Operational Detail")

    l2c1, l2c2, l2c3, l2c4, l2c5 = st.columns(5)
    l2c1.metric("Batching %", f"{l2['batching_pct']:.1f}%",
                help="Orders batched / Total orders")
    l2c2.metric("Avg Batch Size", f"{l2['avg_batch_size']:.1f}",
                help="Orders per multi-order trip")
    l2c3.metric("Gross Time Saved", f"{l2['gross_time_saved']:.0f} min")
    l2c4.metric("Service Overhead", f"{l2['service_penalty']:.0f} min",
                help=f"{l2['extra_stops']} extra stops x {SERVICE_TIME_PENALTY_MIN} min")
    l2c5.metric("Net Time Saved", f"{l2['net_time_saved']:.0f} min")

    with st.expander("L2 Reasoning: Full calculation chain", expanded=False):
        st.markdown(f"""
**Batching %** = {inp['total_batched']} batched / {inp['total_orders']} total = **{l2['batching_pct']:.1f}%**

This is determined by the actual Haversine distance matrix:
- {cb['pairs_within_both']:,} of {cb['n_pairs']:,} order pairs met BOTH R and T constraints ({cb['pct_within_both']:.1f}%)
- The greedy algorithm converted these eligible pairs into {inp['n_batch_trips']} batches

**Avg Batch Size** = {f"{inp['total_batched']} / {inp['n_batch_trips']} = **{l2['avg_batch_size']:.1f}**" if inp['n_batch_trips'] else "N/A (no multi-order batches)"} orders/batch

**Time calculation:**
1. Gross = {l1['trips_saved']} saved trips x {inp['avg_o2r']:.1f} min avg trip time = **{l2['gross_time_saved']:.0f} min**
2. Penalty = {l2['extra_stops']} extra stops x {SERVICE_TIME_PENALTY_MIN} min = **{l2['service_penalty']:.0f} min**
3. Net = {l2['gross_time_saved']:.0f} - {l2['service_penalty']:.0f} = **{l2['net_time_saved']:.0f} min**

**{SERVICE_TIME_PENALTY_MIN}-min service penalty:** Each additional stop in a batch adds ~{SERVICE_TIME_PENALTY_MIN} min
for parking, navigation to door, OTP verification, and handoff.
""")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  SENSITIVITY ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    st.subheader("Parameter Sensitivity")
    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        st.markdown("**Batching % vs Radius**")
        data_r = []
        for r in np.arange(0.5, 5.25, 0.5):
            res = run_batching(sim_orders, r, T_min, S_max, rider_cap)
            pct = res["batched_count"] / total_n * 100 if total_n else 0
            data_r.append({"Radius (km)": r, "Batching %": pct,
                           "Trips": res["total_trips"]})
        df_r = pd.DataFrame(data_r)
        fig_r = px.line(df_r, x="Radius (km)", y="Batching %",
                        markers=True, color_discrete_sequence=["#6366f1"])
        fig_r.add_vline(x=R_km, line_dash="dash", line_color="red",
                        annotation_text=f"R={R_km}")
        fig_r.update_layout(height=280)
        st.plotly_chart(fig_r, use_container_width=True)

    with sc2:
        st.markdown("**Batching % vs Time Window**")
        data_t = []
        for t in range(3, 21):
            res = run_batching(sim_orders, R_km, t, S_max, rider_cap)
            pct = res["batched_count"] / total_n * 100 if total_n else 0
            data_t.append({"Window (min)": t, "Batching %": pct})
        df_t = pd.DataFrame(data_t)
        fig_t = px.line(df_t, x="Window (min)", y="Batching %",
                        markers=True, color_discrete_sequence=["#f59e0b"])
        fig_t.add_vline(x=T_min, line_dash="dash", line_color="red",
                        annotation_text=f"T={T_min}")
        fig_t.update_layout(height=280)
        st.plotly_chart(fig_t, use_container_width=True)

    with sc3:
        st.markdown("**Trips vs Batch Size**")
        data_s = []
        for s in range(2, 6):
            res = run_batching(sim_orders, R_km, T_min, s, rider_cap)
            data_s.append({"Max Batch Size": s, "Trips": res["total_trips"],
                           "Batching %": res["batched_count"] / total_n * 100 if total_n else 0})
        df_s = pd.DataFrame(data_s)
        fig_s = px.bar(df_s, x="Max Batch Size", y="Trips",
                       text="Batching %", color_discrete_sequence=["#10b981"])
        fig_s.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig_s.update_layout(height=280)
        st.plotly_chart(fig_s, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("🚚 Logistics Batching Optimizer")
    st.caption("FirstClub Quick Commerce — Historical Performance Audit & What-If Simulation")

    data_source = None

    if DEFAULT_DATA_PATH.exists():
        data_source = str(DEFAULT_DATA_PATH)
    else:
        st.info("Upload your **Batching Analysis** Excel file to get started.")
        uploaded = st.file_uploader(
            "Upload Excel file", type=["xlsx"],
            help="Expected sheets: orders, breach, Rider Availability, Distance, Distance Buckets, WH Coordinates, Orders Raw",
        )
        if uploaded:
            data_source = uploaded
        else:
            st.stop()

    orders, breach, rider, distance, buckets, wh, raw, global_breach_delay = load_data(data_source)

    tab1, tab2 = st.tabs(["📊 Descriptive Analytics", "🔬 Batching Simulator"])

    with tab1:
        render_analytics(breach, rider, distance, buckets, wh)

    with tab2:
        render_simulator(raw, rider, breach, wh, global_breach_delay)


if __name__ == "__main__":
    main()
