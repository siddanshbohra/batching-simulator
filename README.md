# Logistics Batching Optimizer

**FirstClub Quick Commerce** — Historical performance audit & what-if simulation for delivery batching.

Determines how changes in batching logic (radius, time windows, batch size) would have historically impacted rider productivity and delivery breach rates.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

## Data

Place your `Batching Analysis.xlsx` file in the `data/` folder. If no file is found, the app shows a file uploader.

**Expected Excel sheets:**

| Sheet | Purpose |
|---|---|
| `orders` | Per-order timestamps, breach status, delay minutes |
| `Orders Raw` | Per-order delivery lat/lng, distances, timestamps |
| `breach` | Hourly hub-level breach aggregates |
| `Rider Availability` | Hourly hub-level rider counts |
| `Distance` | Order-level distance data |
| `Distance Buckets` | Aggregated distance bucket breakdown |
| `WH Coordinates` | Warehouse ID, name, lat/lng |

## Features

### Descriptive Analytics
- Warehouse breach heatmap (OpenStreetMap)
- Rider utilisation vs. hour-of-day curves
- Order volume by distance bucket
- Hourly breach rate by hub (heatmap)

### Batching Simulator
- **Greedy Proximity Algorithm** using actual delivery coordinates
- Adjustable parameters: radius (0.5–5 km), time window (5–20 min), max batch size (2–5)
- Rider capacity ceiling from actual availability data
- Full constraint funnel visibility (distance pairs, time pairs, both)

### Metrics
- **L1 (North-Star):** Rider productivity, SLA breach rate, total trips — with detailed reasoning
- **L2 (Operational):** Batching %, avg batch size, gross/net time saved, service overhead
- **Sensitivity Analysis:** Batching % and trips vs. radius, time window, and batch size

## Technical Details

- Vectorised Haversine distance matrix (NumPy)
- Breach impact model with 3-min service penalty per extra stop
- Breach delay estimation from per-order data (not aggregate O2R — avoids ~7x overestimation)
- `@st.cache_data` for data loading performance

## Stack

Python 3.9+ · Streamlit · Pandas · NumPy · Plotly · openpyxl
