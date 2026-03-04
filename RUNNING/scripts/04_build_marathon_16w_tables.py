import pandas as pd
import numpy as np

RACE_DATE = pd.Timestamp("2025-10-05")
WINDOW_WEEKS = 16

WINDOW_START = (RACE_DATE - pd.Timedelta(weeks=WINDOW_WEEKS)).normalize()
WINDOW_END = (RACE_DATE - pd.Timedelta(days=1)).normalize()  # anti-fuite: J0-1

RUNS_FILE = "runs_master_pre_marathon.csv"

# ---------- helpers ----------
def wavg(values, weights):
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    m = v.notna() & w.notna() & (w > 0)
    if m.sum() == 0:
        return np.nan
    return float((v[m] * w[m]).sum() / w[m].sum())

# ---------- load ----------
runs = pd.read_csv(RUNS_FILE)
runs["start_time"] = pd.to_datetime(runs["start_time"], errors="coerce")
runs = runs.dropna(subset=["start_time"]).sort_values("start_time")

# ---------- filter 16w ----------
w = runs[(runs["start_time"] >= WINDOW_START) & (runs["start_time"] <= WINDOW_END)].copy()
w = w[(w["distance_km"] > 0) & (w["duration_min"] > 0)].copy()

# convert duration for weighting
w["duration_s"] = w["duration_min"] * 60.0

# ---------- export runs table (the thing you want to give to LLM) ----------
# (keep it reasonably small: only the key cols)
keep_cols = [
    "start_time",
    "distance_km",
    "duration_min",
    "pace_min_km",
    "avg_heart_rate",
    "max_heart_rate",
    "avg_running_cadence",
    "total_ascent",
    "avg_temperature",
    "file_name",
]
keep_cols = [c for c in keep_cols if c in w.columns]
runs_16w = w[keep_cols].copy().sort_values("start_time")
runs_16w.to_csv("marathon_runs_16w.csv", index=False)

# ---------- weekly aggregation ----------
# resample needs datetime index [web:144]
wx = w.set_index("start_time")

# choose a fixed weekly index so missing weeks appear as 0
weekly_index = pd.date_range(WINDOW_START, WINDOW_END, freq="W-SUN")
weekly = pd.DataFrame(index=weekly_index)

weekly["km"] = wx["distance_km"].resample("W-SUN").sum().reindex(weekly_index, fill_value=0.0)
weekly["duration_min"] = wx["duration_min"].resample("W-SUN").sum().reindex(weekly_index, fill_value=0.0)
weekly["sessions"] = wx["distance_km"].resample("W-SUN").count().reindex(weekly_index, fill_value=0).astype(int)

if "total_ascent" in wx.columns:
    weekly["elev_gain_m"] = pd.to_numeric(wx["total_ascent"], errors="coerce").resample("W-SUN").sum().reindex(weekly_index, fill_value=0.0)
else:
    weekly["elev_gain_m"] = 0.0

weekly["long_run_km"] = wx["distance_km"].resample("W-SUN").max().reindex(weekly_index)

# pace weighted by distance (more stable than mean pace) [web:112]
tmp = wx.copy()
tmp["pace_x_km"] = pd.to_numeric(tmp["pace_min_km"], errors="coerce") * pd.to_numeric(tmp["distance_km"], errors="coerce")
agg = tmp.resample("W-SUN")[["pace_x_km", "distance_km"]].sum()
weekly["pace_wavg_min_km"] = (agg["pace_x_km"] / agg["distance_km"]).reindex(weekly_index)

# HR weighted by duration
if "avg_heart_rate" in wx.columns:
    tmp2 = wx.copy()
    tmp2["hr_x_s"] = pd.to_numeric(tmp2["avg_heart_rate"], errors="coerce") * pd.to_numeric(tmp2["duration_s"], errors="coerce")
    agg2 = tmp2.resample("W-SUN")[["hr_x_s", "duration_s"]].sum()
    weekly["avg_hr_wavg"] = (agg2["hr_x_s"] / agg2["duration_s"]).reindex(weekly_index)
else:
    weekly["avg_hr_wavg"] = np.nan

weekly = weekly.reset_index().rename(columns={"index": "week_end"})
weekly.to_csv("marathon_weekly_16w.csv", index=False)

# ---------- 1-line snapshot ----------
snapshot = {
    "race_date": str(RACE_DATE.date()),
    "train_window_start": str(WINDOW_START.date()),
    "train_window_end": str(WINDOW_END.date()),
    "n_runs_16w": int(len(runs_16w)),
    "total_km_16w": float(runs_16w["distance_km"].sum()),
    "total_duration_h_16w": float(runs_16w["duration_min"].sum() / 60.0),
    "max_week_km": float(weekly["km"].max()),
    "max_long_run_km": float(runs_16w["distance_km"].max()),
    "pace_16w_wavg_min_km": wavg(runs_16w["pace_min_km"], runs_16w["distance_km"]),
    "avg_hr_16w_wavg": wavg(runs_16w.get("avg_heart_rate", np.nan), runs_16w["duration_min"]),
}

pd.DataFrame([snapshot]).to_csv("marathon_snapshot_16w.csv", index=False)

# ---------- prints for validation ----------
print("Window:", WINDOW_START.date(), "->", WINDOW_END.date())
print("Runs in window:", len(runs_16w))
print("OK -> marathon_runs_16w.csv", runs_16w.shape)
print("OK -> marathon_weekly_16w.csv", weekly.shape)
print("OK -> marathon_snapshot_16w.csv (1,", len(snapshot), ")")
print(pd.DataFrame([snapshot]).T)
print(weekly.head(6))
