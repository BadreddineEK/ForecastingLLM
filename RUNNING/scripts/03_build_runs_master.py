import pandas as pd
import numpy as np

RACE_DATE = pd.Timestamp("2025-10-05")
J0_LAST_ALLOWED = RACE_DATE - pd.Timedelta(days=1)  # 2025-10-04

INPUT = "activities_session_241_fixed.csv"  # IMPORTANT: on lit le fixed

df = pd.read_csv(INPUT)

# Parse dates (gérer éventuel timezone en mode robuste)
df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

# Passer en tz-naive UTC (option simple)
df["start_time"] = df["start_time"].dt.tz_convert(None)   # convert UTC then drop tz [web:173]
df["timestamp"] = df["timestamp"].dt.tz_convert(None)     # convert UTC then drop tz [web:173]

# Durée: prioriser total_timer_time, sinon total_elapsed_time
df["duration_s"] = df.get("total_timer_time")
df.loc[df["duration_s"].isna(), "duration_s"] = df.get("total_elapsed_time")

# Distance
df["distance_km"] = df.get("total_distance") / 1000.0

# Durée minutes
df["duration_min"] = df["duration_s"] / 60.0

# Pace min/km
df["pace_min_km"] = df["duration_min"] / df["distance_km"]
df.loc[(df["distance_km"].isna()) | (df["distance_km"] <= 0), "pace_min_km"] = np.nan

# Normaliser sport/sub_sport
df["sport"] = df["sport"].astype(str).str.lower()
df["sub_sport"] = df["sub_sport"].astype(str).str.lower()

# Filtre run-like + plausible
is_run_like = (
    df["sport"].str.contains("run", na=False)
    | df["sub_sport"].str.contains("run", na=False)
)

is_plausible_run = (
    (df["distance_km"] >= 0.5)
    & (df["duration_min"] > 0)
    & (df["pace_min_km"].between(2.5, 12.0, inclusive="both"))
)

runs = df[is_run_like & is_plausible_run].copy()

keep_cols = [
    "file_name", "start_time", "timestamp", "sport", "sub_sport",
    "distance_km", "duration_min", "pace_min_km",
    "total_ascent", "total_descent",
    "avg_heart_rate", "max_heart_rate", "min_heart_rate",
    "avg_running_cadence", "max_running_cadence",
    "enhanced_avg_speed", "enhanced_max_speed",
    "avg_temperature",
    "total_calories", "total_strides",
]
keep_cols = [c for c in keep_cols if c in runs.columns]
runs = runs[keep_cols].sort_values("start_time")

runs.to_csv("runs_master_v0.csv", index=False)

runs_pre = runs[runs["start_time"] <= J0_LAST_ALLOWED].copy()
runs_pre.to_csv("runs_master_pre_marathon.csv", index=False)

print("INPUT:", INPUT)
print("ALL activities:", len(df))
print("RUNS (all time):", len(runs))
print("RUNS pre-marathon (<= 2025-10-04):", len(runs_pre))
print("Date range runs_pre:", runs_pre["start_time"].min(), "->", runs_pre["start_time"].max())
print(runs_pre[["start_time", "distance_km", "duration_min", "pace_min_km"]].head(5))
