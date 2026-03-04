import pandas as pd
import numpy as np

RACE_DATE = pd.Timestamp("2025-10-05")
WINDOW_START = (RACE_DATE - pd.Timedelta(weeks=16)).normalize()
WINDOW_END = (RACE_DATE - pd.Timedelta(days=1)).normalize()

df = pd.read_csv("activities_session_241.csv")
df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")

df["duration_s"] = df.get("total_timer_time")
df.loc[df["duration_s"].isna(), "duration_s"] = df.get("total_elapsed_time")
df["distance_km"] = df.get("total_distance") / 1000.0
df["duration_min"] = df["duration_s"] / 60.0
df["pace_min_km"] = df["duration_min"] / df["distance_km"]
df.loc[(df["distance_km"] <= 0) | (~np.isfinite(df["pace_min_km"])), "pace_min_km"] = np.nan

w = df[(df["start_time"] >= WINDOW_START) & (df["start_time"] <= WINDOW_END)].copy()

cols = [c for c in ["start_time","sport","sub_sport","distance_km","duration_min","pace_min_km","avg_heart_rate","file_name"] if c in w.columns]
w = w.sort_values("start_time")[cols]

w.to_csv("window_activities_16w_from_export.csv", index=False)

print("Window:", WINDOW_START.date(), "->", WINDOW_END.date())
print("Activities in window (all sports):", len(w))
print(w["sport"].value_counts(dropna=False))
print(w.head(30))

