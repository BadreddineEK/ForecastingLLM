import pandas as pd
race = pd.Timestamp("2025-10-05")
start = (race - pd.Timedelta(weeks=16)).normalize()
end = (race - pd.Timedelta(days=1)).normalize()

df = pd.read_csv("activities_session_241_fixed.csv")
df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True).dt.tz_convert(None)  # [web:173]

w = df[(df["start_time"] >= start) & (df["start_time"] <= end)]
print("rows in 16w window (all sports):", len(w))
print(w["sport"].astype(str).str.lower().value_counts().head(10))
