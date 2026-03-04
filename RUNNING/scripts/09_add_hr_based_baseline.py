import numpy as np
import pandas as pd

MARATHON_KM = 42.195
AGE = 24

# --- inputs ---
runs = pd.read_csv("marathon_runs_16w.csv")
snap = pd.read_csv("marathon_snapshot_16w.csv").iloc[0].to_dict()
preds = pd.read_csv("marathon_predictions_oneshot.csv")

# --- clean ---
runs["distance_km"] = pd.to_numeric(runs["distance_km"], errors="coerce")
runs["duration_min"] = pd.to_numeric(runs["duration_min"], errors="coerce")
runs["avg_heart_rate"] = pd.to_numeric(runs.get("avg_heart_rate"), errors="coerce")
runs = runs.dropna(subset=["distance_km", "duration_min", "avg_heart_rate"]).copy()

runs["pace_min_km"] = runs["duration_min"] / runs["distance_km"]

# garder plutôt les runs "steady" (évite que les fractionnés explosent la relation HR->pace)
train = runs[
    (runs["distance_km"].between(5, 25, inclusive="both")) &
    (runs["pace_min_km"].between(4.2, 7.5, inclusive="both")) &
    (runs["avg_heart_rate"].between(120, 190, inclusive="both"))
].copy()

if len(train) < 8:
    raise ValueError(f"Pas assez de runs pour entraîner le modèle HR->pace (train rows={len(train)}).")

# --- estimate HRmax ---
hrmax_tanaka = 208 - 0.7 * AGE  # [web:371] (citation in your writeup, not in code)
hrmax_data = float(np.nanmax(pd.to_numeric(runs.get("max_heart_rate"), errors="coerce"))) if "max_heart_rate" in runs.columns else np.nan

# HRmax final = max(tanaka, observed) si observed existe, sinon tanaka
hrmax_est = hrmax_tanaka if not np.isfinite(hrmax_data) else max(hrmax_tanaka, hrmax_data)

# target marathon HR band
hr_low = 0.82 * hrmax_est
hr_mid = 0.85 * hrmax_est
hr_high = 0.88 * hrmax_est

# --- "ML" model: pace = a*HR + b ---
x = train["avg_heart_rate"].to_numpy()
y = train["pace_min_km"].to_numpy()
a, b = np.polyfit(x, y, 1)

pace_at_hr_mid = float(a * hr_mid + b)
pred_min_raw = pace_at_hr_mid * MARATHON_KM

# endurance penalty (simple)
total_km_16w = float(snap.get("total_km_16w"))
long_run = float(snap.get("max_long_run_km"))
penalty = 0.0
if np.isfinite(long_run) and long_run < 28:
    penalty += 0.04
if np.isfinite(long_run) and long_run < 24:
    penalty += 0.03
if np.isfinite(total_km_16w) and total_km_16w < 450:
    penalty += 0.03

pred_min = pred_min_raw * (1.0 + penalty)

row = {
    "method": "hr_pace_linear_model",
    "pred_min": pred_min,
    "pred_hhmmss": "",
    "p10_min": np.nan,
    "p10_hhmmss": "",
    "p90_min": np.nan,
    "p90_hhmmss": "",
    "details": (
        f"HRmax_est={hrmax_est:.1f} (tanaka={hrmax_tanaka:.1f}, data={hrmax_data:.1f}); "
        f"HR_target_mid={hr_mid:.1f} (band {hr_low:.1f}-{hr_high:.1f}); "
        f"pace(HR)=a*HR+b with a={a:.5f}, b={b:.3f}; "
        f"pace@HRmid={pace_at_hr_mid:.3f}; penalty={penalty*100:.1f}%"
    ),
}

# compute hh:mm:ss
s = int(round(float(row["pred_min"]) * 60))
row["pred_hhmmss"] = f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

preds2 = pd.concat([preds, pd.DataFrame([row])], ignore_index=True)
preds2.to_csv("marathon_predictions_oneshot.csv", index=False)

print("OK -> marathon_predictions_oneshot.csv (updated)")
print("Added:", row["method"], row["pred_min"], row["pred_hhmmss"])
print("Details:", row["details"])
