import pandas as pd
import numpy as np

MARATHON_KM = 42.195
RIEGEL_EXP = 1.06  # exponent standard de Riegel [web:253]

runs = pd.read_csv("marathon_runs_16w.csv")
snap = pd.read_csv("marathon_snapshot_16w.csv").iloc[0].to_dict()

runs["start_time"] = pd.to_datetime(runs["start_time"], errors="coerce")
runs = runs.dropna(subset=["distance_km", "duration_min"]).copy()

runs["time_min"] = runs["duration_min"]
runs["pace_min_km"] = runs["time_min"] / runs["distance_km"]

def riegel_predict_min(t1_min, d1_km, d2_km=MARATHON_KM, exp=RIEGEL_EXP):
    return float(t1_min * (d2_km / d1_km) ** exp)

def pick_best_effort(df, dmin, dmax):
    cand = df[(df["distance_km"] >= dmin) & (df["distance_km"] <= dmax)].copy()
    if len(cand) == 0:
        return None
    return cand.sort_values("pace_min_km", ascending=True).iloc[0]

# --- choisir une référence (du plus “marathon-like” au plus court) ---
candidates = [
    pick_best_effort(runs, 20, 23),     # ~20-23 km
    pick_best_effort(runs, 14, 17),     # ~15 km
    pick_best_effort(runs, 9, 11),      # ~10 km
    pick_best_effort(runs, 4.5, 6.0),   # ~5 km
]
ref = next((x for x in candidates if x is not None), None)

if ref is None:
    raise ValueError("Aucun run exploitable pour baseline (vérifie marathon_runs_16w.csv).")

ref_d = float(ref["distance_km"])
ref_t = float(ref["time_min"])

# 1) Baseline Riegel “pure”
pred_riegel_pure = riegel_predict_min(ref_t, ref_d)

# 2) Correction marathon (heuristique simple)
long_run = float(snap.get("max_long_run_km", np.nan))
total_km = float(snap.get("total_km_16w", np.nan))

penalty = 0.0
if np.isfinite(long_run) and long_run < 28:
    penalty += 0.04
if np.isfinite(long_run) and long_run < 24:
    penalty += 0.03
if np.isfinite(total_km) and total_km < 450:
    penalty += 0.03

pred_riegel_corrected = pred_riegel_pure * (1.0 + penalty)

out = pd.DataFrame([
    {
        "method": "riegel_pure",
        "ref_distance_km": ref_d,
        "ref_time_min": ref_t,
        "pred_marathon_min": round(pred_riegel_pure, 2),
        "penalty_pct": 0.0,
    },
    {
        "method": "riegel_corrected",
        "ref_distance_km": ref_d,
        "ref_time_min": ref_t,
        "pred_marathon_min": round(pred_riegel_corrected, 2),
        "penalty_pct": round(penalty * 100, 1),
    },
])

out.to_csv("baseline_predictions.csv", index=False)

print("OK -> baseline_predictions.csv")
print(out)
print("Reference run picked:")
print(ref[["start_time", "distance_km", "duration_min", "pace_min_km"]])
