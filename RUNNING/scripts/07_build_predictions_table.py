import json
import numpy as np
import pandas as pd

MARATHON_KM = 42.195

# ========= INPUTS =========
RUNS_CSV = "marathon_runs_16w.csv"
SNAP_CSV = "marathon_snapshot_16w.csv"
BASELINES_CSV = "baseline_predictions.csv"

# Vérité terrain / Coros (stockées dans le CSV final)
TRUE_TIME_MIN = 223   # 3h43
COROS_PRED_MIN = 231  # 3h51

# Réponse LLM (coller telle quelle)
LLM_JSON_STR = r'''{"time_p10_min":237,"time_p50_min":244,"time_p90_min":253,"justification":"Prediction is anchored on the best 20+ km training pace (21.379 km at 5.349 min/km) and adjusted slower by ~5–12% to reflect marathon fatigue given a longest long run of 25.01 km and peak week of 53.12 km. The uncertainty band reflects variability from a relatively moderate 16-week volume (417.86 km across 35 runs) and limited marathon-specific long-run exposure."}'''

# ========= helpers =========
def min_to_hhmmss(x_min: float) -> str:
    if x_min is None or (isinstance(x_min, float) and np.isnan(x_min)):
        return ""
    s = int(round(float(x_min) * 60))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

def safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

# ========= load =========
runs = pd.read_csv(RUNS_CSV)
snap = pd.read_csv(SNAP_CSV).iloc[0].to_dict()
base = pd.read_csv(BASELINES_CSV)

runs["start_time"] = pd.to_datetime(runs["start_time"], errors="coerce")
runs = runs.dropna(subset=["distance_km", "duration_min"]).copy()
runs["distance_km"] = pd.to_numeric(runs["distance_km"], errors="coerce")
runs["duration_min"] = pd.to_numeric(runs["duration_min"], errors="coerce")
runs = runs.dropna(subset=["distance_km", "duration_min"]).copy()

runs["pace_min_km"] = runs["duration_min"] / runs["distance_km"]

total_km_16w = safe_float(snap.get("total_km_16w"))
max_long_run_km = safe_float(snap.get("max_long_run_km"))

# ========= method: Riegel from baseline_predictions.csv =========
b = base.set_index("method")["pred_marathon_min"].to_dict()
pred_riegel_pure = safe_float(b.get("riegel_pure"))
pred_riegel_corr = safe_float(b.get("riegel_corrected"))

# ========= method: Heuristic based on best 20+km training pace =========
# Pick fastest run among distance>=20km and use a fatigue factor depending on long_run + volume.
runs_20 = runs[runs["distance_km"] >= 20].copy()
if len(runs_20) > 0:
    ref = runs_20.sort_values("pace_min_km", ascending=True).iloc[0]
    ref_pace = float(ref["pace_min_km"])
    ref_dist = float(ref["distance_km"])
    ref_time = float(ref["duration_min"])
else:
    # fallback: best 15-20km, else best 10-15
    ref = None
    for lo, hi in [(15, 20), (10, 15)]:
        cand = runs[(runs["distance_km"] >= lo) & (runs["distance_km"] < hi)].copy()
        if len(cand) > 0:
            ref = cand.sort_values("pace_min_km", ascending=True).iloc[0]
            break
    if ref is None:
        ref = runs.sort_values("pace_min_km", ascending=True).iloc[0]
    ref_pace = float(ref["pace_min_km"])
    ref_dist = float(ref["distance_km"])
    ref_time = float(ref["duration_min"])

# fatigue factor (simple, transparent)
# base 1.06 (marathon slower than strong long run), + penalties if long_run/volume are low
fatigue = 1.06
if np.isfinite(max_long_run_km) and max_long_run_km < 28:
    fatigue += 0.03
if np.isfinite(max_long_run_km) and max_long_run_km < 24:
    fatigue += 0.02
if np.isfinite(total_km_16w) and total_km_16w < 450:
    fatigue += 0.02

pred_pace_factor = MARATHON_KM * ref_pace * fatigue

# ========= method: "ML" learned exponent (extended Riegel via log-log regression) =========
# Fit: log(time) = a + b*log(distance) over your fastest runs (to approximate max-effort relation).
# Then predict at marathon distance.
fast = runs[(runs["distance_km"] >= 3) & (runs["distance_km"] <= 25)].copy()
# keep top N fastest by pace to reduce easy jog noise
fast = fast.sort_values("pace_min_km", ascending=True).head(20)

if len(fast) >= 5:
    x = np.log(fast["distance_km"].to_numpy())
    y = np.log(fast["duration_min"].to_numpy())
    b_hat, a_hat = np.polyfit(x, y, 1)  # y = b*x + a
    pred_extended_riegel = float(np.exp(a_hat) * (MARATHON_KM ** b_hat))
else:
    b_hat, a_hat = np.nan, np.nan
    pred_extended_riegel = np.nan

# ========= LLM =========
llm = json.loads(LLM_JSON_STR)

# ========= output table =========
rows = []

def add(method, pred_min, p10=np.nan, p90=np.nan, details=""):
    rows.append({
        "method": method,
        "pred_min": safe_float(pred_min),
        "pred_hhmmss": min_to_hhmmss(pred_min) if np.isfinite(safe_float(pred_min)) else "",
        "p10_min": safe_float(p10),
        "p10_hhmmss": min_to_hhmmss(p10) if np.isfinite(safe_float(p10)) else "",
        "p90_min": safe_float(p90),
        "p90_hhmmss": min_to_hhmmss(p90) if np.isfinite(safe_float(p90)) else "",
        "details": details,
    })

# include ground truth / coros as "methods" for easy comparison later
add("true_official", TRUE_TIME_MIN, details="Ground truth (evaluation only)")
add("coros_pred", COROS_PRED_MIN, details="Coros prediction before race")

add("llm_p50", llm["time_p50_min"], p10=llm["time_p10_min"], p90=llm["time_p90_min"], details=llm["justification"])

add("riegel_pure", pred_riegel_pure, details="From baseline_predictions.csv")
add("riegel_corrected", pred_riegel_corr, details="From baseline_predictions.csv (penalty on long_run/volume)")

add("heuristic_best_longrun_pace_factor",
    pred_pace_factor,
    details=f"Ref run: {ref_dist:.2f}km in {ref_time:.2f}min (pace {ref_pace:.3f} min/km). Fatigue factor={fatigue:.3f}")

add("extended_riegel_loglog_fit",
    pred_extended_riegel,
    details=f"Fitted on top-20 fastest runs (3-25km). exponent_b={b_hat:.4f}")

out = pd.DataFrame(rows)
out.to_csv("marathon_predictions_oneshot.csv", index=False)

print("OK -> marathon_predictions_oneshot.csv")
print(out[["method","pred_min","pred_hhmmss","details"]])
