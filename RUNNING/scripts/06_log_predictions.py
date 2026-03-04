import json
import pandas as pd

# Vérité terrain (sert UNIQUEMENT à l'évaluation)
TRUE_TIME_MIN = 223   # 3h43
COROS_PRED_MIN = 231  # 3h51

LLM_JSON_STR = r'''{"time_p10_min":237,"time_p50_min":244,"time_p90_min":253,"justification":"Prediction is anchored on the best 20+ km training pace (21.379 km at 5.349 min/km) and adjusted slower by ~5–12% to reflect marathon fatigue given a longest long run of 25.01 km and peak week of 53.12 km. The uncertainty band reflects variability from a relatively moderate 16-week volume (417.86 km across 35 runs) and limited marathon-specific long-run exposure."}'''
llm = json.loads(LLM_JSON_STR)

row = {
    "race_name": "Run In Lyon Marathon",
    "race_date": "2025-10-05",
    "j0_cutoff": "2025-10-04",

    "true_time_min": TRUE_TIME_MIN,
    "coros_pred_min": COROS_PRED_MIN,

    "llm_p10_min": llm["time_p10_min"],
    "llm_p50_min": llm["time_p50_min"],
    "llm_p90_min": llm["time_p90_min"],
    "llm_justification": llm["justification"],
}

pd.DataFrame([row]).to_csv("experiments_marathon_oneshot.csv", index=False)
print("OK -> experiments_marathon_oneshot.csv")
