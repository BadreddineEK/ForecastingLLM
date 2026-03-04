import pandas as pd

ATHLETE_PROFILE = "Age=24; WeightKg=73; Sex=unknown"
RACE_NAME = "Run In Lyon Marathon"
RACE_DATE = "2025-10-05"
J0 = "2025-10-05"
LAST_ALLOWED = "2025-10-04"  # anti-fuite

snap = pd.read_csv("marathon_snapshot_16w.csv").iloc[0].to_dict()
weekly = pd.read_csv("marathon_weekly_16w.csv")
runs = pd.read_csv("marathon_runs_16w.csv")

# arrondir un peu pour éviter des floats moches
weekly_round = weekly.copy()
for c in ["km", "duration_min", "elev_gain_m", "long_run_km", "pace_wavg_min_km", "avg_hr_wavg"]:
    if c in weekly_round.columns:
        weekly_round[c] = weekly_round[c].round(3)

runs_round = runs.copy()
for c in ["distance_km", "duration_min", "pace_min_km", "avg_heart_rate", "max_heart_rate", "avg_running_cadence", "total_ascent", "avg_temperature"]:
    if c in runs_round.columns:
        runs_round[c] = pd.to_numeric(runs_round[c], errors="coerce").round(3)

weekly_csv = weekly_round.to_csv(index=False)
runs_csv = runs_round.to_csv(index=False)

prompt = f"""Web OFF. One-shot evaluation.
Ignore all previous conversation history. Use ONLY the data below.

Task: Predict marathon finish time for the race on {RACE_DATE} ({RACE_NAME}).
You must respect the anti-leak rule: you can only use training data with date <= {LAST_ALLOWED}. Any info after {J0} is forbidden.

Return STRICT JSON only (no extra text, no markdown):
{{"time_p10_min": number, "time_p50_min": number, "time_p90_min": number, "justification": "max 2 sentences, no line breaks"}}
Rules:
- Ensure time_p10_min < time_p50_min < time_p90_min.
- Times must be plausible marathon finish times in minutes.
- justification: 2 sentences max, no quotes (") inside.

ATHLETE_PROFILE: {ATHLETE_PROFILE}

TRAINING_SNAPSHOT_SUMMARY (16 weeks ending {LAST_ALLOWED}):
- n_runs_16w: {snap.get("n_runs_16w")}
- total_km_16w: {snap.get("total_km_16w")}
- total_duration_h_16w: {snap.get("total_duration_h_16w")}
- max_week_km: {snap.get("max_week_km")}
- max_long_run_km: {snap.get("max_long_run_km")}
- pace_16w_wavg_min_km: {snap.get("pace_16w_wavg_min_km")}
- avg_hr_16w_wavg: {snap.get("avg_hr_16w_wavg")}

WEEKLY_TABLE_CSV:
{weekly_csv}

RUNS_TABLE_CSV:
{runs_csv}
"""

with open("marathon_llm_prompt_oneshot.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

print("OK -> marathon_llm_prompt_oneshot.txt")
print(prompt[:700])
