import pandas as pd
import numpy as np

df = pd.read_csv("marathon_predictions_oneshot.csv")

true_row = df[df["method"] == "true_official"].iloc[0]
true_min = float(true_row["pred_min"])

df["err_signed_min"] = df["pred_min"] - true_min
df["err_abs_min"] = (df["pred_min"] - true_min).abs()

# indicateur "interval covers true" pour le LLM (si P10/P90 présents)
df["covers_true"] = np.where(
    df["p10_min"].notna() & df["p90_min"].notna(),
    (true_min >= df["p10_min"]) & (true_min <= df["p90_min"]),
    np.nan
)

df.to_csv("marathon_predictions_oneshot_scored.csv", index=False)
print("OK -> marathon_predictions_oneshot_scored.csv")
print(df[["method","pred_min","pred_hhmmss","err_signed_min","err_abs_min","covers_true"]])
