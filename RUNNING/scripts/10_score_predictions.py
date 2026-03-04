import pandas as pd
import numpy as np

df = pd.read_csv("marathon_predictions_oneshot.csv")

true = float(df.loc[df["method"]=="true_official","pred_min"].iloc[0])

df["err_signed_min"] = df["pred_min"] - true
df["err_abs_min"] = (df["pred_min"] - true).abs()

# Couverture P10–P90 (utile seulement pour llm_p50)
df["covers_true"] = np.where(
    df["p10_min"].notna() & df["p90_min"].notna(),
    (true >= df["p10_min"]) & (true <= df["p90_min"]),
    np.nan
)

# tri par erreur absolue, en laissant la ligne true_official en haut
df["sort_key"] = np.where(df["method"]=="true_official", -1, df["err_abs_min"])
df = df.sort_values("sort_key").drop(columns=["sort_key"])

df.to_csv("marathon_predictions_oneshot_scored.csv", index=False)

print("OK -> marathon_predictions_oneshot_scored.csv")
print(df[["method","pred_hhmmss","pred_min","err_signed_min","err_abs_min","covers_true"]])
