import pandas as pd
from datetime import date, timedelta

# Paramètres de l'étude
cities = ["Lyon", "Paris", "London", "New York", "Sao Paulo"]
start = date(2025, 12, 18)
end = date(2025, 12, 24)
dates = pd.date_range(start, end, freq="D").date

# Deux targets météo (continu, quantiles plus tard)
targets = ["TMAX", "RAIN"]

rows = []
for city in cities:
    for d in dates:
        for target in targets:
            rows.append({
                "city": city,
                "date_local": d.isoformat(),                         # jour à prédire
                "target": target,                                   # TMAX ou RAIN
                "snapshot_start": (d - timedelta(days=7)).isoformat(),# J-7
                "snapshot_end": (d - timedelta(days=1)).isoformat(),  # J-1
                "prompt_text": "",                                  # à remplir après
                "web_mode": ""                                      # ON/OFF (tu dupliqueras plus tard)
            })

df = pd.DataFrame(rows)
df.to_csv("weather_question_bank.csv", index=False)
print("OK:", len(df), "rows -> weather_question_bank.csv")

