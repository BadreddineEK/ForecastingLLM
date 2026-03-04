import pandas as pd
import requests
from datetime import date, timedelta
import time

# -----------------------
# CONFIG
# -----------------------
INPUT_CSV = "weather_question_bank.csv"        # ton fichier existant (70 lignes)
OUT_BASE = "weather_prompts_base.csv"          # 35 lignes (1 prompt par ville/jour)
OUT_EXPANDED = "weather_prompts_expanded.csv"  # 70 lignes (ON + OFF)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"  # [web:203]
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"   # [web:119]

# Pour la reproductibilité : fixe la date de run (J0)
# Comme on est le 17/12/2025 dans ton protocole, on fige J0 = 2025-12-17.
RUN_DATE = date(2025, 12, 17)

# Snapshot PRO : 7 derniers jours complets disponibles au moment du forecast
SNAPSHOT_END = RUN_DATE - timedelta(days=1)    # 2025-12-16
SNAPSHOT_START = SNAPSHOT_END - timedelta(days=6)  # 2025-12-10 (7 jours)

# Villes + filtre pays ISO alpha-2 (évite les résultats ambigus) [web:203]
CITY_GEOCODE = {
    "Lyon": {"name": "Lyon", "countryCode": "FR"},
    "Paris": {"name": "Paris", "countryCode": "FR"},
    "London": {"name": "London", "countryCode": "GB"},
    "New York": {"name": "New York", "countryCode": "US"},
    "Sao Paulo": {"name": "Sao Paulo", "countryCode": "BR"},
}

# -----------------------
# HELPERS
# -----------------------
def geocode_city(city: str):
    cfg = CITY_GEOCODE[city]
    params = {
        "name": cfg["name"],
        "count": 1,
        "language": "en",
        "format": "json",
        "countryCode": cfg["countryCode"],  # filtre pays [web:203]
    }
    r = requests.get(GEOCODE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    if not results:
        raise ValueError(f"No geocoding result for city={city} params={params}")
    res = results[0]
    return float(res["latitude"]), float(res["longitude"])

def fetch_snapshot(lat: float, lon: float, start_d: date, end_d: date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_d.isoformat(),
        "end_date": end_d.isoformat(),
        "daily": "precipitation_sum,temperature_2m_max",  # variables daily [web:119]
        "timezone": "auto",  # simplifie l'alignement sur l'heure locale [web:119]
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    daily = data["daily"]
    times = daily["time"]
    prcp = daily["precipitation_sum"]
    tmax = daily["temperature_2m_max"]

    # Format 7 lignes
    lines = []
    for d, p, t in zip(times, prcp, tmax):
        # p peut être None parfois -> gérer proprement
        p_val = float(p) if p is not None else 0.0
        t_val = float(t) if t is not None else float("nan")
        lines.append(f"- {d}: precip_sum={p_val:.2f} mm, tmax={t_val:.1f} °C")
    return "\n".join(lines)

def build_prompt(city: str, date_local: str, snapshot_lines: str):
    return f"""Donne une prévision probabiliste pour la journée suivante.
Réponds UNIQUEMENT en JSON strict avec les champs:
tmax_p10, tmax_p50, tmax_p90 (°C),
p_rain_gt0 (0-1),
rain_p50, rain_p90 (mm),
justification (2 phrases max).

Ville: {city}
Date (locale): {date_local}

Snapshot observations (7 derniers jours complets disponibles à J0={RUN_DATE.isoformat()}):
{snapshot_lines}

À prédire pour le jour {date_local}:
- Tmax (°C): quantiles P10/P50/P90
- Pluie (mm): p(pluie>0), P50 et P90 de precipitation_sum

Deadline: {date_local} 23:59 (heure locale)
"""

# -----------------------
# MAIN
# -----------------------
def main():
    df = pd.read_csv(INPUT_CSV)

    # 1) On compacte à 35 lignes : 1 prompt par (city, date_local)
    base = (
        df.sort_values(["city", "date_local"])
          .drop_duplicates(subset=["city", "date_local"], keep="first")
          .copy()
    )
    base["target"] = "BOTH"

    # 2) Géocodage + snapshot + prompt
    lats, lons, snapshots, prompts = [], [], [], []

    for _, row in base.iterrows():
        city = row["city"]
        dloc = row["date_local"]

        lat, lon = geocode_city(city)
        snap = fetch_snapshot(lat, lon, SNAPSHOT_START, SNAPSHOT_END)
        prompt = build_prompt(city, dloc, snap)

        lats.append(lat)
        lons.append(lon)
        snapshots.append(snap)
        prompts.append(prompt)

        time.sleep(0.25)  # rate limit soft

    base["latitude"] = lats
    base["longitude"] = lons
    base["snapshot_start_fixed"] = SNAPSHOT_START.isoformat()
    base["snapshot_end_fixed"] = SNAPSHOT_END.isoformat()
    base["snapshot_text"] = snapshots
    base["prompt_text"] = prompts

    # 3) Sauvegarde base (35)
    base.to_csv(OUT_BASE, index=False)

    # 4) Expansion ON/OFF (70) - même prompt, seul web_mode change
    expanded = pd.concat(
        [base.assign(web_mode="OFF"), base.assign(web_mode="ON")],
        ignore_index=True,
    ).sort_values(["city", "date_local", "web_mode"])

    expanded.to_csv(OUT_EXPANDED, index=False)

    print(f"OK: {len(base)} rows -> {OUT_BASE}")
    print(f"OK: {len(expanded)} rows -> {OUT_EXPANDED}")
    print(f"Snapshot fixed: {SNAPSHOT_START.isoformat()} to {SNAPSHOT_END.isoformat()}")

if __name__ == "__main__":
    main()
