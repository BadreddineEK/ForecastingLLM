from pathlib import Path
import pandas as pd
import requests
from datetime import datetime, timezone

# --- Chemins (fixés explicitement) ---
IN_XLSX = Path(r"C:\Users\elkhamli\OneDrive - Boehringer Ingelheim\Bureau\ForecastingLLM\WEATHER\weather_experiment_FINAL.xlsx")
OUT_XLSX = Path(r"C:\Users\elkhamli\OneDrive - Boehringer Ingelheim\Bureau\ForecastingLLM\WEATHER\weather_experiment_FINAL_filled_obs.xlsx")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"  # [web:419]

def fetch_daily_obs(lat, lon, date_str):
    """
    date_str: 'YYYY-MM-DD'
    Returns: {'tmax': float|None, 'precip_sum': float|None} or None
    """
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": date_str,
        "end_date": date_str,
        "daily": "temperature_2m_max,precipitation_sum",  # daily vars [web:420]
        "timezone": "auto",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    daily = js.get("daily", {})
    times = daily.get("time", [])
    tmax = daily.get("temperature_2m_max", [])
    prcp = daily.get("precipitation_sum", [])

    if not times:
        return None

    return {
        "tmax": float(tmax[0]) if tmax and tmax[0] is not None else None,
        "precip_sum": float(prcp[0]) if prcp and prcp[0] is not None else None,
    }

def main():
    if not IN_XLSX.exists():
        raise FileNotFoundError(f"Input file not found: {IN_XLSX}")

    df = pd.read_excel(IN_XLSX)

    # Lignes à remplir: exp_id + date_local présents, obs_tmax manquant
    mask = df["exp_id"].notna() & df["date_local"].notna() & (df["obs_tmax"].isna())
    to_fill = df[mask].copy()

    print("Input:", IN_XLSX)
    print("Rows needing obs fill:", len(to_fill))

    filled = 0
    errors = 0

    for idx, row in to_fill.iterrows():
        d = pd.to_datetime(row["date_local"]).date().isoformat()
        lat = row["latitude"]
        lon = row["longitude"]

        try:
            obs = fetch_daily_obs(lat, lon, d)
            if obs is None:
                errors += 1
                continue

            df.at[idx, "obs_run_datetime_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            df.at[idx, "obs_tmax"] = obs["tmax"]
            df.at[idx, "obs_precip_sum"] = obs["precip_sum"]
            df.at[idx, "obs_rain_gt0"] = 1 if (obs["precip_sum"] is not None and obs["precip_sum"] > 0) else 0

            filled += 1

        except Exception as e:
            errors += 1
            print(f"[ERROR] idx={idx} exp_id={row.get('exp_id')} date={d} -> {type(e).__name__}: {e}")

    df.to_excel(OUT_XLSX, index=False)

    print("Filled rows:", filled)
    print("Errors:", errors)
    print("OK ->", OUT_XLSX)

if __name__ == "__main__":
    main()
