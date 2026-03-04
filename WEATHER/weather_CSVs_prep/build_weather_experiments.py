import pandas as pd
import requests
from datetime import datetime, timezone

IN_BASE = "weather_prompts_base.csv"
OUT = "weather_experiments_simple.csv"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"  # [web:146]

THREAD_OFF = "Forecasting Study — Weather — Web OFF"
THREAD_ON  = "Forecasting Study — Weather — Web ON"
MODEL_NAME = "GPT-5.2 (reasoning=ON)"

def fetch_open_meteo_forecast_daily(lat: float, lon: float, start_date: str, end_date: str):
    # daily: temperature_2m_max, precipitation_sum, precipitation_probability_max [web:146]
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,precipitation_sum,precipitation_probability_max",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",  # aligne les jours sur le fuseau local [web:146]
    }
    r = requests.get(FORECAST_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    daily = data["daily"]

    out = {}
    for d, tmax, psum, pprob in zip(
        daily["time"],
        daily["temperature_2m_max"],
        daily["precipitation_sum"],
        daily["precipitation_probability_max"],
    ):
        out[d] = {
            "baseline_tmax": float(tmax) if tmax is not None else None,
            "baseline_precip_sum": float(psum) if psum is not None else None,
            "baseline_precip_prob_max": float(pprob) if pprob is not None else None,  # en %
        }
    return out

def main():
    base = pd.read_csv(IN_BASE).copy()

    # 35 lignes attendues
    base = base.sort_values(["city", "date_local"]).reset_index(drop=True)

    start_date = base["date_local"].min()
    end_date = base["date_local"].max()
    baseline_run_utc = datetime.now(timezone.utc).isoformat()

    # Structure "simple" : 1 ligne = 1 expérience
    df = pd.DataFrame({
        "exp_id": base.apply(lambda r: f"WEATHER_{r['city']}_{r['date_local']}", axis=1),
        "city": base["city"],
        "latitude": base.get("latitude", None),
        "longitude": base.get("longitude", None),
        "date_local": base["date_local"],

        # Snapshot
        "snapshot_start_fixed": base.get("snapshot_start_fixed", ""),
        "snapshot_end_fixed": base.get("snapshot_end_fixed", ""),
        "snapshot_text": base.get("snapshot_text", ""),

        # Prompt identique pour OFF/ON
        "prompt_text": base.get("prompt_text", ""),

        # Traçabilité LLM
        "model": MODEL_NAME,
        "thread_off_name": THREAD_OFF,
        "thread_on_name": THREAD_ON,
        "datetime_pred_off_utc": "",
        "datetime_pred_on_utc": "",

        # Tu remplis ça à la main (copier-coller brut depuis Perplexity)
        "off_json_raw": "",
        "on_json_raw": "",

        # Baseline tool (auto)
        "baseline_provider": "Open-Meteo Forecast API",
        "baseline_run_datetime_utc": baseline_run_utc,
        "baseline_tmax": None,
        "baseline_precip_sum": None,
        "baseline_precip_prob_max": None,

        # Réalité (à remplir jour par jour)
        "obs_run_datetime_utc": "",
        "obs_tmax": "",
        "obs_precip_sum": "",
        "obs_rain_gt0": "",
    })

    # Remplir baseline (1 appel par ville)
    for city, grp in df.groupby("city"):
        lat = float(grp.iloc[0]["latitude"])
        lon = float(grp.iloc[0]["longitude"])

        fc_map = fetch_open_meteo_forecast_daily(lat, lon, start_date, end_date)

        for idx in grp.index:
            dloc = df.at[idx, "date_local"]
            if dloc in fc_map:
                df.at[idx, "baseline_tmax"] = fc_map[dloc]["baseline_tmax"]
                df.at[idx, "baseline_precip_sum"] = fc_map[dloc]["baseline_precip_sum"]
                df.at[idx, "baseline_precip_prob_max"] = fc_map[dloc]["baseline_precip_prob_max"]

    # Export CSV (pandas gère le quoting nécessaire pour les champs multi-lignes) [web:281]
    df.to_csv(OUT, index=False)
    print(f"OK -> {OUT} ({len(df)} rows)")
    print(f"Baseline window: {start_date} to {end_date}")
    print(f"Baseline run UTC: {baseline_run_utc}")

if __name__ == "__main__":
    main()
