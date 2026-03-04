import pandas as pd
import requests
from datetime import datetime, timezone, timedelta

EXCEL_PATH = "weather_experiment_FINAL.xlsx"
SHEET_NAME = "weather_experiments_simple"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_ARCHIVE_VARS = "temperature_2m_max,precipitation_sum"
EPS_TMAX = 0.05       # tolérance °C pour décider "ça a changé"
EPS_PSUM = 0.05       # tolérance mm

def utc_now():
    return datetime.now(timezone.utc)

def now_utc_excel_ts():
    # timestamp UTC naïf (pratique pour Excel)
    return pd.Timestamp(utc_now().replace(tzinfo=None, second=0, microsecond=0))

def get_utc_offset_seconds(lat: float, lon: float) -> int:
    # Open-Meteo renvoie utc_offset_seconds quand timezone=auto [web:755]
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        # on demande un truc minimal; utc_offset_seconds est dans la réponse
        "daily": "temperature_2m_max",
        "forecast_days": 1,
    }
    r = requests.get(FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    return int(j.get("utc_offset_seconds", 0))

def local_today_from_offset(lat: float, lon: float):
    offset_s = get_utc_offset_seconds(lat, lon)
    local_now = utc_now() + timedelta(seconds=offset_s)
    return local_now.date()

def fetch_archive_range(lat: float, lon: float, start_date, end_date) -> pd.DataFrame:
    # Historical API: daily temperature_2m_max + precipitation_sum [web:754]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": DAILY_ARCHIVE_VARS,
        "timezone": "auto",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    d = j.get("daily", {})
    out = pd.DataFrame({
        "date": pd.to_datetime(d.get("time", []), errors="coerce").dt.date,
        "tmax": d.get("temperature_2m_max", []),
        "precip_sum": d.get("precipitation_sum", []),
    }).dropna(subset=["date"])
    return out

def is_close(a, b, eps):
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) <= eps

def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)  # [file:753]
    df["date_local"] = pd.to_datetime(df["date_local"], errors="coerce")  # [file:753]
    df["obs_run_datetime_utc"] = pd.to_datetime(df["obs_run_datetime_utc"], errors="coerce")  # [file:753]

    # villes uniques
    cities = (
        df[["city", "latitude", "longitude"]]
        .dropna()
        .drop_duplicates()
        .to_dict(orient="records")
    )

    run_ts = now_utc_excel_ts()
    touched_rows = 0

    for c in cities:
        city = str(c["city"])
        lat = float(c["latitude"])
        lon = float(c["longitude"])

        local_today = local_today_from_offset(lat, lon)  # [web:755]

        # lignes de cette ville, journées clôturées uniquement: date_local < local_today
        mask_city = (df["city"].astype(str) == city)
        mask_has_date = df["date_local"].notna()
        mask_closed = df["date_local"].dt.date < local_today

        idxs = df[mask_city & mask_has_date & mask_closed].index.tolist()
        if not idxs:
            continue

        dates = sorted({df.loc[i, "date_local"].date() for i in idxs})
        start_d, end_d = min(dates), max(dates)

        arch = fetch_archive_range(lat, lon, start_d, end_d)
        if arch.empty:
            continue

        arch_map = {row["date"]: row for _, row in arch.iterrows()}

        for i in idxs:
            dloc = df.loc[i, "date_local"].date()
            if dloc not in arch_map:
                continue

            tmax_new = arch_map[dloc]["tmax"]
            psum_new = arch_map[dloc]["precip_sum"]
            if pd.isna(tmax_new) or pd.isna(psum_new):
                continue

            tmax_old = df.loc[i, "obs_tmax"]
            psum_old = df.loc[i, "obs_precip_sum"]

            changed = (
                pd.isna(tmax_old) or pd.isna(psum_old) or
                (not is_close(tmax_old, tmax_new, EPS_TMAX)) or
                (not is_close(psum_old, psum_new, EPS_PSUM))
            )

            if changed:
                df.loc[i, "obs_tmax"] = float(tmax_new)
                df.loc[i, "obs_precip_sum"] = float(psum_new)
                df.loc[i, "obs_rain_gt0"] = int(float(psum_new) > 0.0)
                df.loc[i, "obs_run_datetime_utc"] = run_ts
                touched_rows += 1

    # write back
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        df.to_excel(w, sheet_name=SHEET_NAME, index=False)

    print(f"OK -> obs updated in {EXCEL_PATH}. Rows updated: {touched_rows}")

if __name__ == "__main__":
    main()
