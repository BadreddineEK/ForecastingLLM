from pathlib import Path
import pandas as pd
import numpy as np
import fitdecode

DATA_DIR = Path("CorosData")

IN_CSV = "activities_session_241_fixed.csv"
OUT_CSV = "activities_session_241_fixed.csv"
REPORT_CSV = "fix_missing_time_report.csv"

def get_field(frame, field_name):
    # frame.fields contient des FitField avec .name et .value [page:0]
    for f in frame.fields:
        if getattr(f, "name", None) == field_name:
            return f.value
    return None

df = pd.read_csv(IN_CSV)

# parse datetimes (peut être NaT)
df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

to_fix = df[df["start_time"].isna() | df["timestamp"].isna()].copy()
print("Rows needing fix:", len(to_fix))

rows_report = []

for i, r in to_fix.iterrows():
    fn = r["file_name"]
    fp = DATA_DIR / fn

    t_min = None
    t_max = None
    status = "ok"
    err = ""

    try:
        with fitdecode.FitReader(
            fp,
            check_crc=fitdecode.CrcCheck.DISABLED,
            error_handling=fitdecode.ErrorHandling.IGNORE,
        ) as reader:  # FitReader itère sur les frames/messages [page:1]
            for frame in reader:
                if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                    continue

                # 1) Priorité: record.timestamp (souvent le plus fiable)
                if frame.name == "record":
                    ts = get_field(frame, "timestamp")
                    if ts is not None:
                        if (t_min is None) or (ts < t_min):
                            t_min = ts
                        if (t_max is None) or (ts > t_max):
                            t_max = ts

                # (optionnel) 2) fallback: activity.timestamp si jamais pas de record
                # if frame.name == "activity":
                #     ts2 = get_field(frame, "timestamp")
                #     ...

    except Exception as e:
        status = "error"
        err = f"{type(e).__name__}: {e}"

    rows_report.append({
        "file_name": fn,
        "status": status,
        "err": err,
        "start_time_old": r["start_time"],
        "timestamp_old": r["timestamp"],
        "start_time_fix": t_min,
        "timestamp_fix": t_max,
    })

    # appliquer fix si on a trouvé des timestamps
    if (status == "ok") and (t_min is not None) and (t_max is not None):
        t_min = pd.Timestamp(t_min)
        t_max = pd.Timestamp(t_max)

        if t_min.tzinfo is not None:
            t_min = t_min.tz_convert(None)  # [web:173]
        if t_max.tzinfo is not None:
            t_max = t_max.tz_convert(None)  # [web:173]

        df.at[i, "start_time"] = t_min
        df.at[i, "timestamp"] = t_max

rep = pd.DataFrame(rows_report)
rep.to_csv(REPORT_CSV, index=False)
df.to_csv(OUT_CSV, index=False)

print("OK ->", OUT_CSV)
print("OK ->", REPORT_CSV)
print("Fixed start_time count:", int(pd.read_csv(OUT_CSV)["start_time"].notna().sum()))
