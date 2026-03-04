from pathlib import Path
import pandas as pd
from fitparse import FitFile, FitParseError

DATA_DIR = Path("CorosData")

rows = []
report = []

fit_files = sorted(DATA_DIR.glob("*.fit"))

for fp in fit_files:
    status, err = "ok", ""
    try:
        ff = FitFile(str(fp))
        # optionnel: force un parsing complet pour détecter les erreurs tôt
        ff.parse()  # peut lever FitParseError [page:1]

        sessions = list(ff.get_messages("session"))
        if not sessions:
            status = "empty_session"
            rows.append({"file_name": fp.name})
        else:
            s = sessions[0]
            d = s.get_values()  # dict de tous les champs session [page:1]
            d["file_name"] = fp.name
            rows.append(d)

    except FitParseError as e:
        status, err = "error", str(e)
    except Exception as e:
        status, err = "error", str(e)

    report.append({"file_name": fp.name, "status": status, "error": err})

df = pd.DataFrame(rows)
rep = pd.DataFrame(report)

df.to_csv("activities_session_allfields.csv", index=False)
rep.to_csv("fit_parse_report.csv", index=False)

print("n_fit_files:", len(fit_files))
print(rep["status"].value_counts(dropna=False))
print("cols activities_session_allfields:", df.shape[1])
print(df.head(3))
