from pathlib import Path
import pandas as pd
import fitdecode


DATA_DIR = Path("CorosData")

# Input issu de l'étape fitparse
FITPARSE_REPORT = Path("fit_parse_report.csv")
FITPARSE_OK_SESSIONS = Path("activities_session_allfields.csv")

# Outputs fallback + merge
FITDECODE_REPORT = Path("fitdecode_report.csv")
FITDECODE_SESSIONS = Path("activities_session_fitdecode.csv")
MERGED_SESSIONS = Path("activities_session_241.csv")


def session_dict_from_fitdecode_frame(frame):
    """
    frame: FitDataMessage (fitdecode)
    On reconstruit un dict {field.name: field.value} à partir de frame.fields. [page:0]
    """
    d = {}
    for f in frame.fields:  # liste de champs du message [page:0]
        name = getattr(f, "name", None)
        if name:
            d[name] = f.value
    return d


def main():
    rep = pd.read_csv(FITPARSE_REPORT)
    error_files = rep.loc[rep["status"] == "error", "file_name"].tolist()

    rows = []
    report2 = []

    for fn in error_files:
        fp = DATA_DIR / fn
        status, err = "ok", ""
        session_values = None

        try:
            with fitdecode.FitReader(
                fp,
                check_crc=fitdecode.CrcCheck.DISABLED,            # le plus permissif/rapide [page:1]
                error_handling=fitdecode.ErrorHandling.IGNORE,   # "Parser proceeds when possible" [page:1]
            ) as reader:
                for frame in reader:
                    if frame.frame_type == fitdecode.FIT_FRAME_DATA and frame.name == "session":
                        session_values = session_dict_from_fitdecode_frame(frame)
                        break

            if session_values is None:
                status = "no_session"

        except Exception as e:
            status, err = "error", f"{type(e).__name__}: {e}"

        report2.append({"file_name": fn, "status": status, "error": err})

        if session_values is not None:
            session_values["file_name"] = fn
            rows.append(session_values)

    df2 = pd.DataFrame(rows)
    rep2 = pd.DataFrame(report2)

    rep2.to_csv(FITDECODE_REPORT, index=False)
    print("error_files:", len(error_files))
    print(rep2["status"].value_counts(dropna=False))
    print("decoded_sessions:", len(df2))

    if len(df2) > 0:
        df2.to_csv(FITDECODE_SESSIONS, index=False)
        print("OK ->", str(FITDECODE_SESSIONS))
    else:
        print("No decoded sessions -> skipping activities_session_fitdecode.csv")

    # Merge (2C) uniquement si df2 non vide (sinon EmptyDataError)
    a = pd.read_csv(FITPARSE_OK_SESSIONS)

    if len(df2) > 0:
        all_sessions = pd.concat([a, df2], ignore_index=True)
    else:
        all_sessions = a.copy()

    all_sessions = all_sessions.drop_duplicates(subset=["file_name"], keep="first")  # [web:80]
    all_sessions.to_csv(MERGED_SESSIONS, index=False)

    print("OK ->", str(MERGED_SESSIONS))
    print("unique file_name:", all_sessions["file_name"].nunique())
    print("rows:", len(all_sessions))


if __name__ == "__main__":
    main()
