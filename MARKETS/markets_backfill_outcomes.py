from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# === Chemins EXACTS (MARKETS, pas WEATHER) ===
IN_XLSX = Path(r"C:\Users\elkhamli\OneDrive - Boehringer Ingelheim\Bureau\ForecastingLLM\MARKETS\markets_experiments_FINAL.xlsx")
OUT_XLSX = IN_XLSX.with_name("markets_experiments_FINAL_filled.xlsx")

def get_close_yf(ticker: str, date_str: str):
    """
    Récupère la clôture daily la plus proche de date_str (YYYY-MM-DD) pour ticker,
    en prenant la première séance >= date_str (utile si J0 ou J+5 tombent sur un jour férié/week-end).
    """
    d = pd.to_datetime(date_str).date().isoformat()
    start = pd.to_datetime(date_str) - pd.Timedelta(days=1)
    end = pd.to_datetime(date_str) + pd.Timedelta(days=7)

    data = yf.download(
        ticker,
        start=start.date().isoformat(),
        end=(end.date() + pd.Timedelta(days=1)).isoformat(),
        progress=False,
    )
    if data.empty:
        return None, None

    data = data.sort_index()
    future = data[data.index >= pd.to_datetime(d)]
    if future.empty:
        return None, None

    date_used = future.index[0].date().isoformat()
    close = float(future.iloc[0]["Close"])
    return close, date_used

def main():
    if not IN_XLSX.exists():
        raise FileNotFoundError(f"Input file not found: {IN_XLSX}")

    df = pd.read_excel(IN_XLSX)

    # On ne remplit que les lignes sans outcome
    mask = df["exp_id"].notna() & df["outcome"].isna()
    to_fill = df[mask].copy()

    print("Input:", IN_XLSX)
    print("Rows needing outcome fill:", len(to_fill))

    filled = 0
    errors = 0

    for idx, row in to_fill.iterrows():
        ticker = str(row["ticker"])
        j0_date = pd.to_datetime(row["j0_date"]).date().isoformat()
        j5_date = pd.to_datetime(row["j5_target_date"]).date().isoformat()

        try:
            # close_j0 : si déjà présent, on garde; sinon on le récupère
            close_j0 = row.get("close_j0")
            if pd.isna(close_j0):
                c0, used0 = get_close_yf(ticker, j0_date)
                if c0 is None:
                    raise ValueError(f"no close_j0 for {ticker} on/after {j0_date}")
                close_j0 = c0
                df.at[idx, "close_j0"] = close_j0

            # close_j5
            c5, used5 = get_close_yf(ticker, j5_date)
            if c5 is None:
                raise ValueError(f"no close_j5 for {ticker} on/after {j5_date}")
            close_j5 = c5
            df.at[idx, "close_j5"] = close_j5

            # outcome (1 si close_j5 > close_j0, sinon 0)
            outcome = 1 if close_j5 > close_j0 else 0
            df.at[idx, "outcome"] = outcome

            # timestamp obs
            df.at[idx, "outcome_run_datetime_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

            filled += 1

        except Exception as e:
            errors += 1
            print(f"[ERROR] idx={idx} exp_id={row.get('exp_id')} ticker={ticker}: {type(e).__name__}: {e}")

    df.to_excel(OUT_XLSX, index=False)

    print("Filled rows:", filled)
    print("Errors:", errors)
    print("OK ->", OUT_XLSX)

if __name__ == "__main__":
    main()
