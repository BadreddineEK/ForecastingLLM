import pandas as pd
import requests
from datetime import datetime, timezone, date, timedelta

EXCEL_PATH = "markets_experiments_FINAL.xlsx"
SHEET_NAME = "markets_experiments_simple"

# Providers
STOOQ_URL = "https://stooq.com/q/d/l/"
COINGECKO_RANGE_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"

COINGECKO_IDS = {"BTC": "bitcoin", "ETH": "ethereum"}

EPS = 1e-12  # outcome strict >

def utc_now():
    return datetime.now(timezone.utc)

def utc_today_date():
    return utc_now().date()

def now_utc_excel_ts():
    return pd.Timestamp(utc_now().replace(tzinfo=None, second=0, microsecond=0))

def to_date_safe(x):
    if pd.isna(x):
        return None
    return pd.to_datetime(x, errors="coerce").date()

def stooq_close_on_us(ticker: str, target: date):
    """
    Stooq daily close on target date for US symbols (ticker.us).
    CSV includes columns Date,Open,High,Low,Close,Volume. [web:797]
    """
    symbol = f"{ticker.lower()}.us"
    r = requests.get(STOOQ_URL, params={"s": symbol, "i": "d"}, timeout=30)
    r.raise_for_status()

    from io import StringIO
    px = pd.read_csv(StringIO(r.text))
    if "Date" not in px.columns or "Close" not in px.columns:
        return None

    px["Date"] = pd.to_datetime(px["Date"], errors="coerce").dt.date
    px = px.dropna(subset=["Date"]).sort_values("Date")
    row = px[px["Date"] == target]
    if row.empty:
        return None
    return float(row.iloc[0]["Close"])

def coingecko_close_utc_eod(ticker: str, target: date):
    """
    Crypto close convention (scientifique): price at 23:59:59 UTC of target date
    (or last price point <= that timestamp). [web:813]
    """
    coin_id = COINGECKO_IDS.get(ticker.upper())
    if not coin_id:
        return None

    end_dt = datetime(target.year, target.month, target.day, 23, 59, 59, tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(hours=6)  # buffer to ensure we have points near the end

    url = COINGECKO_RANGE_URL.format(coin_id=coin_id)
    params = {
        "vs_currency": "usd",
        "from": int(start_dt.timestamp()),
        "to": int(end_dt.timestamp()),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()

    prices = j.get("prices", [])
    if not prices:
        return None

    # pick last point (ms, price) <= end_dt
    end_ms = int(end_dt.timestamp() * 1000)
    last = None
    for ms, price in prices:
        if ms <= end_ms:
            last = float(price)
        else:
            break
    return last

def compute_outcome(close_j0: float, close_j5: float) -> int:
    return int((float(close_j5) - float(close_j0)) > EPS)

def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)  # [file:787]
    df["j0_date"] = pd.to_datetime(df["j0_date"], errors="coerce")          # [file:787]
    df["j5_target_date"] = pd.to_datetime(df["j5_target_date"], errors="coerce")  # [file:787]

    run_ts = now_utc_excel_ts()
    today_utc = utc_today_date()

    updated = 0
    skipped_not_due = 0
    skipped_missing = 0

    for i, row in df.iterrows():
        ticker = str(row["ticker"]).upper()
        asset_class = str(row["asset_class"]).lower()
        j0 = to_date_safe(row["j0_date"])
        j5 = to_date_safe(row["j5_target_date"])

        if j0 is None or j5 is None:
            continue

        # on n'update que si J5 est strictement passé (sinon close pas "finale")
        if not (today_utc > j5):
            skipped_not_due += 1
            continue

        close_j0 = row.get("close_j0")
        if pd.isna(close_j0):
            skipped_missing += 1
            continue
        close_j0 = float(close_j0)

        # fetch close_j5
        if asset_class == "crypto":
            close_j5 = coingecko_close_utc_eod(ticker, j5)
        else:
            close_j5 = stooq_close_on_us(ticker, j5)

        if close_j5 is None:
            skipped_missing += 1
            continue

        # write (overwrite allowed => si provider révise, tu restes correct)
        df.loc[i, "close_j5"] = float(close_j5)
        df.loc[i, "outcome"] = compute_outcome(close_j0, close_j5)
        df.loc[i, "outcome_run_datetime_utc"] = run_ts
        updated += 1

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        df.to_excel(w, sheet_name=SHEET_NAME, index=False)

    print(f"OK -> updated {EXCEL_PATH}")
    print(f"Rows updated: {updated} | skipped_not_due: {skipped_not_due} | skipped_missing: {skipped_missing}")
    print(df[["exp_id","ticker","j0_date","j5_target_date","close_j0","close_j5","outcome"]])

if __name__ == "__main__":
    main()
