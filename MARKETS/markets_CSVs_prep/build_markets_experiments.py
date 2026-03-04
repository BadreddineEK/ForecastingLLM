import pandas as pd
import requests
from datetime import date, timedelta
from io import StringIO

OUT = "markets_experiments_simple.csv"

J0 = date(2025, 12, 17)
J5_TARGET = date(2025, 12, 24)

TICKERS = [
    ("SPY", "equity_etf"),
    ("QQQ", "equity_etf"),
    ("GLD", "equity_etf"),
    ("TLT", "equity_etf"),
    ("AAPL", "equity_stock"),
    ("MSFT", "equity_stock"),
    ("NVDA", "equity_stock"),
    ("TSLA", "equity_stock"),
    ("BTC", "crypto"),
    ("ETH", "crypto"),
]

def fmt_pct(x):
    return f"{x:.2f}%"

def compute_range_pct(closes):
    mn, mx = min(closes), max(closes)
    return (mx / mn - 1.0) * 100.0 if mn > 0 else None

def build_prompt(ticker, snapshot_text):
    return f"""Donne une probabilité p entre 0 et 1 que l’événement suivant arrive.
Réponds UNIQUEMENT en JSON : {{"p": <number>, "justification": "<2 phrases max>"}}.

Actif: {ticker}
Événement: Close(J+5 séances) > Close(J0)

Snapshot à J0 :
{snapshot_text}

Deadline: 24 décembre 2025 23:59 UTC
"""

# ---------- DATA SOURCES ----------
def fetch_stooq_daily(ticker: str) -> pd.DataFrame:
    """
    Télécharge les données daily depuis Stooq via le endpoint CSV:
    https://stooq.com/q/d/l/?s={symbol}&i=d
    """
    stooq_symbol = f"{ticker}.US".lower()
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    # Normaliser noms de colonnes (Stooq renvoie souvent: Date, Open, High, Low, Close, Volume)
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"Unexpected Stooq columns for {ticker}: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")
    return df[["date", "close"]].rename(columns={"date": "Date", "close": "Close"})

def fetch_coingecko_daily(coin_id: str, start_d: date, end_d: date) -> pd.DataFrame:
    """
    CoinGecko market_chart_range -> on agrège en daily close (dernière valeur du jour UTC).
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": int(pd.Timestamp(start_d).timestamp()),
        "to": int(pd.Timestamp(end_d + timedelta(days=1)).timestamp()),
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    prices = pd.DataFrame(data["prices"], columns=["ts_ms", "price"])
    prices["Date"] = pd.to_datetime(prices["ts_ms"], unit="ms", utc=True).dt.date
    daily_close = (
        prices.sort_values(["Date", "ts_ms"])
              .groupby("Date")["price"]
              .last()
              .reset_index()
              .rename(columns={"price": "Close"})
              .sort_values("Date")
    )
    return daily_close

def get_last_6_closes(df: pd.DataFrame, j0: date) -> pd.DataFrame:
    dff = df[df["Date"] <= j0].sort_values("Date")
    tail = dff.tail(6)
    if len(tail) < 6:
        raise ValueError(f"Not enough history up to {j0} (got {len(tail)})")
    return tail

# ---------- MAIN ----------
def main():
    rows = []
    start_window = J0 - timedelta(days=45)  # marge pour crypto et jours non-tradés

    for ticker, asset_class in TICKERS:
        if asset_class in ["equity_etf", "equity_stock"]:
            hist = fetch_stooq_daily(ticker)
        else:
            coin_id = "bitcoin" if ticker == "BTC" else "ethereum"
            hist = fetch_coingecko_daily(coin_id, start_window, J0)

        tail = get_last_6_closes(hist, J0)
        dates = tail["Date"].tolist()      # 6 dates (les 6 dernières <= J0)
        closes = [float(x) for x in tail["Close"].tolist()]  # 6 closes

        close_j0 = closes[-1]
        close_jm5 = closes[0]
        perf_5d = (close_j0 / close_jm5 - 1.0) * 100.0
        range_pct = compute_range_pct(closes)

        # Closes J-1..J-5 dans l’ordre décroissant (J-1, J-2, ..., J-5)
        closes_jm1_to_jm5 = closes[-2::-1]   # ex: [J-1, J-2, ..., J-5]
        dates_jm1_to_jm5 = dates[-2::-1]

        snapshot_lines = [
            f"- Close(J0={dates[-1]}): {close_j0:.4f}",
            f"- Close(J-1..J-5) (dates {dates_jm1_to_jm5[0]}→{dates_jm1_to_jm5[-1]}): "
            + ", ".join([f"{x:.4f}" for x in closes_jm1_to_jm5]),
            f"- Perf 5 jours (%): {fmt_pct(perf_5d)}",
            f"- Range% 5 jours: {fmt_pct(range_pct)}",
        ]
        snapshot_text = "\n".join(snapshot_lines)
        prompt_text = build_prompt(ticker, snapshot_text)

        rows.append({
            "exp_id": f"MKT_{ticker}_{J0.isoformat()}",
            "ticker": ticker,
            "asset_class": asset_class,
            "j0_date": J0.isoformat(),
            "j5_target_date": J5_TARGET.isoformat(),
            "snapshot_text": snapshot_text,
            "prompt_text": prompt_text,
            "datetime_pred_off_utc": "",
            "off_json_raw": "",
            "datetime_pred_on_utc": "",
            "on_json_raw": "",
            "baseline_p": 0.5,
            "baseline_rule": "Coin flip baseline: p=0.5 for Close(J+5) > Close(J0).",
            "outcome": "",
            "close_j0": close_j0,
            "close_j5": "",
            "outcome_run_datetime_utc": "",
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"OK -> {OUT} ({len(df)} rows)")

if __name__ == "__main__":
    main()
