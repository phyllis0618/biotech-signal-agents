from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from src.connectors.http import get_json


def fetch_yahoo_snapshots(tickers: List[str]) -> List[Dict]:
    snapshots: List[Dict] = []
    for ticker in tickers:
        snapshots.append(fetch_yahoo_snapshot(ticker))
    return snapshots


def fetch_yahoo_snapshot(ticker: str) -> Dict:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": "1m", "range": "1d"}
    try:
        data = get_json(url, params=params, timeout=10)
        result = data.get("chart", {}).get("result", [{}])[0]
        meta = result.get("meta", {})
        price = float(meta.get("regularMarketPrice", 0.0))
        prev_close = float(meta.get("previousClose", 0.0))
        change = price - prev_close
        pct = (change / prev_close * 100) if prev_close else 0.0
        return {
            "ticker": ticker,
            "price": round(price, 4),
            "change": round(change, 4),
            "change_pct": round(pct, 2),
            "market_time": str(meta.get("regularMarketTime", "")),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "yahoo_chart",
        }
    except Exception:
        return {
            "ticker": ticker,
            "price": 0.0,
            "change": 0.0,
            "change_pct": 0.0,
            "market_time": "",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "fallback",
        }


def fetch_yahoo_intraday_series(ticker: str, interval: str = "5m", range_: str = "1d") -> List[Dict]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": interval, "range": range_}
    try:
        data = get_json(url, params=params, timeout=10)
        result = data.get("chart", {}).get("result", [{}])[0]
        timestamps = result.get("timestamp", [])
        quote = (
            result.get("indicators", {})
            .get("quote", [{}])[0]
        )
        closes = quote.get("close", [])
        rows: List[Dict] = []
        for ts, close in zip(timestamps, closes):
            if close is None:
                continue
            rows.append(
                {
                    "time": datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat(),
                    "close": float(close),
                }
            )
        return rows
    except Exception:
        return []


def fetch_yahoo_daily_bars(ticker: str, range_: str = "1mo", max_rows: int = 10) -> List[Dict]:
    """
    Daily OHLC closes from Yahoo chart API (last up to `max_rows` trading days).
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": "1d", "range": range_}
    try:
        data = get_json(url, params=params, timeout=15)
        result = data.get("chart", {}).get("result", [{}])[0]
        timestamps = result.get("timestamp", [])
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        closes = quote.get("close", [])
        rows: List[Dict] = []
        for ts, close in zip(timestamps, closes):
            if close is None:
                continue
            rows.append(
                {
                    "date": datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat(),
                    "close": float(close),
                }
            )
        if max_rows and len(rows) > max_rows:
            rows = rows[-max_rows:]
        return rows
    except Exception:
        return []
