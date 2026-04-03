from __future__ import annotations

from typing import Any, Dict, List

from src.agentic_research.data_extractor import DataExtractor
from src.connectors.market_data import fetch_yahoo_daily_bars
from src.connectors.sec_edgar import fetch_latest_8k_plain_text


def tool_sec_material_events(ticker: str, max_chars: int = 12000) -> Dict[str, Any]:
    """SEC: latest 8-K material text slice for LLM / risk context."""
    text = fetch_latest_8k_plain_text(ticker, max_chars=max_chars)
    return {
        "tool": "sec_edgar_8k",
        "ticker": ticker,
        "chars": len(text),
        "excerpt": text[:4000],
    }


def tool_clinical_trials_fda_bundle(ticker: str, company: str) -> Dict[str, Any]:
    """ClinicalTrials.gov + FDA calendar bundle (same as research DataExtractor)."""
    de = DataExtractor()
    return {"tool": "clinicaltrials_fda", "bundle": de.run(ticker, company)}


def tool_price_backtest_proxy(ticker: str, lookback_days: int = 60) -> Dict[str, Any]:
    """
    Historical price statistics as a lightweight 'backtester' proxy:
    realized vol, trailing return, max drawdown on daily closes.
    """
    try:
        bars = fetch_yahoo_daily_bars(
            ticker, range_=f"{min(lookback_days, 730)}d", max_rows=lookback_days + 5
        )
    except Exception as e:
        return {"tool": "yahoo_backtest_proxy", "ticker": ticker, "error": str(e)}
    if not bars:
        return {"tool": "yahoo_backtest_proxy", "ticker": ticker, "error": "no_bars"}
    closes = [float(b["close"]) for b in bars if b.get("close") is not None]
    if len(closes) < 5:
        return {"tool": "yahoo_backtest_proxy", "ticker": ticker, "error": "too_few_closes"}
    rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
    vol = float((sum(r * r for r in rets) / max(1, len(rets))) ** 0.5)
    cum_ret = (closes[-1] - closes[0]) / closes[0]
    peak = closes[0]
    max_dd = 0.0
    for c in closes:
        peak = max(peak, c)
        dd = (peak - c) / peak
        max_dd = max(max_dd, dd)
    return {
        "tool": "yahoo_backtest_proxy",
        "ticker": ticker,
        "trailing_return": round(cum_ret, 4),
        "realized_vol_daily": round(vol, 4),
        "max_drawdown": round(max_dd, 4),
        "n_days": len(closes),
    }


def list_tool_specs() -> List[Dict[str, str]]:
    return [
        {"name": "sec_material_events", "desc": "SEC 8-K text for financing / clinical updates"},
        {"name": "clinical_trials_fda_bundle", "desc": "CT.gov + FDA calendar extraction"},
        {"name": "price_backtest_proxy", "desc": "Daily bars → vol / return / drawdown"},
    ]
