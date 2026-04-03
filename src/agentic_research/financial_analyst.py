from __future__ import annotations

import re
from typing import Any, Dict

from src.connectors.sec_edgar import fetch_latest_8k_plain_text


class FinancialAnalyst:
    """
    FinancialAnalyst Agent: SEC text heuristics for cash runway vs burn (MVP).
    """

    def run(self, ticker: str, company: str) -> Dict[str, Any]:
        text = ""
        try:
            text = fetch_latest_8k_plain_text(ticker, max_chars=120_000)
        except Exception:
            text = ""

        cash = self._find_number_near_keywords(
            text, ["cash and cash equivalents", "cash and equivalents", "total cash"]
        )
        burn = self._find_number_near_keywords(
            text, ["net loss", "operating loss", "research and development"]
        )

        runway_months: float | None = None
        if cash and burn and burn > 0:
            quarterly_burn = burn / 4.0 if burn > 1e6 else burn
            runway_months = float((cash / max(quarterly_burn, 1e-6)) * 3.0)

        return {
            "ticker": ticker,
            "filing_chars": len(text),
            "cash_proxy_millions": cash,
            "loss_or_rnd_proxy_millions": burn,
            "runway_months_est": runway_months,
            "confidence": 0.75 if len(text) > 5000 else 0.35,
            "note": "Heuristic from latest 8-K text; prefer 10-Q for production.",
        }

    def _find_number_near_keywords(self, text: str, keywords: list[str]) -> float | None:
        low = text.lower()
        for kw in keywords:
            idx = low.find(kw)
            if idx == -1:
                continue
            window = text[idx : idx + 400]
            m = re.search(r"[\$]?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?", window)
            if not m:
                continue
            val = float(m.group(1).replace(",", ""))
            if "billion" in window.lower() or " B" in window:
                val *= 1000.0
            return val
        return None
