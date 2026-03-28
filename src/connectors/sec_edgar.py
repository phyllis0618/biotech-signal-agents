from __future__ import annotations

import re
from typing import List, Optional

import requests

SEC_HEADERS = {
    "User-Agent": "BiotechSignalAgents/0.1 (research@localhost)",
    "Accept-Encoding": "gzip, deflate",
}


def _strip_html(html: str) -> str:
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_cik_for_ticker(ticker: str) -> Optional[str]:
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        data = requests.get(url, headers=SEC_HEADERS, timeout=30).json()
    except Exception:
        return None
    rows: List[dict] = []
    if isinstance(data, dict):
        rows = [v for v in data.values() if isinstance(v, dict)]
    elif isinstance(data, list):
        rows = [r for r in data if isinstance(r, dict)]
    for row in rows:
        if str(row.get("ticker", "")).upper() == ticker.upper():
            return str(row.get("cik_str", "")).zfill(10)
    return None


def fetch_latest_8k_plain_text(ticker: str, max_chars: int = 80000) -> str:
    cik = get_cik_for_ticker(ticker)
    if not cik:
        return ""
    cik_int = str(int(cik))
    try:
        sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        js = requests.get(sub_url, headers=SEC_HEADERS, timeout=30).json()
    except Exception:
        return ""

    recent = js.get("filings", {}).get("recent", {})
    forms: List[str] = recent.get("form", [])
    accession: List[str] = recent.get("accessionNumber", [])
    primary: List[str] = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        if i >= len(accession) or i >= len(primary):
            continue
        acc = accession[i]
        doc = primary[i]
        acc_nodash = acc.replace("-", "")
        file_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
        try:
            resp = requests.get(file_url, headers=SEC_HEADERS, timeout=45)
            resp.raise_for_status()
            plain = _strip_html(resp.text)
            return plain[:max_chars]
        except Exception:
            continue
    return ""


def build_fundamental_context(
    ticker: str,
    company: str,
    ctgov_studies: List[dict],
    fda_calendar_events: List[dict],
    filing_text: str,
) -> str:
    lines = [
        f"Ticker: {ticker}",
        f"Company: {company}",
        "",
        "=== SEC filing excerpt (8-K) ===",
        filing_text or "(no 8-K text fetched)",
        "",
        "=== ClinicalTrials.gov study titles (metadata) ===",
    ]
    for study in ctgov_studies[:15]:
        title = (
            study.get("protocolSection", {})
            .get("identificationModule", {})
            .get("briefTitle", "")
        )
        nct = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
        if title:
            lines.append(f"- {nct} {title}")
    lines.append("")
    lines.append("=== FDA calendar notices (metadata) ===")
    for ev in fda_calendar_events[:10]:
        lines.append(str(ev))
    return "\n".join(lines)
