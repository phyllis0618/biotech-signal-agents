from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


def build_catalyst_rows(
    ctgov_studies: List[Dict],
    ticker: str,
    company: str,
    fda_calendar_events: Optional[List[Dict]] = None,
    max_rows: int = 30,
) -> List[Dict]:
    rows: List[Dict] = []
    for study in ctgov_studies[:max_rows]:
        protocol = study.get("protocolSection", {})
        ident = protocol.get("identificationModule", {})
        status_mod = protocol.get("statusModule", {})
        design_mod = protocol.get("designModule", {})

        phase = _phase_text(design_mod.get("phases", []))
        status = str(status_mod.get("overallStatus", "N/A"))
        completion = (
            status_mod.get("primaryCompletionDateStruct", {}).get("date")
            or status_mod.get("completionDateStruct", {}).get("date")
            or ""
        )

        catalyst_type = _infer_catalyst_type(phase=phase, status=status)
        eta_bucket = _estimate_eta_bucket(completion, status)
        if not catalyst_type:
            continue
        days_to_target, priority_tier = _priority_fields(completion, status)
        rows.append(
            {
                "ticker": ticker,
                "company": company,
                "nct_id": ident.get("nctId", ""),
                "brief_title": ident.get("briefTitle", ""),
                "phase": phase or "N/A",
                "status": status,
                "catalyst_type": catalyst_type,
                "target_date": completion,
                "eta_bucket": eta_bucket,
                "days_to_target": days_to_target,
                "priority_tier": priority_tier,
            }
        )
    if fda_calendar_events:
        for item in fda_calendar_events[: max_rows // 2]:
            target = item.get("target_date", "")
            eta_bucket = _estimate_eta_bucket(target, "calendar")
            days_to_target, priority_tier = _priority_fields(target, "calendar")
            rows.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "nct_id": "",
                    "brief_title": item.get("title", ""),
                    "phase": "Regulatory",
                    "status": "Calendar Notice",
                    "catalyst_type": item.get("type", "FDA Calendar"),
                    "target_date": target,
                    "eta_bucket": eta_bucket,
                    "days_to_target": days_to_target,
                    "priority_tier": priority_tier,
                }
            )
    return rows


def dedupe_fda_calendar_rows(rows: List[Dict]) -> List[Dict]:
    """Same FDA notice was attached per ticker; keep one row per notice."""
    seen = set()
    out: List[Dict] = []
    for row in rows:
        ctype = str(row.get("catalyst_type", ""))
        title = str(row.get("brief_title", ""))
        if "FDA" in ctype or "PDUFA" in ctype or row.get("phase") == "Regulatory":
            key = (ctype, title, str(row.get("target_date", "")))
            if key in seen:
                continue
            seen.add(key)
        out.append(row)
    return out


def split_priority_watchlists(rows: List[Dict]) -> Dict[str, List[Dict]]:
    """Buckets for direct dashboard output: week, 8-30d, 31-90d (no overlap)."""
    week: List[Dict] = []
    d8_30: List[Dict] = []
    d31_90: List[Dict] = []
    for row in rows:
        d = row.get("days_to_target")
        if d is None:
            continue
        if d < 0:
            continue
        if d <= 7:
            week.append(row)
        elif d <= 30:
            d8_30.append(row)
        elif d <= 90:
            d31_90.append(row)

    def _sort_key(r: Dict) -> Tuple:
        dd = r.get("days_to_target")
        if dd is None:
            return (9999, r.get("ticker", ""))
        return (dd, r.get("ticker", ""))

    return {
        "next_7d": sorted(week, key=_sort_key),
        "next_8_30d": sorted(d8_30, key=_sort_key),
        "next_31_90d": sorted(d31_90, key=_sort_key),
    }


def _priority_fields(target_date: str, status: str) -> Tuple[Optional[int], str]:
    dt = _parse_date(target_date)
    if not dt:
        sl = status.lower()
        if "recruit" in sl:
            return None, "active-enrollment"
        if "completed" in sl:
            return None, "awaiting-results"
        return None, "unknown-date"
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    delta = (dt - today).days
    if delta < 0:
        return delta, "past"
    if delta <= 7:
        return delta, "this_week"
    if delta <= 30:
        return delta, "next_30d"
    if delta <= 90:
        return delta, "next_90d"
    return delta, "later"


def _phase_text(phases: object) -> str:
    if isinstance(phases, list):
        return " / ".join([str(x) for x in phases if x])
    return str(phases or "")


def _infer_catalyst_type(phase: str, status: str) -> str:
    phase_low = phase.lower()
    status_low = status.lower()
    if "phase 3" in phase_low and ("recruit" in status_low or "active" in status_low):
        return "Phase 3 readout risk"
    if "phase 2" in phase_low and ("recruit" in status_low or "active" in status_low):
        return "Phase 2 data update"
    if "completed" in status_low:
        return "Post-completion publication/readout"
    if "not yet recruiting" in status_low:
        return "Trial initiation"
    return ""


def _estimate_eta_bucket(target_date: str, status: str) -> str:
    target_dt = _parse_date(target_date)
    if target_dt:
        delta_days = (target_dt - datetime.utcnow()).days
        if delta_days <= 30:
            return "<=30d"
        if delta_days <= 90:
            return "31-90d"
        if delta_days <= 180:
            return "91-180d"
        return ">180d"
    status_low = status.lower()
    if "recruit" in status_low:
        return "active-enrollment"
    if "completed" in status_low:
        return "awaiting-results"
    return "unknown"


def _parse_date(value: str):
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    # Sometimes date comes as "Month YYYY"
    try:
        return datetime.strptime(value, "%B %Y") + timedelta(days=15)
    except ValueError:
        return None
