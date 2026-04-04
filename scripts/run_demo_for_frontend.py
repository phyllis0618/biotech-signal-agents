#!/usr/bin/env python3
"""
Zero-argument demo: run Agentic PM on several large-cap biotech names and write
`outputs/pm_dashboard_state.json` for the Next.js dashboard (includes universe, catalyst, scorecard).

Usage (from repo root):
  python3 scripts/run_demo_for_frontend.py

Then:
  cd web && npm install && npm run dev
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Repo root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from src.analytics.catalyst import build_catalyst_rows, dedupe_fda_calendar_rows
from src.analytics.pipeline_snapshot import summarize_pipeline_strength
from src.analytics.institutional_score import compute_institutional_scorecard
from src.connectors.market_data import fetch_yahoo_snapshots
from src.portfolio_pipeline import run_portfolio_pm_cycle, write_pm_dashboard_state

# Common liquid biotech + XBI — no CLI args (3 names keeps runtime reasonable).
DEMO_UNIVERSE = [
    ("AMGN", "Amgen Inc.", 24, False),
    ("GILD", "Gilead Sciences Inc.", 24, False),
    ("XBI", "SPDR S&P Biotech ETF", 18, False),
]


def main() -> None:
    demo_csv = _ROOT / "data" / "demo_biotech_frontend.csv"
    demo_csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ticker,company,cash_runway_months,single_asset_exposure\n"]
    for t, c, m, s in DEMO_UNIVERSE:
        lines.append(f"{t},{c},{m},{str(s).lower()}\n")
    demo_csv.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {demo_csv} (universe list for the Python demo).")

    batch_runs: List[Dict[str, Any]] = []
    catalyst_rows: List[Dict[str, Any]] = []
    scorecard_rows: List[Dict[str, Any]] = []
    last_report = None
    last_rs = None

    for i, (ticker, company, runway, sae) in enumerate(DEMO_UNIVERSE):
        print(f"[{i + 1}/{len(DEMO_UNIVERSE)}] {ticker} …", flush=True)
        try:
            _msgs, report, raw, rs = run_portfolio_pm_cycle(
                ticker,
                company,
                cash_runway_months=runway,
                single_asset_exposure=sae,
                write_dashboard=False,
            )
            batch_runs.append(
                {
                    "ticker": report.ticker,
                    "company": report.company,
                    "final_signal": report.final_signal,
                    "confidence": report.confidence,
                    "trade_id": report.trade_id,
                    "risk_status": report.risk_status[:200] if report.risk_status else "",
                }
            )
            ctgov = raw.get("ctgov_studies", []) or []
            catalyst_rows.extend(
                build_catalyst_rows(
                    ctgov_studies=ctgov,
                    ticker=ticker,
                    company=company,
                    fda_calendar_events=raw.get("fda_calendar_events", []),
                    max_rows=15,
                )
            )
            ps = summarize_pipeline_strength(ctgov)
            sc = compute_institutional_scorecard(
                pipeline_summary=ps,
                report_confidence=report.confidence,
                signal=report.final_signal,
                cash_runway_months=runway,
                single_asset_exposure=sae,
            )
            scorecard_rows.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "final_signal": report.final_signal,
                    "confidence": report.confidence,
                    "total_trials": ps["total_trials"],
                    "phase23_trials": ps["phase23_trials"],
                    "completed_trials": ps["completed_trials"],
                    **sc,
                }
            )
            last_report, last_rs = report, rs
        except Exception as e:
            batch_runs.append({"ticker": ticker, "error": str(e)})
            print(f"  ERROR {ticker}: {e}", flush=True)

    catalyst_deduped = dedupe_fda_calendar_rows(catalyst_rows)[:80]

    tickers = [t for t, _, _, _ in DEMO_UNIVERSE]
    snapshots = fetch_yahoo_snapshots(tickers)
    universe_snapshot: List[Dict[str, Any]] = []
    for ticker, company, runway, sae in DEMO_UNIVERSE:
        snap = next((s for s in snapshots if s.get("ticker") == ticker), {})
        universe_snapshot.append(
            {
                "ticker": ticker,
                "company": company,
                "cash_runway_months": runway,
                "single_asset_exposure": sae,
                "price": snap.get("price"),
                "change_pct": snap.get("change_pct"),
                "change": snap.get("change"),
            }
        )

    if last_report is not None and last_rs is not None:
        path = write_pm_dashboard_state(
            last_report,
            last_rs,
            batch_runs=batch_runs,
            universe_snapshot=universe_snapshot,
            catalyst_calendar=catalyst_deduped,
            institutional_scorecard=scorecard_rows,
        )
        print(f"\nDashboard JSON: {path}")
        print("Primary timeline in JSON is from last successful ticker:", last_report.ticker)
    else:
        print("No successful runs — check network / API keys.")

    print("\nNext.js dashboard:  cd web && npm install && npm run dev  →  http://localhost:3001")


if __name__ == "__main__":
    main()
