#!/usr/bin/env python3
"""
Zero-argument demo: run Agentic PM on several large-cap biotech names and write
`outputs/pm_dashboard_state.json` for the Next.js dashboard.

Usage (from repo root):
  python3 scripts/run_demo_for_frontend.py

Then:
  cd web && npm install && npm run dev
  # or: streamlit run frontend/app.py  (set Universe CSV to data/demo_biotech_frontend.csv)
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
    print(f"Wrote {demo_csv} for Streamlit default universe.")

    batch_runs: List[Dict[str, Any]] = []
    last_report = None
    last_rs = None

    for i, (ticker, company, runway, sae) in enumerate(DEMO_UNIVERSE):
        write_dash = False
        print(f"[{i + 1}/{len(DEMO_UNIVERSE)}] {ticker} …", flush=True)
        try:
            _msgs, report, _raw, rs = run_portfolio_pm_cycle(
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
            last_report, last_rs = report, rs
        except Exception as e:
            batch_runs.append({"ticker": ticker, "error": str(e)})
            print(f"  ERROR {ticker}: {e}", flush=True)

    if last_report is not None and last_rs is not None:
        path = write_pm_dashboard_state(last_report, last_rs, batch_runs=batch_runs)
        print(f"\nDashboard JSON: {path}")
        print("Primary timeline in JSON is from last successful ticker:", last_report.ticker)
    else:
        print("No successful runs — check network / API keys.")

    print("\nNext.js:  cd web && npm install && npm run dev  →  http://localhost:3001")
    print("Streamlit: streamlit run frontend/app.py  → set Universe CSV to data/demo_biotech_frontend.csv")


if __name__ == "__main__":
    main()
