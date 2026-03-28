from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.analytics.impact import estimate_signal_price_impact
from src.connectors.market_data import fetch_yahoo_snapshot
from src.pipeline import run_pipeline
from src.universe import load_biotech_universe
from src.utils.history_store import append_signal_history, read_signal_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run signal pipeline on biotech universe")
    parser.add_argument(
        "--universe-csv",
        default="data/biotech_universe_sample.csv",
        help="CSV path with ticker,company,cash_runway_months,single_asset_exposure",
    )
    parser.add_argument("--output-csv", default="outputs/biotech_universe_latest.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    universe = load_biotech_universe(args.universe_csv)
    rows = []
    history = read_signal_history(limit=20000)
    impact_map = estimate_signal_price_impact(history)

    for item in universe:
        ticker = item["ticker"]
        company = item["company"]
        _, report, _ = run_pipeline(
            ticker=ticker,
            company=company,
            cash_runway_months=item["cash_runway_months"],
            single_asset_exposure=item["single_asset_exposure"],
        )
        snap = fetch_yahoo_snapshot(ticker)
        expected_impact_pct = impact_map.get(report.final_signal, 0.0)
        row = {
            "as_of": datetime.utcnow().isoformat(),
            "ticker": ticker,
            "company": company,
            "final_signal": report.final_signal,
            "confidence": report.confidence,
            "horizon": report.horizon,
            "price": snap["price"],
            "change_pct": snap["change_pct"],
            "expected_impact_pct": expected_impact_pct,
            "risk_flags": "; ".join(report.risk_flags),
        }
        rows.append(row)
        append_signal_history(row)

    df = pd.DataFrame(rows)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved universe output: {out}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
