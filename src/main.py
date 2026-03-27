from __future__ import annotations

import argparse

from src.agents.coordinator_agent import run_coordinator_agent
from src.agents.ingestion_agent import run_ingestion_agent
from src.agents.market_impact_agent import run_market_impact_agent
from src.agents.regulatory_agent import run_regulatory_agent
from src.agents.signal_agent import run_signal_agent
from src.agents.trial_progress_agent import run_trial_progress_agent
from src.utils.report_writer import final_report_to_pretty_json, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Biotech multi-agent signal runner")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. XBI")
    parser.add_argument("--company", required=True, help="Company or drug sponsor name")
    parser.add_argument(
        "--cash-runway-months",
        type=int,
        default=18,
        help="Estimated cash runway in months (risk factor).",
    )
    parser.add_argument(
        "--single-asset-exposure",
        action="store_true",
        help="Flag if company is highly dependent on one asset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ingestion_msg, raw_data = run_ingestion_agent(args.ticker, args.company)
    trial_msg = run_trial_progress_agent(args.ticker, args.company, raw_data)
    reg_msg = run_regulatory_agent(args.ticker, args.company, raw_data)
    risk_profile = {
        "cash_runway_months": args.cash_runway_months,
        "single_asset_exposure": args.single_asset_exposure,
    }
    market_msg = run_market_impact_agent(
        args.ticker, args.company, trial_msg, reg_msg, raw_data, risk_profile
    )
    signal_msg = run_signal_agent(args.ticker, args.company, trial_msg, reg_msg, market_msg)

    report = run_coordinator_agent(
        [ingestion_msg, trial_msg, reg_msg, market_msg, signal_msg]
    )
    path = write_report(report)

    print(final_report_to_pretty_json(report))
    print(f"\nSaved report: {path}")


if __name__ == "__main__":
    main()
