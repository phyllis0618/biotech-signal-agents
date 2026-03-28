from __future__ import annotations

import argparse

from src.pipeline import run_pipeline
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
    _, report, _ = run_pipeline(
        args.ticker,
        args.company,
        cash_runway_months=args.cash_runway_months,
        single_asset_exposure=args.single_asset_exposure,
    )
    path = write_report(report)

    print(final_report_to_pretty_json(report))
    print(f"\nSaved report: {path}")


if __name__ == "__main__":
    main()
