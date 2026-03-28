from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from src.connectors.nasdaq_screener import fetch_us_biotech_from_nasdaq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-build US biotech stock universe from Nasdaq screener"
    )
    parser.add_argument(
        "--output-csv",
        default="data/biotech_universe_auto.csv",
        help="Output CSV for downstream batch pipeline",
    )
    parser.add_argument(
        "--default-cash-runway-months",
        type=int,
        default=18,
        help="Default runway used for names without custom fundamentals",
    )
    parser.add_argument(
        "--default-single-asset-exposure",
        action="store_true",
        help="Set default single-asset exposure to true for all names",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        rows = fetch_us_biotech_from_nasdaq()
    except Exception as exc:
        print(f"Nasdaq fetch failed: {exc}")
        rows = []
    if not rows:
        # Fallback to local sample so downstream flow still runs.
        sample_path = Path("data/biotech_universe_sample.csv")
        if not sample_path.exists():
            print("No biotech rows fetched and no local sample available.")
            return
        with sample_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print("Using fallback sample universe due to Nasdaq fetch issue.")

    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker"]).sort_values("ticker")
    if "cash_runway_months" not in df.columns:
        df["cash_runway_months"] = args.default_cash_runway_months
    if "single_asset_exposure" not in df.columns:
        df["single_asset_exposure"] = bool(args.default_single_asset_exposure)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["ticker", "company", "cash_runway_months", "single_asset_exposure"]].to_csv(
        out, index=False
    )
    print(f"Saved auto universe: {out} (rows={len(df)})")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
