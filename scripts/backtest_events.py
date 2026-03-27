from __future__ import annotations

import argparse
import csv
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple event-driven backtest from labeled signals."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="CSV with columns: signal,ret_1d,ret_1w,ret_1m",
    )
    return parser.parse_args()


def apply_signal_return(signal: str, raw_return: float) -> float:
    if signal == "long":
        return raw_return
    if signal == "short":
        return -raw_return
    return 0.0


def main() -> None:
    args = parse_args()
    pnl_1d = []
    pnl_1w = []
    pnl_1m = []
    count = 0

    with open(args.input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            signal = row.get("signal", "no_trade").strip().lower()
            r1d = float(row.get("ret_1d", 0.0))
            r1w = float(row.get("ret_1w", 0.0))
            r1m = float(row.get("ret_1m", 0.0))
            pnl_1d.append(apply_signal_return(signal, r1d))
            pnl_1w.append(apply_signal_return(signal, r1w))
            pnl_1m.append(apply_signal_return(signal, r1m))
            count += 1

    if count == 0:
        print("No rows found.")
        return

    print(f"rows={count}")
    print(f"avg_pnl_1d={mean(pnl_1d):.4f}")
    print(f"avg_pnl_1w={mean(pnl_1w):.4f}")
    print(f"avg_pnl_1m={mean(pnl_1m):.4f}")


if __name__ == "__main__":
    main()
