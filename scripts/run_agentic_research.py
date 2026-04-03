#!/usr/bin/env python3
"""CLI: Input -> Plan -> Research -> Reflect -> Final Alpha (LangGraph or FSM fallback)."""

from __future__ import annotations

import argparse

from src.agents.research_manager import research_report_json, run_agentic_research


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic research framework")
    p.add_argument("--ticker", required=True)
    p.add_argument("--company", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    state = run_agentic_research(args.ticker, args.company)
    print(research_report_json(state))


if __name__ == "__main__":
    main()
