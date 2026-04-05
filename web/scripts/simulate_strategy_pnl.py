#!/usr/bin/env python3
"""
Thin wrapper so you can run from the `web/` folder:

  python3 scripts/simulate_strategy_pnl.py

The real script lives at repo root: ../../scripts/simulate_strategy_pnl.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL = _REPO_ROOT / "scripts" / "simulate_strategy_pnl.py"


def main() -> None:
    if not _REAL.is_file():
        print(f"Missing {_REAL}", file=sys.stderr)
        sys.exit(1)
    raise SystemExit(subprocess.call([sys.executable, str(_REAL)] + sys.argv[1:], cwd=str(_REPO_ROOT)))


if __name__ == "__main__":
    main()
