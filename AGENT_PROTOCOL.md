# Agent Message Protocol

All agents emit `AgentMessage` objects with these required fields:

- `agent`: one of `ingestion`, `trial_progress`, `regulatory`, `market_impact`, `signal`, `coordinator`
- `ticker`: target symbol
- `company`: target company string
- `summary`: short plain-language summary
- `confidence`: integer 0-100
- `signal_hint`: `bullish | bearish | neutral`
- `evidence`: list of references (`source`, `title`, `url`, `snippet`)

## Coordinator rules

1. Consume all upstream messages.
2. Compute directional vote from `signal_hint`.
3. Penalize confidence when bullish and bearish messages both appear.
4. Output `FinalReport` with:
   - `final_signal`: `long | short | no_trade`
   - `confidence`
   - `horizon`
   - `key_points`
   - `risk_flags`
   - `evidence`
5. Incorporate event-window context (e.g., recent 180-day updates) and company risk context before final directional output.

## Extension points

- Add earnings/SEC parser as a new upstream agent.
- Add model-based ranking in `signal_agent`.
- Add historical backtesting and calibration for confidence scoring.
- Add explicit PDUFA / AdCom event calendars as additional event agents.
