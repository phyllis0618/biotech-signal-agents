# Biotech Signal Agents

Multi-agent pipeline for biotech long/short research signals using:

- FDA data (`openFDA` as baseline)
- ClinicalTrials.gov API
- PureGlobal clinical trials source (adapter scaffold)

## What this project includes

- A runnable Python skeleton
- Message protocol for agent-to-agent communication
- Coordinator agent prompt and summary logic
- Data ingestion templates for FDA, ClinicalTrials.gov, and PureGlobal
- Rule-based signal generation (long/short/no-trade)
- Event-window aware scoring (recent 180-day activity)
- Company risk factors (cash runway and single-asset exposure)
- Simple event-driven backtest utility

## Quick start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy env template:

```bash
cp .env.example .env
```

4. Run one cycle for a ticker:

```bash
python -m src.main --ticker XBI --company "Example Biotech"
```

The output report will be written to `outputs/`.

Optional risk flags:

```bash
python -m src.main --ticker XBI --company "Example Biotech" --cash-runway-months 10 --single-asset-exposure
```

## Backtest utility

Prepare a CSV with columns:

- `signal` (`long|short|no_trade`)
- `ret_1d`
- `ret_1w`
- `ret_1m`

Run:

```bash
python scripts/backtest_events.py --input-csv data/events.csv
```

## Architecture

- `src/agents/ingestion_agent.py`
- `src/agents/trial_progress_agent.py`
- `src/agents/regulatory_agent.py`
- `src/agents/market_impact_agent.py`
- `src/agents/signal_agent.py`
- `src/agents/coordinator_agent.py`

Each agent emits a normalized message (`AgentMessage`) and the coordinator composes the final report.

## Notes

- This is a research assistant, not financial advice.
- Start with paper trading and event backtesting before any real deployment.
- The PureGlobal adapter currently expects structured parsing rules to be customized for your access pattern and terms of use.

## Push to GitHub

1. Initialize git locally:

```bash
git init
git add .
git commit -m "Initial biotech multi-agent signal pipeline"
```

2. Create a new empty repo on GitHub, then connect remote:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If you use GitHub CLI:

```bash
gh repo create <your-repo> --private --source . --remote origin --push
```
