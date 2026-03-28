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

## Visualization frontend

Run Streamlit dashboard (global universe view):

```bash
python -m streamlit run frontend/app.py
```

What the dashboard shows:

- Agent stream flow graph (`ingestion -> ... -> coordinator`)
- Per-ticker signal stream with confidence and evidence counts
- Cross-ticker signal board (`long/short/no_trade`)
- Market snapshot for universe tickers (use sidebar **Refresh market prices**; no auto full-page reload)
- Intraday multi-ticker price chart (5m interval)
- Signal history time series (`confidence` + mapped signal score)
- Universe mode: run all biotech names from CSV and compare signal vs price move
- Institutional scorecard (pipeline depth/execution, financing quality, composite score)
- Pipeline monitor table for each ticker (`NCT ID`, phase, status, completion dates)
- Catalyst calendar from trial phases/status/target dates + FDA AdCom/PDUFA notices
- Event-window attribution table (`avg_change_pct`, `hit_rate`) by signal and confidence bucket
- Risk-managed top long/short baskets with confidence and financing thresholds

## Biotech universe run

Prepare CSV with:

- `ticker`
- `company` (used for clinical/regulatory query)
- `cash_runway_months`
- `single_asset_exposure`

Then run:

```bash
python -m scripts.run_biotech_universe --universe-csv data/biotech_universe_sample.csv
```

### Auto-build US biotech universe (recommended)

Common data source used here: Nasdaq Stock Screener API.

```bash
python -m scripts.build_us_biotech_universe --output-csv data/biotech_universe_auto.csv
python -m scripts.run_biotech_universe --universe-csv data/biotech_universe_auto.csv
```

Output:

- `outputs/biotech_universe_latest.csv`
- historical accumulation in `outputs/signal_history.jsonl`

## Fundamental extraction (clinical / FDA / cash / deals / TAM)

Five prompts live in `src/prompts/fundamental_prompts.py`. JSON schemas: `src/models/fundamental_schemas.py`.

- Pulls latest **8-K** text from **SEC EDGAR** when possible (`src/connectors/sec_edgar.py`).
- Runs **OpenAI JSON mode** when `OPENAI_API_KEY` is set, or **Ollama** locally when `USE_OLLAMA=1` (no cloud key).  
  **Cursor’s in-editor model is not available to `python` / Streamlit** — use Ollama on your machine if you want zero API key.
- Feeds the **`fundamental`** agent (`src/agents/fundamental_agent.py`), which aggregates into `signal_hint` and is blended in `signal_agent` (15% higher weight on fundamentals).

Ollama is the default in `.env.example`. After cloning, copy once:

```bash
cp .env.example .env
```

If you already have a local `.env` with `USE_OLLAMA=1`, fundamental extraction uses **Ollama only** until you set `OPENAI_API_KEY`.

## Architecture

- `src/agents/ingestion_agent.py`
- `src/agents/fundamental_agent.py`
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
