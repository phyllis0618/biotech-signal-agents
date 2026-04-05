# Biotech Signal Agents

Biotech multi-agent research pipeline with a **single web UI**: the **Next.js app** in `web/`.

**Streamlit was removed** — it overlapped with Next (both were dashboards on the same signals). Next stays because it matches the Agentic PM JSON (`pm_dashboard_state.json`) and is easier to style as a product UI.

---

## Next dashboard — what each step does

### Step 1 — Python environment

**What it does:** Installs the backend that talks to FDA/CT.gov/Yahoo, runs agents, RL hooks, and writes JSON for the UI.

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

- **`.env`**: API keys / `USE_OLLAMA=1` for local LLM instead of OpenAI (fundamentals and other LLM steps).

---

### Step 2 — Generate `outputs/pm_dashboard_state.json`

**What it does:** For each row in `data/demo_biotech_frontend.csv`, runs the full **Agentic PM cycle** (ReasoningManager: plan → tools → memory → critic → decision), then the main signal pipeline, risk text, approval queue entry, and finally **one JSON file** the Next app reads.

```bash
python3 scripts/run_demo_for_frontend.py
```

- **No CLI arguments.** Takes a few minutes (network + optional LLM).
- **Outputs:** `outputs/pm_dashboard_state.json` (primary), `data/demo_biotech_frontend.csv` (refreshed copy of the universe list), `outputs/approval_queue.jsonl` (audit).

**Refresh:** Run the same command again when you want updated numbers.

---

### Step 3 — Start the Next.js server

**What it does:** Serves the React UI. It does **not** run the biotech pipeline; it only **loads** the JSON from disk via `web/app/api/state/route.ts` (reads `../outputs/pm_dashboard_state.json` relative to the `web/` folder).

```bash
cd web
npm install
npm run dev
```

Open **http://localhost:3002** (port set in `web/package.json`; change if busy).

### Step 4 (optional) — 100d window: 70d train + 30d real OOS PnL

**What it does:** Last **100** daily returns from Yahoo: **TD tabular Q trains** on the first **70** days; **in-sample BT** is that same 70d segment; **OOS forward** is the last **30** real return days (no simulated prices). Optional `--loop` polls Yahoo and appends when a new daily bar appears. Produces `outputs/pnl_simulation_state.json`; the Next app reads it via `/api/pnl` (polls every 3s).

From **repo root** (recommended):

```bash
python3 scripts/simulate_strategy_pnl.py
python3 scripts/simulate_strategy_pnl.py --ticker XBI --total-days 100 --train-days 70 --oos-days 30
python3 scripts/simulate_strategy_pnl.py --loop --interval 60
```

From **`web/`** (wrapper + npm):

```bash
python3 scripts/simulate_strategy_pnl.py
npm run sim:pnl
npm run sim:pnl:loop
```

This is **not** the multi-agent alpha — it is a **quant-style toy** for charting PnL mechanics. For agent signals, use Step 2.

### Step 5 — Interactive walk-forward (30 OOS days × you decide)

**What it does:** Uses the **same 100-day Yahoo window** as Step 4 (70 train + 30 forward / OOS). The drill replays only the **last 30 sessions** (OOS segment). Each step shows momentum→coordinator-style signal, **tabular Q** suggests an action (ε-greedy; **locked** until you change ε or re-roll). You **Approve / Reject / Defer** like trader review: Q updates (`reward` ±1 / 0); **Approve** also moves paper equity by `RL action × daily return`.

- **UI:** open **http://localhost:3002/sim** (after `npm run dev` in `web/`).
- **Or CLI first:** `python3 scripts/build_interactive_sim.py --total-days 100 --oos-days 30` → writes `outputs/interactive_sim.json`; the page reads/writes it via `/api/sim`.

**RL inputs on that page:** sliders for **ε**, **tabular Q learning rate**, **γ** (reserved). This is the **same tabular Q** semantics as `src/trading/q_learning.py`, not the PPO `rl_config.json` (that’s the Gym env).

---

## What you see on the dashboard

| Area | Meaning |
|------|---------|
| **Universe** | Demo tickers with Yahoo price/change plus CSV fields (runway months, single-asset exposure). |
| **Institutional scorecard** | Pipeline depth/execution, financing quality, signal alignment, composite — same analytics as before. |
| **Catalyst calendar** | Trial + FDA calendar rows (deduped), sorted by `days_to_target`. |
| **Batch signals** | Per-ticker PM output from the demo script. |
| **Reasoning timeline** | Planning → tools → memory → critic → decision (primary ticker = last successful run). |
| **Alpha confidence** | Confidence for that primary ticker. |
| **Counter-thesis / Risk / PM** | Critic text, VaR-style line, PM weight preview. |
| **RL hyperparameters** | From `config/rl_config.json`. |
| **HITL** | `trade_id` and review status (no auto-execution). |
| **100d + paper PnL** | Top section: backtest return / forward sim / optional live loop from Step 4. |

---

## Optional CLI (no browser)

Single ticker, Agentic PM, writes report files under `outputs/`:

```bash
python -m src.main --ticker AMGN --company "Amgen Inc." --agentic-pm
```

Simulated multi-ticker backtest (separate from the dashboard JSON):

```bash
python3 scripts/sim_backtest_demo.py
python3 scripts/sim_backtest_demo.py --fast
```

---

## Notes

Research tooling only — not investment advice.
