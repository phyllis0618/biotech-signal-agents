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

Open **http://localhost:3001**.

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
