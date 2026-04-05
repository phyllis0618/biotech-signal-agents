"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type TimelineStep = {
  phase?: string;
  title?: string;
  detail?: string;
};

type BatchRow = {
  ticker?: string;
  company?: string;
  final_signal?: string;
  confidence?: number;
  trade_id?: string;
  risk_status?: string;
  error?: string;
};

type PnlSimulationState = {
  updated_at?: string;
  last_bar_date?: string;
  ticker?: string;
  company?: string;
  split?: {
    total_return_days?: number;
    train_days?: number;
    oos_days?: number;
    train_date_range?: string[];
    oos_date_range?: string[];
    forward_mode?: string;
  };
  strategy?: {
    name?: string;
    q_table?: string;
    lookback?: number;
    train_episodes?: number;
    note?: string;
    backtest_total_return_pct?: number;
    backtest_max_drawdown_pct?: number;
    backtest_sharpe_like?: number;
  };
  backtest_curve?: { dates?: string[]; equity?: number[] };
  forward_sim?: {
    mode?: string;
    equity?: number[];
    cumulative_pnl_pct?: number;
    labels?: string[];
  };
  live_sim?: { running?: boolean; step?: number; equity?: number[]; note?: string };
  error?: string;
  by_ticker?: Record<string, PnlSimulationState>;
  default_ticker?: string;
  _view_ticker?: string;
};

type OvernightReport = {
  by_ticker?: Record<string, OvernightReport>;
  ticker?: string;
  company?: string;
  total_days?: number;
  train_days?: number;
  train_sessions?: number;
  sessions?: number;
  episodes_trained?: number;
  seconds_wall?: number;
  hyperparams?: Record<string, number>;
  best_greedy_eval?: {
    episode?: number;
    equity?: number;
    sharpe_like?: number;
    total_return_pct?: number;
  };
  final_greedy_eval?: {
    equity?: number;
    sharpe_like?: number;
    total_return_pct?: number;
  };
  q_states?: number;
  note?: string;
};

type TabularRlTraining = {
  flow_summary?: string;
  overnight_report?: OvernightReport | null;
  report_available?: boolean;
  q_table_available?: boolean;
  q_state_count?: number | null;
};

type RlTrainApi = {
  flow_summary?: string;
  train_script?: string;
  report_available?: boolean;
  q_table_available?: boolean;
  q_state_count?: number | null;
  overnight_report?: OvernightReport | null;
  report_error?: string | null;
};

type RlDeployed = {
  state_key?: string;
  policy_action?: string;
  final_signal_after_pipeline?: string;
  source_table?: string;
  forward_path?: string;
};

type DashboardState = {
  system_tag?: string;
  ticker?: string;
  company?: string;
  confidence?: number;
  final_signal?: string;
  counter_thesis?: string;
  pm_weights_preview?: string;
  risk_status?: string;
  reasoning_timeline?: TimelineStep[];
  rl_hyperparams?: Record<string, number>;
  rl_training?: { status?: string; last_episode_returns_tail?: number[]; note?: string };
  tabular_rl_training?: TabularRlTraining;
  rl_deployed?: RlDeployed;
  hitl?: { trade_id?: string; execution_status?: string; approval_required?: boolean };
  batch_runs?: BatchRow[];
  batch_note?: string;
  universe?: Record<string, unknown>[];
  catalyst_calendar?: Record<string, unknown>[];
  institutional_scorecard?: Record<string, unknown>[];
  error?: string;
  /** Per-ticker Agentic slices (same logic as batch demo); UI picks one workspace ticker */
  by_ticker?: Record<string, Partial<DashboardState>>;
  default_ticker?: string;
  _view_ticker?: string;
};

function JsonTable({
  title,
  description,
  rows,
  maxRows = 50,
}: {
  title: string;
  description?: string;
  rows?: Record<string, unknown>[];
  maxRows?: number;
}) {
  const slice = useMemo(() => (rows ?? []).slice(0, maxRows), [rows, maxRows]);
  if (!slice.length) {
    return (
      <section className="mb-10 rounded-xl border border-slate-700 bg-slate-900/30 p-6">
        <h2 className="text-lg font-semibold text-white">{title}</h2>
        {description && <p className="mt-1 text-sm text-slate-500">{description}</p>}
        <p className="mt-3 text-sm text-slate-500">No data — run `python3 scripts/run_demo_for_frontend.py`.</p>
      </section>
    );
  }
  const keys = Object.keys(slice[0]);
  return (
    <section className="mb-10 rounded-xl border border-slate-700 bg-slate-900/30 p-6">
      <h2 className="text-lg font-semibold text-white">{title}</h2>
      {description && <p className="mt-1 text-sm text-slate-500">{description}</p>}
      <div className="mt-4 max-h-[480px] overflow-auto">
        <table className="w-full min-w-[640px] text-left text-xs text-slate-300">
          <thead className="sticky top-0 bg-slate-950">
            <tr className="border-b border-slate-600 text-slate-400">
              {keys.map((k) => (
                <th key={k} className="whitespace-nowrap px-2 py-2 font-medium">
                  {k}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {slice.map((row, i) => (
              <tr key={i} className="border-b border-slate-800 hover:bg-slate-900/80">
                {keys.map((k) => (
                  <td key={k} className="max-w-[220px] truncate px-2 py-1.5 font-mono text-[11px]">
                    {formatCell(row[k])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {(rows?.length ?? 0) > maxRows && (
        <p className="mt-2 text-[10px] text-slate-500">Showing first {maxRows} of {rows?.length} rows.</p>
      )}
    </section>
  );
}

function formatCell(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function mergeDashboardView(raw: DashboardState | null, ticker: string): DashboardState | null {
  if (!raw) return null;
  if (!ticker) return raw;
  const slice = raw.by_ticker?.[ticker];
  if (!slice) return raw;
  return { ...raw, ...slice, _view_ticker: ticker };
}

function mergePnlView(raw: PnlSimulationState | null, ticker: string): PnlSimulationState | null {
  if (!raw) return null;
  if (!ticker) return raw;
  const slice = raw.by_ticker?.[ticker];
  if (!slice) return raw;
  return { ...raw, ...slice, _view_ticker: ticker };
}

/**
 * In-window equity only (caller passes sliced series). Y = % change from the **first**
 * point in that series so small absolute moves don’t look like a flat line at ~1.0.
 */
function Sparkline({
  values,
  label,
  xEndLabel,
}: {
  values: number[];
  label: string;
  /** e.g. "70" for last day index in segment */
  xEndLabel?: string;
}) {
  if (!values.length) return null;
  if (values.length === 1) {
    return (
      <div className="mt-2">
        <p className="text-[10px] uppercase text-slate-500">{label}</p>
        <p className="mt-1 text-xs text-slate-500">Single point — no curve.</p>
      </div>
    );
  }

  const plot = values.map((v) => ((v / values[0]) - 1) * 100);
  let min = Math.min(...plot);
  let max = Math.max(...plot);
  let span = max - min || 1e-9;
  const pad = Math.max(span * 0.08, 1e-6);
  min -= pad;
  max += pad;
  span = max - min;

  const W = 400;
  const H = 88;
  const ml = 44;
  const mb = 18;
  const cw = W - ml;
  const ch = H - mb;

  const pts = plot
    .map((v, i) => {
      const x = ml + (i / Math.max(1, values.length - 1)) * cw;
      const y = 4 + ch - ((v - min) / span) * ch;
      return `${x},${y}`;
    })
    .join(" ");

  const gridY = [min, (min + max) / 2, max];
  const fmt = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;

  return (
    <div className="mt-2">
      <p className="text-[10px] uppercase text-slate-500">{label}</p>
      <p className="text-[9px] text-slate-600">Y: % change from first day in this segment (in-window only)</p>
      <svg
        width="100%"
        height={H + 8}
        viewBox={`0 0 ${W} ${H + 8}`}
        className="text-emerald-400"
        role="img"
        aria-label={label}
      >
        {gridY.map((gy, i) => {
          const y = 4 + ch - ((gy - min) / span) * ch;
          return (
            <line
              key={i}
              x1={ml}
              y1={y}
              x2={W}
              y2={y}
              stroke="currentColor"
              strokeOpacity={0.12}
              strokeWidth={1}
            />
          );
        })}
        <polyline fill="none" stroke="currentColor" strokeWidth="1.5" points={pts} />
        {gridY.map((gy, i) => {
          const y = 4 + ch - ((gy - min) / span) * ch;
          return (
            <text
              key={`yl-${i}`}
              x={ml - 4}
              y={y + 3}
              textAnchor="end"
              className="fill-slate-500"
              fontSize={9}
            >
              {fmt(gy)}
            </text>
          );
        })}
        <text x={ml} y={H + 2} className="fill-slate-500" fontSize={9} textAnchor="start">
          1
        </text>
        <text x={W} y={H + 2} className="fill-slate-500" fontSize={9} textAnchor="end">
          {xEndLabel ?? String(values.length)}
        </text>
      </svg>
    </div>
  );
}

export default function Home() {
  const [data, setData] = useState<DashboardState | null>(null);
  const [pnl, setPnl] = useState<PnlSimulationState | null>(null);
  const [rlTrain, setRlTrain] = useState<RlTrainApi | null>(null);
  /** One workspace symbol: Agentic slice + PnL slice + overnight slice (when JSON has `by_ticker`) */
  const [viewTicker, setViewTicker] = useState<string>("");

  useEffect(() => {
    fetch("/api/state")
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData({ error: "fetch failed" }));
  }, []);

  useEffect(() => {
    const load = () =>
      fetch("/api/pnl")
        .then((r) => r.json())
        .then(setPnl)
        .catch(() => setPnl({ error: "pnl fetch failed" }));
    load();
    const id = setInterval(load, 3000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const load = () =>
      fetch("/api/rl-train")
        .then((r) => r.json())
        .then(setRlTrain)
        .catch(() => setRlTrain(null));
    load();
    const id = setInterval(load, 30000);
    return () => clearInterval(id);
  }, []);

  const sortedCatalyst = useMemo(() => {
    const rows = data?.catalyst_calendar ?? [];
    return [...rows].sort((a, b) => {
      const da = typeof a.days_to_target === "number" ? a.days_to_target : 9999;
      const db = typeof b.days_to_target === "number" ? b.days_to_target : 9999;
      return da - db;
    });
  }, [data?.catalyst_calendar]);

  const batchTickerList = useMemo(
    () => [...new Set((data?.batch_runs ?? []).map((r) => r.ticker).filter(Boolean))] as string[],
    [data?.batch_runs]
  );

  const workspaceOptions = useMemo(() => {
    const s = new Set<string>();
    if (data?.by_ticker) Object.keys(data.by_ticker).forEach((k) => s.add(k));
    if (pnl?.by_ticker) Object.keys(pnl.by_ticker).forEach((k) => s.add(k));
    batchTickerList.forEach((t) => s.add(t));
    if (data?.ticker) s.add(data.ticker);
    if (pnl?.ticker) s.add(pnl.ticker);
    return Array.from(s).sort();
  }, [data?.by_ticker, data?.ticker, pnl?.by_ticker, pnl?.ticker, batchTickerList]);

  useEffect(() => {
    if (!workspaceOptions.length) return;
    setViewTicker((prev) => {
      if (prev && workspaceOptions.includes(prev)) return prev;
      if (data?.default_ticker && workspaceOptions.includes(data.default_ticker)) return data.default_ticker;
      if (pnl?.default_ticker && workspaceOptions.includes(pnl.default_ticker)) return pnl.default_ticker;
      return workspaceOptions[0];
    });
  }, [workspaceOptions, data?.default_ticker, pnl?.default_ticker]);

  const mergedData = useMemo(() => mergeDashboardView(data, viewTicker), [data, viewTicker]);
  const mergedPnl = useMemo(() => mergePnlView(pnl, viewTicker), [pnl, viewTicker]);

  const conf = mergedData?.confidence ?? 0;
  const gaugePct = Math.min(100, Math.max(0, conf));

  const overnightReportRaw =
    rlTrain?.overnight_report ?? data?.tabular_rl_training?.overnight_report ?? null;
  const overnightRepForView = useMemo(() => {
    if (!overnightReportRaw) return null;
    const bt = overnightReportRaw.by_ticker?.[viewTicker];
    if (bt) return bt;
    return overnightReportRaw;
  }, [overnightReportRaw, viewTicker]);

  const qStates =
    rlTrain?.q_state_count ??
    data?.tabular_rl_training?.q_state_count ??
    overnightRepForView?.q_states;
  const hasQTable = rlTrain?.q_table_available ?? data?.tabular_rl_training?.q_table_available;
  const hasReport = rlTrain?.report_available ?? data?.tabular_rl_training?.report_available;

  /** Only plot in-window points so the line isn’t padded with flat segments outside train/OOS. */
  const pnlSparkSeries = useMemo(() => {
    const trainDays = mergedPnl?.split?.train_days;
    const oosDays = mergedPnl?.split?.oos_days;
    const btRaw = mergedPnl?.backtest_curve?.equity;
    const fwdRaw = mergedPnl?.forward_sim?.equity;
    let bt: number[] = [];
    if (btRaw?.length) {
      bt =
        typeof trainDays === "number" && trainDays > 0
          ? btRaw.slice(0, trainDays)
          : [...btRaw];
    }
    let fwd: number[] = [];
    if (fwdRaw?.length) {
      const cap =
        typeof oosDays === "number" && oosDays >= 0 ? oosDays + 1 : fwdRaw.length;
      fwd = fwdRaw.slice(0, cap);
    }
    return { bt, fwd };
  }, [mergedPnl]);

  const pnlExplain = useMemo(() => {
    const btEq = pnlSparkSeries.bt;
    const fwdEq = pnlSparkSeries.fwd;
    if (!btEq?.length || !fwdEq?.length) return null;
    const btEnd = btEq[btEq.length - 1];
    const fwdStart = fwdEq[0];
    const fwdEnd = fwdEq[fwdEq.length - 1];
    const totalFromOnePct = (fwdEnd - 1) * 100;
    const btOnlyPct = (btEnd - 1) * 100;
    return { btEnd, fwdStart, fwdEnd, totalFromOnePct, btOnlyPct };
  }, [pnlSparkSeries]);

  const pnlSparkLabels = useMemo(() => {
    const btEq = mergedPnl?.backtest_curve?.equity;
    const fwdEq = mergedPnl?.forward_sim?.equity;
    const trainDays = mergedPnl?.split?.train_days ?? btEq?.length;
    const oosDays =
      mergedPnl?.split?.oos_days ?? (fwdEq && fwdEq.length > 1 ? fwdEq.length - 1 : undefined);
    return {
      bt: `In-sample BT equity (${trainDays ?? "?"} trading days)`,
      oos: `OOS forward equity (${oosDays ?? "?"} real days, chained from BT end)`,
      live: "Live trail (new bar per --loop poll)",
    };
  }, [mergedPnl]);

  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-10 border-b border-slate-700 pb-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-accent-cyan">
              {data?.system_tag ?? "AUTONOMOUS COGNITIVE SYSTEM"}
            </p>
            <h1 className="mt-2 text-3xl font-semibold text-white">
              Agentic Portfolio Management
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-slate-400">
              Universe, catalysts, institutional scorecard, reasoning timeline, RL config from{" "}
              <code className="rounded bg-slate-800 px-1">outputs/pm_dashboard_state.json</code>.
            </p>
            <p className="mt-3 text-sm">
              <Link
                href="/sim"
                className="font-medium text-cyan-400 underline decoration-cyan-600/50 hover:text-cyan-300"
              >
                Walk-forward trader drill
              </Link>
              <span className="text-slate-500">
                {" "}
                — 30-day OOS replay (100d window, 70 train + 30 forward), RL + your approve/reject
              </span>
            </p>
          </div>
          <div className="rounded-lg border border-slate-600 bg-slate-900/80 px-4 py-3 text-right">
            <p className="text-xs text-slate-500">PPO (Gym) status</p>
            <p className="font-mono text-sm text-accent-amber">
              {data?.rl_training?.status ?? "—"}
            </p>
            <p className="mt-1 text-[10px] text-slate-500">
              Tabular Q: {hasQTable ? `${qStates ?? "?"} states` : "no rl_qtable.json"}
            </p>
          </div>
        </div>
      </header>

      <section className="mb-8 rounded-xl border border-slate-600 bg-slate-900/70 p-5" aria-label="Workspace ticker">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 className="text-sm font-semibold text-white">Workspace ticker</h2>
            <p className="mt-1 max-w-2xl text-xs text-slate-500">
              When JSON includes <code className="text-slate-400">by_ticker</code> (from{" "}
              <code className="text-slate-400">run_demo_for_frontend.py</code> and{" "}
              <code className="text-slate-400">simulate_strategy_pnl.py --multi-demo</code>), the whole page below uses
              the same symbol: Agentic timeline, rolling PnL, gauges, and overnight metrics (if that report also has{" "}
              <code className="text-slate-400">by_ticker</code>).
            </p>
          </div>
          <label className="flex flex-col gap-1 text-xs text-slate-500">
            <span>Symbol</span>
            {workspaceOptions.length > 0 ? (
              <select
                className="min-w-[10rem] rounded border border-slate-600 bg-slate-950 px-2 py-1.5 font-mono text-sm text-slate-200"
                value={viewTicker}
                onChange={(e) => setViewTicker(e.target.value)}
              >
                {workspaceOptions.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            ) : (
              <span className="text-slate-500">— run demo scripts to populate tickers</span>
            )}
          </label>
        </div>
        {batchTickerList.length > 0 && (
          <p className="mt-3 text-xs text-slate-500">
            <span className="font-medium text-slate-400">Demo universe (same pipeline each):</span>{" "}
            <span className="font-mono text-slate-300">{batchTickerList.join(", ")}</span>
          </p>
        )}
        {overnightReportRaw &&
          !overnightReportRaw.by_ticker &&
          overnightReportRaw.ticker &&
          viewTicker &&
          overnightReportRaw.ticker !== viewTicker && (
            <p className="mt-3 text-xs text-amber-200/90">
              <code className="text-slate-400">overnight_train_report.json</code> is a single-symbol run (
              <span className="font-mono">{overnightReportRaw.ticker}</span>). Tabular Q metrics below match that file,
              not necessarily the workspace symbol. Re-run training per ticker or add{" "}
              <code className="text-slate-400">by_ticker</code> in the report for full alignment.
            </p>
          )}
        <p className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500">
          Jump:{" "}
          <a href="#section-pnl" className="text-cyan-400/90 underline decoration-cyan-700 hover:text-cyan-300">
            Rolling PnL
          </a>
          <a
            href="#section-agentic-primary"
            className="text-cyan-400/90 underline decoration-cyan-700 hover:text-cyan-300"
          >
            Timeline &amp; gauges
          </a>
          <a href="#section-td-train" className="text-cyan-400/90 underline decoration-cyan-700 hover:text-cyan-300">
            Tabular Q block
          </a>
        </p>
      </section>

      <section id="section-pnl" className="mb-10 rounded-xl border border-emerald-900/40 bg-emerald-950/15 p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-white">Rolling window: train RL + BT, then real OOS</h2>
            <p className="mt-1 text-sm text-slate-400">
              Default: last{" "}
              <span className="text-slate-300">{mergedPnl?.split?.total_return_days ?? 100}</span> daily returns — first{" "}
              <span className="text-slate-300">{mergedPnl?.split?.train_days ?? 70}</span> days{" "}
              <span className="text-slate-500">(TD Q train + in-sample BT)</span>, last{" "}
              <span className="text-slate-300">{mergedPnl?.split?.oos_days ?? 30}</span> days{" "}
              <span className="text-slate-500">(real Yahoo forward, no simulated prices)</span>. State = rolling
              momentum → <code className="rounded bg-slate-800 px-1 text-[11px]">outputs/rl_qtable.json</code>.{" "}
              <code className="rounded bg-slate-800 px-1">outputs/pnl_simulation_state.json</code> ·{" "}
              <code className="rounded bg-slate-800 px-1">python3 scripts/simulate_strategy_pnl.py --multi-demo</code> ·{" "}
              <code className="rounded bg-slate-800 px-1">--loop</code> polls Yahoo and appends when a new daily bar
              appears.
            </p>
            {mergedPnl?.last_bar_date && (
              <p className="mt-1 text-[11px] text-slate-500">
                Latest Yahoo bar in window:{" "}
                <span className="font-mono text-slate-400">{mergedPnl.last_bar_date}</span>
              </p>
            )}
            <p className="mt-2 text-sm text-slate-500">
              {mergedPnl?.strategy?.note ??
                "OOS uses actual historical returns in the window (not a Monte Carlo draw)."}
            </p>
            {mergedPnl?.live_sim?.running && (
              <span className="mt-2 inline-block rounded-full bg-red-500/20 px-2 py-0.5 text-xs font-semibold text-red-400">
                LIVE SIM · step {mergedPnl.live_sim.step}
              </span>
            )}
          </div>
          <div className="text-right font-mono text-xs text-slate-300">
            <div>
              {mergedPnl?.ticker ?? "—"} <span className="text-slate-500">{mergedPnl?.company}</span>
            </div>
            <div className="mt-1 text-emerald-300">
              BT return: {mergedPnl?.strategy?.backtest_total_return_pct?.toFixed(2) ?? "—"}% · MDD:{" "}
              {mergedPnl?.strategy?.backtest_max_drawdown_pct?.toFixed(2) ?? "—"}% · Sharpe≈{" "}
              {mergedPnl?.strategy?.backtest_sharpe_like?.toFixed(2) ?? "—"}
            </div>
            <div className="mt-1 text-cyan-300">
              OOS forward (real): {mergedPnl?.forward_sim?.cumulative_pnl_pct?.toFixed(2) ?? "—"}%{" "}
              <span className="text-slate-500">(vs equity at BT end)</span>
            </div>
            {mergedPnl?.live_sim?.equity && mergedPnl.live_sim.equity.length > 0 && (
              <div className="mt-1 text-amber-300">
                Live trail (last): {mergedPnl.live_sim.equity[mergedPnl.live_sim.equity.length - 1]?.toFixed(4)}
                {mergedPnl.live_sim.equity.length === 1 && !mergedPnl.live_sim.running && (
                  <span className="block text-[10px] font-normal text-slate-500">
                    = OOS end · <code className="text-slate-400">--loop</code> appends when Yahoo adds a new bar
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
        {pnlExplain && (
          <p className="mt-4 max-w-3xl text-xs leading-relaxed text-slate-500">
            <span className="font-medium text-slate-400">Reading the curve:</span> BT is the first{" "}
            {mergedPnl?.split?.train_days ?? 70} return days (in-sample, real prices). Equity goes 1.0000 →{" "}
            <span className="font-mono text-emerald-400/90">{pnlExplain.btEnd.toFixed(4)}</span> (≈+
            {pnlExplain.btOnlyPct.toFixed(2)}% on that segment). The cyan line is the next{" "}
            {mergedPnl?.split?.oos_days ?? 30} <span className="text-slate-300">real</span> Yahoo days (out-of-sample vs
            training), from <span className="font-mono text-slate-300">{pnlExplain.fwdStart.toFixed(4)}</span> →{" "}
            <span className="font-mono text-slate-300">{pnlExplain.fwdEnd.toFixed(4)}</span>. Compound from $1.00: about{" "}
            <span className="text-emerald-400/90">+{pnlExplain.totalFromOnePct.toFixed(2)}%</span>.
          </p>
        )}
        {mergedPnl?.error && !mergedPnl.strategy && (
          <p className="mt-3 text-sm text-amber-200/90">{mergedPnl.error}</p>
        )}
        <div className="mt-4 grid gap-6 md:grid-cols-2">
          {pnlSparkSeries.bt.length > 0 && (
            <Sparkline
              values={pnlSparkSeries.bt}
              label={pnlSparkLabels.bt}
              xEndLabel={String(pnlSparkSeries.bt.length)}
            />
          )}
          {pnlSparkSeries.fwd.length > 0 && (
            <Sparkline
              values={pnlSparkSeries.fwd}
              label={pnlSparkLabels.oos}
              xEndLabel={String(pnlSparkSeries.fwd.length)}
            />
          )}
        </div>
        {mergedPnl?.live_sim?.equity && mergedPnl.live_sim.equity.length > 1 && (
          <Sparkline
            values={mergedPnl.live_sim.equity}
            label={pnlSparkLabels.live}
            xEndLabel={String(mergedPnl.live_sim.equity.length)}
          />
        )}
      </section>

      <section id="section-td-train" className="mb-10 rounded-xl border border-cyan-900/50 bg-slate-900/50 p-6">
        <h2 className="text-lg font-semibold text-white">Tabular RL — backtest training, then forward</h2>
        <p className="mt-2 max-w-3xl text-sm text-slate-400">
          {rlTrain?.flow_summary ??
            data?.tabular_rl_training?.flow_summary ??
            "① 100d Yahoo window: TD Q on first 70 return days → outputs/rl_qtable.json. ② Real OOS: last 30 days in simulate_strategy_pnl.py. ③ Live pipeline: coordinator → rl_policy_agent → trader_review → execution."}
        </p>
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <div className="rounded-lg border border-slate-700 bg-slate-950/60 p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-cyan-400/90">
              ① Backtest training
            </p>
            <p className="mt-2 text-sm text-slate-300">
              Script{" "}
              <code className="rounded bg-slate-800 px-1 text-[11px] text-emerald-300">
                {rlTrain?.train_script ?? "scripts/train_tabular_q_overnight.py"}
              </code>{" "}
              writes{" "}
              <code className="rounded bg-slate-800 px-1 text-[11px]">outputs/rl_qtable.json</code> and{" "}
              <code className="rounded bg-slate-800 px-1 text-[11px]">outputs/overnight_train_report.json</code>.
            </p>
            <ul className="mt-3 list-inside list-disc text-xs text-slate-500">
              <li>Same state buckets as production: coordinator signal + confidence → Q row.</li>
              <li>Reward = scaled position × daily return (train slice, default first 70 of 100 days).</li>
              <li>Not the PPO block below — that uses config/rl_config.json for the Gym env.</li>
            </ul>
          </div>
          <div className="rounded-lg border border-slate-700 bg-slate-950/60 p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-amber-200/90">
              ② Forward execution
            </p>
            <p className="mt-2 text-sm text-slate-300">
              <code className="rounded bg-slate-800 px-1 text-[11px]">run_pipeline</code> →{" "}
              <code className="rounded bg-slate-800 px-1 text-[11px]">run_rl_policy_agent</code> loads{" "}
              <code className="rounded bg-slate-800 px-1 text-[11px]">outputs/rl_qtable.json</code> →{" "}
              <code className="rounded bg-slate-800 px-1 text-[11px]">trader_review</code> (HITL) → paper
              PnL / approval queue.
            </p>
            {mergedData?.rl_deployed?.state_key ? (
              <p className="mt-3 text-xs text-slate-400">
                This run: state{" "}
                <code className="text-cyan-300">{mergedData.rl_deployed.state_key}</code> → policy{" "}
                <span className="text-emerald-300">{mergedData.rl_deployed.policy_action}</span>
                {mergedData.rl_deployed.final_signal_after_pipeline && (
                  <>
                    {" "}
                    (pipeline signal: {mergedData.rl_deployed.final_signal_after_pipeline})
                  </>
                )}
              </p>
            ) : (
              <p className="mt-3 text-xs text-slate-500">
                Regenerate <code className="text-slate-400">pm_dashboard_state.json</code> after training to
                show deployed state/action for the workspace ticker.
              </p>
            )}
          </div>
        </div>
        <div className="mt-6 rounded-lg border border-slate-700 bg-slate-950/40 p-4">
          <p className="text-xs font-medium text-slate-400">Latest overnight report (from disk)</p>
          {!hasReport && !overnightRepForView && (
            <p className="mt-2 text-sm text-slate-500">
              No <code className="text-slate-400">overnight_train_report.json</code> yet — run the training
              script, then refresh.
            </p>
          )}
          {rlTrain?.report_error && (
            <p className="mt-2 text-sm text-amber-200/90">Read error: {rlTrain.report_error}</p>
          )}
          {overnightRepForView && (
            <dl className="mt-3 grid gap-2 text-sm md:grid-cols-2">
              <div>
                <dt className="text-xs text-slate-500">Train ticker / window</dt>
                <dd className="font-mono text-slate-200">
                  {overnightRepForView.ticker} · {overnightRepForView.train_days ?? "—"}d train /{" "}
                  {overnightRepForView.total_days ?? "—"}d total ·{" "}
                  {overnightRepForView.train_sessions ?? overnightRepForView.sessions ?? "—"} TD sessions ·{" "}
                  {overnightRepForView.episodes_trained?.toLocaleString()} episodes ·{" "}
                  {(overnightRepForView.seconds_wall ?? 0).toFixed(1)}s wall
                </dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">Q table (backtest)</dt>
                <dd className="font-mono text-slate-200">
                  {hasQTable ? "present" : "missing"} · {qStates ?? overnightRepForView.q_states ?? "—"} discrete states
                </dd>
              </div>
              {overnightRepForView.best_greedy_eval && (
                <div className="md:col-span-2">
                  <dt className="text-xs text-slate-500">Best greedy eval (in-sample)</dt>
                  <dd className="font-mono text-emerald-300/90">
                    equity {(overnightRepForView.best_greedy_eval.equity ?? 0).toFixed(6)} · total return{" "}
                    {(overnightRepForView.best_greedy_eval.total_return_pct ?? 0).toFixed(2)}% · Sharpe≈{" "}
                    {(overnightRepForView.best_greedy_eval.sharpe_like ?? 0).toFixed(3)} · saved @ episode{" "}
                    {overnightRepForView.best_greedy_eval.episode?.toLocaleString()}
                  </dd>
                </div>
              )}
              {overnightRepForView.hyperparams && (
                <div className="md:col-span-2">
                  <dt className="text-xs text-slate-500">Hyperparameters</dt>
                  <dd className="font-mono text-[11px] text-slate-400">
                    {JSON.stringify(overnightRepForView.hyperparams)}
                  </dd>
                </div>
              )}
              {overnightRepForView.note && (
                <div className="md:col-span-2">
                  <dt className="text-xs text-slate-500">Note</dt>
                  <dd className="text-xs text-slate-500">{overnightRepForView.note}</dd>
                </div>
              )}
            </dl>
          )}
        </div>
      </section>

      <JsonTable
        title="Universe"
        description="Demo tickers with Yahoo snapshot + CSV risk fields (cash runway months, single-asset exposure)."
        rows={data?.universe}
        maxRows={30}
      />

      <JsonTable
        title="Institutional scorecard"
        description="Pipeline depth / execution, financing quality, signal alignment, composite — same logic as the old Streamlit scorecard."
        rows={data?.institutional_scorecard}
        maxRows={30}
      />

      <JsonTable
        title="Catalyst calendar"
        description="Clinical trial + FDA calendar rows (deduped), sorted by days_to_target."
        rows={sortedCatalyst}
        maxRows={60}
      />

      {data?.batch_runs && data.batch_runs.length > 0 && (
        <section className="mb-10 rounded-xl border border-slate-700 bg-slate-900/40 p-6">
          <h2 className="text-lg font-semibold text-white">Batch signals (Agentic PM)</h2>
          {data.batch_note && (
            <p className="mt-1 text-sm text-slate-500">{data.batch_note}</p>
          )}
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-300">
              <thead>
                <tr className="border-b border-slate-600 text-slate-500">
                  <th className="py-2 pr-4">Ticker</th>
                  <th className="py-2 pr-4">Signal</th>
                  <th className="py-2 pr-4">Conf</th>
                  <th className="py-2">Risk (trunc.)</th>
                </tr>
              </thead>
              <tbody>
                {data.batch_runs.map((row, i) => (
                  <tr key={i} className="border-b border-slate-800">
                    <td className="py-2 pr-4 font-mono text-cyan-300">{row.ticker}</td>
                    <td className="py-2 pr-4">{row.error ?? row.final_signal}</td>
                    <td className="py-2 pr-4">{row.confidence ?? "—"}</td>
                    <td className="max-w-md truncate py-2 text-xs text-slate-500">
                      {row.error ? row.error : row.risk_status}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {mergedData?.ticker && (
        <section id="section-agentic-primary" className="mb-10 grid gap-6 md:grid-cols-2">
          <div className="rounded-xl border border-slate-700 bg-slate-900/50 p-6">
            <h2 className="text-sm font-medium text-slate-300">Agentic PM — timeline &amp; gauges</h2>
            <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
              Matches the workspace ticker when <code className="text-slate-400">pm_dashboard_state.json</code> includes{" "}
              <code className="text-slate-400">by_ticker</code> (from the batch demo).
            </p>
            <p className="mt-2 text-2xl font-semibold text-white">
              {mergedData.ticker}{" "}
              <span className="text-base font-normal text-slate-400">{mergedData.company}</span>
            </p>
            <p className="mt-2 text-sm text-slate-400">
              Proposed: <span className="text-white">{mergedData.final_signal}</span>
            </p>
            {mergedData.rl_deployed?.state_key && (
              <p className="mt-2 text-xs leading-relaxed text-slate-500">
                Tabular RL policy: state{" "}
                <code className="text-cyan-400/90">{mergedData.rl_deployed.state_key}</code> →{" "}
                <span className="text-emerald-400/90">{mergedData.rl_deployed.policy_action}</span>
                <span className="text-slate-600"> · </span>
                <span className="text-slate-600">{mergedData.rl_deployed.forward_path}</span>
              </p>
            )}
          </div>
          <div className="rounded-xl border border-slate-700 bg-slate-900/50 p-6">
            <h2 className="text-sm font-medium text-slate-300">Alpha confidence gauge</h2>
            <div className="mt-4 h-3 w-full overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all"
                style={{ width: `${gaugePct}%` }}
              />
            </div>
            <p className="mt-2 text-center text-2xl font-semibold text-white">{conf}%</p>
          </div>
        </section>
      )}

      <section className="mb-10 rounded-xl border border-slate-700 bg-slate-900/30 p-6">
        <h2 className="text-lg font-semibold text-white">Reasoning timeline</h2>
        <p className="text-sm text-slate-500">
          Planning → Tool use → Memory → Reflection → Decision
          {mergedData?.ticker && (
            <span className="ml-2 font-mono text-[11px] text-slate-600">({mergedData.ticker})</span>
          )}
        </p>
        <ol className="relative mt-6 border-l border-slate-600 pl-6">
          {(mergedData?.reasoning_timeline ?? []).map((step, i) => (
            <li key={i} className="mb-8 ml-2">
              <span className="absolute -left-[9px] mt-1.5 h-3 w-3 rounded-full border border-cyan-400 bg-slate-950" />
              <p className="text-xs font-bold uppercase text-cyan-400/90">
                {step.phase ?? "step"}
              </p>
              <p className="font-medium text-slate-200">{step.title}</p>
              <p className="mt-1 whitespace-pre-wrap text-sm text-slate-400">
                {(step.detail ?? "").slice(0, 1500)}
              </p>
            </li>
          ))}
          {(!mergedData?.reasoning_timeline || mergedData.reasoning_timeline.length === 0) && (
            <li className="text-sm text-slate-500">No timeline yet — run the Python demo script.</li>
          )}
        </ol>
      </section>

      <section className="grid gap-6 md:grid-cols-2">
        <div className="rounded-xl border border-slate-700 p-6">
          <h3 className="text-sm font-medium text-slate-300">Counter-thesis</h3>
          <p className="mt-3 text-sm leading-relaxed text-slate-400">
            {mergedData?.counter_thesis || "—"}
          </p>
        </div>
        <div className="rounded-xl border border-slate-700 p-6">
          <h3 className="text-sm font-medium text-slate-300">Risk &amp; PM</h3>
          <p className="mt-3 font-mono text-xs text-slate-400">{mergedData?.risk_status || "—"}</p>
          <p className="mt-4 text-xs text-slate-500">{mergedData?.pm_weights_preview}</p>
        </div>
      </section>

      <section className="mt-10 rounded-xl border border-slate-700 p-6">
        <h3 className="text-sm font-medium text-slate-300">
          PPO / Gym RL hyperparameters (config/rl_config.json) — separate from tabular Q above
        </h3>
        <pre className="mt-3 overflow-x-auto rounded-lg bg-slate-950 p-4 text-xs text-emerald-300">
          {JSON.stringify(data?.rl_hyperparams ?? {}, null, 2)}
        </pre>
        <pre className="mt-4 overflow-x-auto rounded-lg bg-slate-950 p-4 text-xs text-amber-200/80">
          {JSON.stringify(data?.rl_training ?? {}, null, 2)}
        </pre>
      </section>

      <section className="mt-10 rounded-xl border border-amber-900/50 bg-amber-950/20 p-6">
        <h3 className="text-sm font-medium text-amber-200">HITL — Approval queue</h3>
        <p className="mt-2 text-sm text-slate-400">
          trade_id:{" "}
          <code className="text-white">{mergedData?.hitl?.trade_id ?? "—"}</code> · status:{" "}
          <code className="text-white">{mergedData?.hitl?.execution_status ?? "—"}</code>
        </p>
      </section>
    </main>
  );
}
