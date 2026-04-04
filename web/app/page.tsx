"use client";

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
  hitl?: { trade_id?: string; execution_status?: string; approval_required?: boolean };
  batch_runs?: BatchRow[];
  batch_note?: string;
  universe?: Record<string, unknown>[];
  catalyst_calendar?: Record<string, unknown>[];
  institutional_scorecard?: Record<string, unknown>[];
  error?: string;
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

export default function Home() {
  const [data, setData] = useState<DashboardState | null>(null);

  useEffect(() => {
    fetch("/api/state")
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData({ error: "fetch failed" }));
  }, []);

  const conf = data?.confidence ?? 0;
  const gaugePct = Math.min(100, Math.max(0, conf));

  const sortedCatalyst = useMemo(() => {
    const rows = data?.catalyst_calendar ?? [];
    return [...rows].sort((a, b) => {
      const da = typeof a.days_to_target === "number" ? a.days_to_target : 9999;
      const db = typeof b.days_to_target === "number" ? b.days_to_target : 9999;
      return da - db;
    });
  }, [data?.catalyst_calendar]);

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
          </div>
          <div className="rounded-lg border border-slate-600 bg-slate-900/80 px-4 py-3 text-right">
            <p className="text-xs text-slate-500">RL training</p>
            <p className="font-mono text-sm text-accent-amber">
              {data?.rl_training?.status ?? "—"}
            </p>
            <p className="mt-1 text-[10px] text-slate-500">
              {data?.rl_training?.note?.slice(0, 80)}
            </p>
          </div>
        </div>
      </header>

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

      {data?.ticker && (
        <section className="mb-10 grid gap-6 md:grid-cols-2">
          <div className="rounded-xl border border-slate-700 bg-slate-900/50 p-6">
            <h2 className="text-sm font-medium text-slate-300">Primary ticker (timeline source)</h2>
            <p className="mt-2 text-2xl font-semibold text-white">
              {data.ticker}{" "}
              <span className="text-base font-normal text-slate-400">{data.company}</span>
            </p>
            <p className="mt-2 text-sm text-slate-400">
              Proposed: <span className="text-white">{data.final_signal}</span>
            </p>
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
        </p>
        <ol className="relative mt-6 border-l border-slate-600 pl-6">
          {(data?.reasoning_timeline ?? []).map((step, i) => (
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
          {(!data?.reasoning_timeline || data.reasoning_timeline.length === 0) && (
            <li className="text-sm text-slate-500">No timeline yet — run the Python demo script.</li>
          )}
        </ol>
      </section>

      <section className="grid gap-6 md:grid-cols-2">
        <div className="rounded-xl border border-slate-700 p-6">
          <h3 className="text-sm font-medium text-slate-300">Counter-thesis</h3>
          <p className="mt-3 text-sm leading-relaxed text-slate-400">
            {data?.counter_thesis || "—"}
          </p>
        </div>
        <div className="rounded-xl border border-slate-700 p-6">
          <h3 className="text-sm font-medium text-slate-300">Risk &amp; PM</h3>
          <p className="mt-3 font-mono text-xs text-slate-400">{data?.risk_status || "—"}</p>
          <p className="mt-4 text-xs text-slate-500">{data?.pm_weights_preview}</p>
        </div>
      </section>

      <section className="mt-10 rounded-xl border border-slate-700 p-6">
        <h3 className="text-sm font-medium text-slate-300">RL hyperparameters (config/rl_config.json)</h3>
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
          <code className="text-white">{data?.hitl?.trade_id ?? "—"}</code> · status:{" "}
          <code className="text-white">{data?.hitl?.execution_status ?? "—"}</code>
        </p>
      </section>
    </main>
  );
}
