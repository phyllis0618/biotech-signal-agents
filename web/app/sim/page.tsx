"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

type SimView = {
  hasData?: boolean;
  message?: string;
  complete?: boolean;
  ticker?: string;
  company?: string;
  split?: {
    total_return_days?: number;
    oos_drill_days?: number;
    train_days?: number;
  };
  total_days?: number;
  day_index?: number;
  day_of_total?: number;
  current_date?: string;
  daily_return?: number;
  momentum_20d?: number;
  coordinator_signal?: string;
  coordinator_confidence?: number;
  rl_state?: string;
  rl_action?: string;
  q_preview?: Record<string, number>;
  equity?: number;
  agent_context?: string;
  event_log?: Array<Record<string, unknown>>;
  settings?: { epsilon: number; tabular_q_lr: number; gamma: number };
  next_steps?: string;
};

export default function SimPage() {
  const [view, setView] = useState<SimView | null>(null);
  const [loading, setLoading] = useState(false);
  const [ticker, setTicker] = useState("AMGN");
  const [company, setCompany] = useState("Amgen Inc.");
  const [withPipeline, setWithPipeline] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const r = await fetch("/api/sim");
    const j = await r.json();
    setView(j);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function start() {
    setLoading(true);
    setErr(null);
    try {
      const r = await fetch("/api/sim", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "start",
          ticker,
          company,
          withPipeline: withPipeline,
          totalDays: 100,
          oosDays: 30,
        }),
      });
      const j = await r.json();
      if (!j.ok) setErr(j.error || "start failed");
      else {
        setErr(null);
        if (j.view) setView(j.view);
        else await refresh();
      }
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function decide(d: "approved" | "rejected" | "deferred") {
    setLoading(true);
    setErr(null);
    try {
      const r = await fetch("/api/sim", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision: d }),
      });
      const j = await r.json();
      if (j.error) setErr(j.error);
      else {
        setErr(null);
        setView(j);
      }
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function saveSettings(s: Partial<{ epsilon: number; tabular_q_lr: number; gamma: number }>) {
    const r = await fetch("/api/sim", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ updateSettings: true, settings: s }),
    });
    setView(await r.json());
  }

  async function rerollRl() {
    const r = await fetch("/api/sim", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resetRlRoll: true }),
    });
    setView(await r.json());
  }

  const s = view?.settings ?? { epsilon: 0.12, tabular_q_lr: 0.35, gamma: 0.85 };

  return (
    <main className="mx-auto max-w-3xl px-6 py-10 text-slate-200">
      <div className="mb-6 flex items-center justify-between">
        <Link href="/" className="text-sm text-cyan-400 hover:underline">
          ← Back to dashboard
        </Link>
        <button
          type="button"
          onClick={() => refresh()}
          className="text-xs text-slate-500 hover:text-slate-300"
        >
          Refresh
        </button>
      </div>

      <h1 className="text-2xl font-semibold text-white">Walk-forward trader drill (30d OOS)</h1>
      <p className="mt-2 text-sm text-slate-400">
        Same Yahoo <span className="text-slate-300">100-day</span> window as the PnL sim:{" "}
        <span className="text-slate-300">70</span> days are the train region,{" "}
        <span className="text-slate-300">30</span> days are forward / OOS — this drill replays only those{" "}
        <span className="text-slate-300">30</span> sessions. Each step: momentum state → RL tabular Q → you
        approve / reject / defer. Q updates; if you approve, paper equity moves with{" "}
        <code className="text-cyan-300">position × daily return</code>.
      </p>

      <section className="mt-8 rounded-xl border border-slate-700 bg-slate-900/40 p-5">
        <h2 className="text-sm font-medium text-slate-300">Start or rebuild session</h2>
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          <label className="text-xs text-slate-500">
            Ticker
            <input
              className="mt-1 w-full rounded border border-slate-600 bg-slate-950 px-2 py-1 font-mono text-white"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
            />
          </label>
          <label className="text-xs text-slate-500">
            Company
            <input
              className="mt-1 w-full rounded border border-slate-600 bg-slate-950 px-2 py-1 text-white"
              value={company}
              onChange={(e) => setCompany(e.target.value)}
            />
          </label>
        </div>
        <label className="mt-3 flex cursor-pointer items-center gap-2 text-xs text-slate-400">
          <input
            type="checkbox"
            checked={withPipeline}
            onChange={(e) => setWithPipeline(e.target.checked)}
          />
          Run full multi-agent pipeline once for context (slow, needs Python env)
        </label>
        <button
          type="button"
          disabled={loading}
          onClick={start}
          className="mt-4 rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-white hover:bg-cyan-500 disabled:opacity-50"
        >
          {loading ? "…" : "Build 30-day OOS session"}
        </button>
        <p className="mt-2 text-[10px] text-slate-500">
          CLI:{" "}
          <code className="text-slate-400">
            python3 scripts/build_interactive_sim.py --total-days 100 --oos-days 30
          </code>
        </p>
      </section>

      {err && <p className="mt-4 text-sm text-red-400">{err}</p>}

      {view?.hasData === false && (
        <p className="mt-6 text-sm text-amber-200/90">{view.message}</p>
      )}

      {view?.hasData && (
        <>
          <section className="mt-8 rounded-xl border border-emerald-900/40 bg-emerald-950/20 p-5">
            <h2 className="text-lg font-semibold text-white">RL inputs (tabular Q)</h2>
            <p className="mt-1 text-xs text-slate-500">
              ε-greedy exploration rate, Q learning rate (bandit update in code), γ reserved for
              future n-step. These mirror the Python tabular layer, not PPO&apos;s{" "}
              <code className="text-slate-400">config/rl_config.json</code> (that file is for the Gym env).
            </p>
            <div className="mt-4 grid gap-4 md:grid-cols-3">
              <label className="text-xs text-slate-500">
                ε (epsilon)
                <input
                  type="range"
                  min={0}
                  max={0.5}
                  step={0.01}
                  value={s.epsilon}
                  onChange={(e) => saveSettings({ epsilon: parseFloat(e.target.value) })}
                  className="mt-1 w-full"
                />
                <span className="font-mono text-cyan-300">{s.epsilon.toFixed(2)}</span>
              </label>
              <label className="text-xs text-slate-500">
                Tabular Q LR
                <input
                  type="range"
                  min={0.05}
                  max={0.9}
                  step={0.05}
                  value={s.tabular_q_lr}
                  onChange={(e) => saveSettings({ tabular_q_lr: parseFloat(e.target.value) })}
                  className="mt-1 w-full"
                />
                <span className="font-mono text-cyan-300">{s.tabular_q_lr.toFixed(2)}</span>
              </label>
              <label className="text-xs text-slate-500">
                γ (γ)
                <input
                  type="range"
                  min={0.5}
                  max={0.99}
                  step={0.01}
                  value={s.gamma}
                  onChange={(e) => saveSettings({ gamma: parseFloat(e.target.value) })}
                  className="mt-1 w-full"
                />
                <span className="font-mono text-cyan-300">{s.gamma.toFixed(2)}</span>
              </label>
            </div>
            <button
              type="button"
              onClick={rerollRl}
              className="mt-3 text-xs text-amber-300 underline hover:text-amber-200"
            >
              Re-roll RL action for this day (after changing ε)
            </button>
          </section>

          <section className="mt-8 rounded-xl border border-slate-700 bg-slate-900/50 p-5">
            <h2 className="text-sm font-medium text-slate-300">Context</h2>
            {view.split && (
              <p className="mt-2 text-[11px] text-slate-500">
                Window: {view.split.total_return_days} return days · train region {view.split.train_days}d ·
                drill (OOS) {view.split.oos_drill_days}d
              </p>
            )}
            <p className="mt-2 text-sm leading-relaxed text-slate-400">{view.agent_context}</p>
            <p className="mt-4 text-xs text-slate-500">{view.next_steps}</p>
          </section>

          {view.complete ? (
            <p className="mt-8 text-lg text-emerald-400">Session complete. Start again to replay.</p>
          ) : (
            <>
              <section className="mt-8 rounded-xl border border-cyan-900/50 bg-slate-900/60 p-5">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <h2 className="text-xl font-semibold text-white">
                    Session {view.day_of_total} / {view.total_days}
                  </h2>
                  <span className="font-mono text-sm text-cyan-300">{view.current_date}</span>
                </div>
                <dl className="mt-4 grid gap-2 text-sm md:grid-cols-2">
                  <div>
                    <dt className="text-slate-500">Daily return</dt>
                    <dd className="font-mono text-white">
                      {(view.daily_return! * 100).toFixed(3)}%
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-500">20d momentum</dt>
                    <dd className="font-mono text-white">
                      {(view.momentum_20d! * 100).toFixed(3)}%
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-500">Coordinator (momentum proxy)</dt>
                    <dd>
                      {view.coordinator_signal} @ {view.coordinator_confidence}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-500">Paper equity</dt>
                    <dd className="font-mono text-emerald-300">{view.equity?.toFixed(6)}</dd>
                  </div>
                </dl>

                <div className="mt-6 border-t border-slate-700 pt-4">
                  <h3 className="text-sm font-medium text-amber-200">RL layer</h3>
                  <p className="mt-1 font-mono text-xs text-slate-400">state = {view.rl_state}</p>
                  <p className="mt-2 text-sm text-white">
                    Suggested action: <strong>{view.rl_action}</strong>
                  </p>
                  <pre className="mt-2 overflow-x-auto rounded bg-slate-950 p-3 text-xs text-emerald-300">
                    {JSON.stringify(view.q_preview, null, 2)}
                  </pre>
                </div>

                <div className="mt-6 border-t border-slate-700 pt-4">
                  <h3 className="text-sm font-medium text-amber-200">Trader review</h3>
                  <p className="mt-1 text-xs text-slate-500">
                    Approved → reward +1 to Q and PnL if you take the RL action; Rejected → −1;
                    Deferred → 0 reward, no PnL. Then advance to next day.
                  </p>
                  <div className="mt-4 flex flex-wrap gap-3">
                    <button
                      type="button"
                      disabled={loading}
                      onClick={() => decide("approved")}
                      className="rounded-lg bg-emerald-700 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-600"
                    >
                      Approve
                    </button>
                    <button
                      type="button"
                      disabled={loading}
                      onClick={() => decide("rejected")}
                      className="rounded-lg bg-red-900/80 px-4 py-2 text-sm font-medium text-white hover:bg-red-800"
                    >
                      Reject
                    </button>
                    <button
                      type="button"
                      disabled={loading}
                      onClick={() => decide("deferred")}
                      className="rounded-lg border border-slate-600 px-4 py-2 text-sm text-slate-200 hover:bg-slate-800"
                    >
                      Defer
                    </button>
                  </div>
                </div>
              </section>
            </>
          )}

          {view.event_log && view.event_log.length > 0 && (
            <section className="mt-8 rounded-xl border border-slate-700 p-5">
              <h2 className="text-sm font-medium text-slate-300">Recent log</h2>
              <ul className="mt-3 max-h-40 overflow-y-auto font-mono text-[11px] text-slate-400">
                {[...view.event_log].reverse().map((e, i) => (
                  <li key={i} className="border-b border-slate-800 py-1">
                    {JSON.stringify(e)}
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </main>
  );
}
