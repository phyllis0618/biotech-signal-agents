import { execFileSync } from "child_process";
import { existsSync, readFileSync, writeFileSync } from "fs";
import path from "path";

import {
  positionWeight,
  qPreview,
  selectAction,
  stateKey,
  traderReward,
  updateQ,
  type Action,
} from "@/lib/qSim";

export const runtime = "nodejs";

const FILE = path.join(process.cwd(), "..", "outputs", "interactive_sim.json");

type DayRow = {
  date: string;
  daily_return: number;
  coordinator_signal: string;
  coordinator_confidence: number;
  momentum_20d: number;
  locked_rl_action?: string;
};

type SimFile = {
  version: number;
  ticker: string;
  company: string;
  split?: {
    total_return_days: number;
    oos_drill_days: number;
    train_days: number;
  };
  agent_context: string;
  days: DayRow[];
  sim_q: Record<string, Record<string, number>>;
  current_index: number;
  equity: number;
  event_log: Array<{
    date: string;
    decision: string;
    rl_state: string;
    rl_action: string;
    reward: number;
    pnl_step: number;
    equity_after: number;
  }>;
  settings: { epsilon: number; tabular_q_lr: number; gamma: number };
  status: string;
};

function load(): SimFile | null {
  if (!existsSync(FILE)) return null;
  try {
    return JSON.parse(readFileSync(FILE, "utf8")) as SimFile;
  } catch {
    return null;
  }
}

function save(data: SimFile) {
  writeFileSync(FILE, JSON.stringify(data, null, 2), "utf8");
}

function ensureLockedRl(data: SimFile) {
  const idx = data.current_index;
  if (idx >= data.days.length) return;
  const day = data.days[idx];
  const st = stateKey(day.coordinator_signal, day.coordinator_confidence);
  if (!day.locked_rl_action) {
    day.locked_rl_action = selectAction(data.sim_q, st, data.settings.epsilon);
    save(data);
  }
}

function buildView(data: SimFile) {
  ensureLockedRl(data);
  const total = data.days.length;
  const idx = data.current_index;
  const complete = idx >= total;
  if (complete) {
    return {
      hasData: true,
      complete: true,
      ticker: data.ticker,
      company: data.company,
      split: data.split,
      total_days: total,
      day_index: total,
      equity: data.equity,
      agent_context: data.agent_context,
      event_log: data.event_log,
      settings: data.settings,
      next_steps:
        "Session finished. Run `python3 scripts/build_interactive_sim.py` or click Start again.",
    };
  }

  const day = data.days[idx];
  const rlState = stateKey(day.coordinator_signal, day.coordinator_confidence);
  const rlAction = day.locked_rl_action as Action;
  const q_pv = qPreview(data.sim_q, rlState);

  return {
    hasData: true,
    complete: false,
    ticker: data.ticker,
    company: data.company,
    split: data.split,
    total_days: total,
    day_index: idx,
    day_of_total: idx + 1,
    current_date: day.date,
    daily_return: day.daily_return,
    momentum_20d: day.momentum_20d,
    coordinator_signal: day.coordinator_signal,
    coordinator_confidence: day.coordinator_confidence,
    rl_state: rlState,
    rl_action: rlAction,
    q_preview: q_pv,
    equity: data.equity,
    agent_context: data.agent_context,
    event_log: data.event_log.slice(-20),
    settings: data.settings,
    next_steps:
      "1) RL picks an action from Q + ε-greedy. 2) You approve/reject/defer — Q updates like apply_trader_feedback. 3) If approved, paper PnL moves with RL action × today’s return. 4) Next session advances one OOS day (last segment of the 100d window).",
  };
}

export async function GET() {
  const data = load();
  if (!data) {
    return Response.json({
      hasData: false,
      message:
        "No session file. Click “Start” or run: python3 scripts/build_interactive_sim.py",
    });
  }
  return Response.json(buildView(data));
}

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}));

  if (body.action === "start") {
    const root = path.join(process.cwd(), "..");
    const script = path.join(root, "scripts", "build_interactive_sim.py");
    const ticker = (body.ticker as string) || "AMGN";
    const company = (body.company as string) || "Amgen Inc.";
    const args = [script, "--ticker", ticker, "--company", company];
    if (typeof body.totalDays === "number") {
      args.push("--total-days", String(body.totalDays));
    } else {
      args.push("--total-days", "100");
    }
    if (typeof body.oosDays === "number") {
      args.push("--oos-days", String(body.oosDays));
    } else {
      args.push("--oos-days", "30");
    }
    if (body.withPipeline) args.push("--with-pipeline");
    try {
      execFileSync("python3", args, { cwd: root, stdio: "pipe" });
    } catch (e) {
      return Response.json(
        { ok: false, error: String(e) },
        { status: 500 }
      );
    }
    const data = load();
    return Response.json({ ok: true, view: data ? buildView(data) : null });
  }

  const data = load();
  if (!data) {
    return Response.json({ error: "no session" }, { status: 400 });
  }

  if (body.updateSettings && body.settings) {
    data.settings = { ...data.settings, ...body.settings };
    save(data);
    return Response.json(buildView(data));
  }

  if (body.decision) {
    const d = body.decision as "approved" | "rejected" | "deferred";
    if (data.current_index >= data.days.length) {
      return Response.json({ error: "already complete" }, { status: 400 });
    }

    const day = data.days[data.current_index];
    const rlState = stateKey(day.coordinator_signal, day.coordinator_confidence);
    ensureLockedRl(data);
    const rlAction = (data.days[data.current_index].locked_rl_action ?? "no_trade") as Action;
    const reward = traderReward(d);
    updateQ(data.sim_q, rlState, rlAction, reward, data.settings.tabular_q_lr);

    let pnlStep = 0;
    if (d === "approved") {
      const w = positionWeight(rlAction);
      pnlStep = w * day.daily_return;
      data.equity *= 1 + pnlStep;
    }

    data.event_log.push({
      date: day.date,
      decision: d,
      rl_state: rlState,
      rl_action: rlAction,
      reward,
      pnl_step: pnlStep,
      equity_after: data.equity,
    });
    data.current_index += 1;
    if (data.current_index >= data.days.length) data.status = "complete";
    save(data);
    return Response.json(buildView(data));
  }

  if (body.resetRlRoll) {
    const idx = data.current_index;
    if (idx < data.days.length) {
      delete data.days[idx].locked_rl_action;
      save(data);
    }
    return Response.json(buildView(data));
  }

  return Response.json({ error: "invalid body" }, { status: 400 });
}
