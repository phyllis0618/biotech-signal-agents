import { existsSync, readFileSync } from "fs";
import path from "path";

export const dynamic = "force-dynamic";

/**
 * Latest tabular RL backtest report + Q table presence (repo outputs/).
 * Keeps training panel fresh without re-running run_demo_for_frontend.py.
 */
export async function GET() {
  const outDir = path.join(process.cwd(), "..", "outputs");
  const reportPath = path.join(outDir, "overnight_train_report.json");
  const qPath = path.join(outDir, "rl_qtable.json");

  let overnight_report: Record<string, unknown> | null = null;
  let reportError: string | null = null;
  let q_state_count: number | null = null;

  try {
    if (existsSync(reportPath)) {
      overnight_report = JSON.parse(readFileSync(reportPath, "utf8")) as Record<string, unknown>;
    }
  } catch (e) {
    reportError = String(e);
  }

  try {
    if (existsSync(qPath)) {
      const q = JSON.parse(readFileSync(qPath, "utf8")) as unknown;
      q_state_count = typeof q === "object" && q !== null ? Object.keys(q as object).length : 0;
    }
  } catch {
    q_state_count = null;
  }

  return Response.json({
    flow_summary:
      "① 100d Yahoo window: TD Q trains on first 70 return days → outputs/rl_qtable.json. " +
      "② PnL sim: real OOS on last 30 days. " +
      "③ Live pipeline: coordinator → rl_policy_agent → trader_review → execution.",
    train_script: "scripts/train_tabular_q_overnight.py",
    report_available: existsSync(reportPath),
    q_table_available: existsSync(qPath),
    q_state_count,
    overnight_report,
    report_error: reportError,
  });
}
