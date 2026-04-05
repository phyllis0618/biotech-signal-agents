import { readFileSync } from "fs";
import path from "path";

export const dynamic = "force-dynamic";

export async function GET() {
  const file = path.join(process.cwd(), "..", "outputs", "pnl_simulation_state.json");
  try {
    const raw = readFileSync(file, "utf8");
    return Response.json(JSON.parse(raw));
  } catch {
    return Response.json(
      {
        error: "Run: python3 scripts/simulate_strategy_pnl.py",
        strategy: null,
        live_sim: { running: false, equity: [], step: 0 },
      },
      { status: 200 }
    );
  }
}
