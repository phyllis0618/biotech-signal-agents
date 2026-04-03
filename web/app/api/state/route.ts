import { readFileSync } from "fs";
import path from "path";

export const dynamic = "force-dynamic";

export async function GET() {
  const file = path.join(process.cwd(), "..", "outputs", "pm_dashboard_state.json");
  try {
    const raw = readFileSync(file, "utf8");
    return Response.json(JSON.parse(raw));
  } catch {
    return Response.json(
      {
        error: "Run Python pipeline with Agentic PM enabled or `write_pm_dashboard_state` to generate outputs/pm_dashboard_state.json",
        system_tag: "AUTONOMOUS_COGNITIVE_SYSTEM",
        reasoning_timeline: [],
        rl_training: { status: "no_data" },
      },
      { status: 200 }
    );
  }
}
