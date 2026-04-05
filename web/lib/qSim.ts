/**
 * Tabular Q helpers aligned with src/trading/q_learning.py (bandit-style update).
 */

export const ACTIONS = ["long", "short", "no_trade"] as const;
export type Action = (typeof ACTIONS)[number];

export function stateKey(signal: string, confidence: number): string {
  const bucket = Math.max(0, Math.min(9, Math.floor(confidence / 10)));
  return `${signal}|${bucket}`;
}

function pickRandomAction(): Action {
  return ACTIONS[Math.floor(Math.random() * ACTIONS.length)];
}

export function selectAction(
  q: Record<string, Record<string, number>>,
  state: string,
  epsilon: number
): Action {
  if (Math.random() < epsilon) return pickRandomAction();
  const row = q[state] ?? {};
  let best: Action = "no_trade";
  let bestV = row[best] ?? 0;
  for (const a of ACTIONS) {
    const v = row[a] ?? 0;
    if (v > bestV) {
      bestV = v;
      best = a;
    }
  }
  return best;
}

export function updateQ(
  q: Record<string, Record<string, number>>,
  state: string,
  action: Action,
  reward: number,
  lr: number
): void {
  if (!q[state]) {
    q[state] = { long: 0, short: 0, no_trade: 0 };
  }
  const old = q[state][action] ?? 0;
  q[state][action] = old + lr * (reward - old);
}

export function traderReward(decision: "approved" | "rejected" | "deferred"): number {
  if (decision === "approved") return 1;
  if (decision === "rejected") return -1;
  return 0;
}

export function positionWeight(action: Action): number {
  if (action === "long") return 1;
  if (action === "short") return -1;
  return 0;
}

export function qPreview(
  q: Record<string, Record<string, number>>,
  state: string
): Record<string, number> {
  const row = q[state] ?? {};
  return {
    long: row.long ?? 0,
    short: row.short ?? 0,
    no_trade: row.no_trade ?? 0,
  };
}
