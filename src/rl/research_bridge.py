from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import numpy as np

from src.agentic_research.state import ResearchState
from src.agents.research_manager import run_agentic_research
from src.connectors.market_data import fetch_yahoo_snapshot
from src.rl.trading_env import BiotechTradingEnv


def env_options_from_research(
    state: ResearchState,
    ticker: str,
    sector_vol_ticker: str = "XBI",
) -> Dict[str, Any]:
    """
    Map Phase-1 research output into BiotechTradingEnv.reset(options=...).

    - signal_strength: final_alpha.signal_strength in [-1, 1]
    - sector_volatility: proxy from |XBI day move| (fallback 0.35)
    - cash_runway_months: FinancialAnalyst estimate or 18
    - days_to_fda: placeholder until catalyst NLP extracts dates (default 90d)
    """
    fa = state.get("final_alpha") or {}
    raw_sig = float(fa.get("signal_strength", 0.0))
    signal_strength = float(np.clip(raw_sig, -1.0, 1.0))

    fin = state.get("financials") or {}
    runway = fin.get("runway_months_est")
    cash_runway_months = float(runway) if runway is not None else 18.0

    snap = fetch_yahoo_snapshot(sector_vol_ticker)
    pct = abs(float(snap.get("change_pct", 0.0)))
    sector_volatility = float(min(1.0, pct / 15.0))
    if snap.get("source") == "fallback":
        sector_volatility = 0.35

    t_snap = fetch_yahoo_snapshot(ticker)
    raw_mom = float(t_snap.get("change_pct", 0.0)) / 100.0
    sector_momentum = float(np.clip(raw_mom * 5.0, -1.0, 1.0))

    rat = fa.get("rationale") or {}
    try:
        days_to_fda = float(rat.get("days_to_fda", 90.0))
    except (TypeError, ValueError):
        days_to_fda = 90.0

    return {
        "signal_strength": signal_strength,
        "sector_volatility": sector_volatility,
        "sector_momentum": sector_momentum,
        "cash_runway_months": cash_runway_months,
        "days_to_fda": days_to_fda,
        "_meta": {
            "ticker": ticker,
            "sector_vol_source": sector_vol_ticker,
        },
    }


def run_joint_rollout(
    ticker: str,
    company: str,
    max_steps: int = 50,
    seed: Optional[int] = None,
    policy: Literal["hold", "random", "buy_bias"] = "buy_bias",
) -> Dict[str, Any]:
    """
    End-to-end: Agentic Research -> env options -> simulated trajectory.
    Policy is a stub (not trained PPO) for demo connectivity.
    """
    research_state = run_agentic_research(ticker, company)
    options = env_options_from_research(research_state, ticker)
    meta = options.pop("_meta", {})

    env = BiotechTradingEnv(max_steps=max_steps, seed=seed)
    obs, _ = env.reset(options=options)
    rng = np.random.default_rng(seed)

    rewards: List[float] = []
    actions: List[int] = []
    for _ in range(max_steps):
        if policy == "hold":
            a = 0
        elif policy == "random":
            a = int(rng.integers(0, 4))
        else:
            # Slight bias toward adding exposure if signal > 0
            if obs[0] > 0.55 and rng.random() < 0.35:
                a = 2 if rng.random() < 0.5 else 1
            else:
                a = int(rng.integers(0, 4))

        obs, r, term, trunc, info = env.step(a)
        rewards.append(float(r))
        actions.append(a)
        if term or trunc:
            break

    return {
        "research": research_state.get("final_alpha"),
        "env_options_used": {**options, **meta},
        "total_reward": sum(rewards),
        "steps": len(rewards),
        "last_info": info,
        "actions_sample": actions[:15],
    }
