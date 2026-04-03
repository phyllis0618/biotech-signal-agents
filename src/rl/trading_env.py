from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.rl.reward import pnl_drawdown_reward


class BiotechTradingEnv(gym.Env):
    """
    Gymnasium-style environment for biotech position sizing.

    Observation (normalized [0,1]):
      [Alpha_strength, Volatility, Sector_momentum, Cash_Runway_Months, Days_to_FDA_Decision]

    Actions:
      0 Hold, 1 Buy 5%, 2 Buy 10%, 3 Sell All
    """

    metadata = {"render_modes": []}
    OBS_DIM = 5

    def __init__(
        self,
        lambda_dd: float = 0.5,
        max_steps: int = 50,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.lambda_dd = lambda_dd
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.OBS_DIM, dtype=np.float32),
            high=np.ones(self.OBS_DIM, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(4)

        self._rng = np.random.default_rng(seed)
        self._position = 0.0
        self._cash = 1.0
        self._price = 100.0
        self._equity_peak = 1.0
        self._max_drawdown = 0.0
        self._step_idx = 0
        self._obs = np.zeros(self.OBS_DIM, dtype=np.float32)

    def _normalize_obs(
        self,
        signal_strength: float,
        sector_vol: float,
        sector_momentum: float,
        runway_months: float,
        days_fda: float,
    ) -> np.ndarray:
        # Map to [0,1]: signal [-1,1] -> [0,1], vol unbounded -> tanh, momentum [-1,1]->[0,1]
        s = float(np.clip((signal_strength + 1.0) / 2.0, 0.0, 1.0))
        v = float(np.tanh(sector_vol / 0.5) * 0.5 + 0.5)
        mom = float(np.clip((sector_momentum + 1.0) / 2.0, 0.0, 1.0))
        r = float(np.clip(runway_months / 36.0, 0.0, 1.0))
        d = float(np.clip(days_fda / 365.0, 0.0, 1.0))
        return np.array([s, v, mom, r, d], dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._position = 0.0
        self._cash = 1.0
        self._price = 100.0
        self._equity_peak = 1.0
        self._max_drawdown = 0.0
        self._step_idx = 0

        opt = options or {}
        signal = float(opt.get("signal_strength", self._rng.uniform(-0.5, 0.5)))
        vol = float(opt.get("sector_volatility", self._rng.uniform(0.1, 0.8)))
        mom = float(opt.get("sector_momentum", self._rng.uniform(-0.4, 0.4)))
        runway = float(opt.get("cash_runway_months", self._rng.uniform(6, 30)))
        fda_days = float(opt.get("days_to_fda", self._rng.uniform(30, 200)))

        self._obs = self._normalize_obs(signal, vol, mom, runway, fda_days)
        return self._obs.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if action == 0:
            target_w = self._position
        elif action == 1:
            target_w = min(1.0, self._position + 0.05)
        elif action == 2:
            target_w = min(1.0, self._position + 0.10)
        else:
            target_w = 0.0

        # Synthetic biotech jump risk scales with (1 - runway_obs) and vol_obs
        base_ret = self._rng.normal(0.0, 0.02)
        event_shock = self._rng.normal(0.0, 0.08) * float(self._obs[1])  # vol channel
        mom_push = 0.01 * (float(self._obs[2]) - 0.5)
        price_ret = base_ret + event_shock + mom_push
        self._price *= 1.0 + price_ret

        prev_equity = self._cash + self._position * self._price
        self._position = target_w
        self._cash = prev_equity - self._position * self._price
        equity = self._cash + self._position * self._price

        self._equity_peak = max(self._equity_peak, equity)
        dd = 1.0 - equity / max(self._equity_peak, 1e-8)
        self._max_drawdown = max(self._max_drawdown, dd)

        pnl_step = equity - prev_equity
        reward = pnl_drawdown_reward(pnl_step, self._max_drawdown, self.lambda_dd)

        self._step_idx += 1
        terminated = self._step_idx >= self.max_steps
        truncated = False

        # evolve latent factors slightly (walk)
        self._obs[0] = float(np.clip(self._obs[0] + self._rng.normal(0, 0.02), 0, 1))
        self._obs[1] = float(np.clip(self._obs[1] + self._rng.normal(0, 0.03), 0, 1))

        info = {"equity": float(equity), "max_drawdown": float(self._max_drawdown)}
        return self._obs.copy(), float(reward), terminated, truncated, info
