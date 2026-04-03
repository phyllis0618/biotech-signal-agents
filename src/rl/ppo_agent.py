from __future__ import annotations

"""
PyTorch PPO-style policy gradient for BiotechTradingEnv sizing.

Reward shaping: environment reward (PnL - λ·DD) + small Sharpe/vol terms from src.rl.reward.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.rl.load_config import clip_reward
from src.rl.reward import sharpe_vol_penalty_reward
from src.rl.trading_env import BiotechTradingEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 5, n_actions: int = 4, hidden: int = 64):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.05
    reward_clip: float = 2.0
    episodes: int = 64
    max_steps: int = 50
    sharpe_weight: float = 0.08
    event_vol_weight: float = 0.12


class PPOAgent:
    """Simplified actor–critic updates (PPO-style clipping can be layered on top)."""

    def __init__(self, cfg: Optional[PPOConfig] = None):
        self.cfg = cfg or PPOConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic().to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=self.cfg.lr)

    def act(self, obs: np.ndarray):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    def train_on_env(self, env: BiotechTradingEnv) -> List[float]:
        cfg = self.cfg
        episode_returns: List[float] = []

        for _ in range(cfg.episodes):
            obs, _ = env.reset(options={})
            logps: List[torch.Tensor] = []
            vals: List[torch.Tensor] = []
            entropies: List[torch.Tensor] = []
            rewards: List[float] = []
            rets_hist: List[float] = []

            for _t in range(cfg.max_steps):
                x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits, value = self.net(x)
                dist = Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)
                entropies.append(dist.entropy())

                next_obs, r, term, trunc, _info = env.step(int(a.item()))
                vol_spike = float(obs[1])
                sh = sharpe_vol_penalty_reward(
                    rets_hist[-20:] if rets_hist else [0.0],
                    event_volatility_spike=vol_spike,
                    vol_penalty=cfg.event_vol_weight,
                )
                r_adj = float(r) + cfg.sharpe_weight * sh
                r_adj = clip_reward(r_adj, cfg.reward_clip)
                rets_hist.append(r_adj)

                logps.append(logp)
                vals.append(value.squeeze(0))
                rewards.append(r_adj)
                obs = next_obs
                if term or trunc:
                    break

            if not rewards or not entropies:
                continue

            R = 0.0
            returns: List[float] = []
            for rw in reversed(rewards):
                R = rw + cfg.gamma * R
                returns.insert(0, R)
            episode_returns.append(float(sum(rewards)))

            returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
            logps_t = torch.stack(logps)
            vals_t = torch.stack(vals).squeeze()
            adv = returns_t - vals_t.detach()

            ent_t = torch.stack(entropies)
            loss = (
                -(logps_t * adv).mean()
                + 0.5 * (returns_t - vals_t).pow(2).mean()
                - cfg.entropy_coef * ent_t.mean()
            )
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.opt.step()

        return episode_returns


def demo_train() -> None:
    from src.rl.load_config import load_rl_hyperparams

    h = load_rl_hyperparams()
    env = BiotechTradingEnv(max_steps=h.max_env_steps)
    agent = PPOAgent(
        PPOConfig(
            lr=h.learning_rate,
            gamma=h.gamma,
            entropy_coef=h.entropy_coef,
            reward_clip=h.reward_clip,
            episodes=min(32, h.ppo_episodes),
            max_steps=h.max_env_steps,
        )
    )
    rets = agent.train_on_env(env)
    print("PPO demo: last episode returns sample:", rets[-5:])


if __name__ == "__main__":
    demo_train()
