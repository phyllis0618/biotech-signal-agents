from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class RLHyperConfig:
    learning_rate: float = 3e-4
    entropy_coef: float = 0.05
    gamma: float = 0.99
    reward_clip: float = 2.0
    ppo_episodes: int = 64
    max_env_steps: int = 50


def default_rl_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "rl_config.json"


def load_rl_hyperparams(path: Path | None = None) -> RLHyperConfig:
    p = path or default_rl_config_path()
    if not p.exists():
        return RLHyperConfig()
    with open(p, encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)
    return RLHyperConfig(
        learning_rate=float(raw.get("learning_rate", 3e-4)),
        entropy_coef=float(raw.get("entropy_coef", 0.05)),
        gamma=float(raw.get("gamma", 0.99)),
        reward_clip=float(raw.get("reward_clip", 2.0)),
        ppo_episodes=int(raw.get("ppo_episodes", 64)),
        max_env_steps=int(raw.get("max_env_steps", 50)),
    )


def clip_reward(r: float, reward_clip: float = 2.0) -> float:
    return max(-reward_clip, min(reward_clip, r))
