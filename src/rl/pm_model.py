from __future__ import annotations

from typing import List, Sequence

import numpy as np

from src.models.messages import AgentMessage, FinalReport

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if _HAS_TORCH:

    class PMPortfolioAggregatorTorch(nn.Module):
        """PyTorch PM layer for training / research."""

        def __init__(self, in_dim: int = 6, n_slots: int = 4, hidden: int = 48) -> None:
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, n_slots),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.softmax(self.mlp(x), dim=-1)

else:
    PMPortfolioAggregatorTorch = None  # type: ignore[misc,valid-type]


def _message_features(messages: Sequence[AgentMessage]) -> List[float]:
    """Map recent agent messages to a fixed 6-d feature vector in [0,1]."""
    feats = [0.0] * 6
    for i, m in enumerate(messages[-6:]):
        h = {"bullish": 1.0, "bearish": 0.0, "neutral": 0.5}.get(m.signal_hint, 0.5)
        feats[i] = float(m.confidence) / 100.0 * 0.5 + h * 0.5
    return feats


def _numpy_pm_softmax(messages: Sequence[AgentMessage]) -> np.ndarray:
    """Deterministic 6→4 softmax (fixed weights) — PM aggregation head."""
    x = np.array(_message_features(messages), dtype=np.float64)
    rng = np.random.default_rng(42)
    W = rng.standard_normal((6, 4))
    b = rng.standard_normal(4) * 0.1
    logits = x @ W + b
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / np.sum(e)


def compute_pm_weights_preview(
    _report: FinalReport,
    messages: Sequence[AgentMessage],
) -> str:
    """Portfolio weights (numpy eval). Use PMPortfolioAggregatorTorch for training."""
    w = _numpy_pm_softmax(messages)
    parts = [f"w{i}={float(w[i]):.3f}" for i in range(len(w))]
    backend = "torch MLP available" if _HAS_TORCH else "numpy only (install torch for training)"
    return f"PM layer ({backend}): " + ", ".join(parts)
