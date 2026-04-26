
"""Discretized Q-learning for the continuous warehouse environment.

Discretizes selected state dimensions into bins.
Train for 2000 episodes and plot learning curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from continuous_warehouse import ContinuousWarehouse


@dataclass
class DiscretizedConfig:
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.997
    epsilon_min: float = 0.05
    episodes: int = 2000


class DiscretizedQAgent:
    """Q-learning with multi-dimensional discretization."""

    def __init__(
        self,
        bins_per_dim: Sequence[int],
        selected_dims: Sequence[int],
        n_actions: int = 4,
        cfg: DiscretizedConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.cfg = cfg if cfg is not None else DiscretizedConfig()
        self.bins_per_dim = list(bins_per_dim)
        self.selected_dims = list(selected_dims)
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.alpha = self.cfg.alpha
        self.gamma = self.cfg.gamma

        # Build Q-table shape from bins_per_dim
        q_shape = tuple(self.bins_per_dim) + (n_actions,)
        self.Q = np.zeros(q_shape, dtype=np.float32)
        
        # Bounds for each selected dimension (hard-coded for warehouse)
        self.bounds = {
            0: (0.0, 4.0),      # x
            1: (0.0, 4.0),      # y
            2: (0.0, 2 * np.pi), # theta
            3: (0.0, 1.0),       # v
            4: (0.0, 1.0),       # load
            5: (0.0, 1.0),       # battery
        }
        
        # Track visit counts for bonus analysis
        self.visit_counts = np.zeros(self.Q.shape[:-1], dtype=np.int32)

    @property
    def n_parameters(self) -> int:
        """Total number of Q-values (parameters)."""
        return int(self.Q.size)

    def _discretize(self, state: np.ndarray) -> tuple[int, ...]:
        """Discretize state dimensions into bin indices."""
        indices = []
        for i, dim in enumerate(self.selected_dims):
            val = float(state[dim])
            low, high = self.bounds[dim]
            normalized = (val - low) / (high - low)
            bin_idx = int(np.clip(normalized * self.bins_per_dim[i], 0, self.bins_per_dim[i] - 1))
            indices.append(bin_idx)
        return tuple(indices)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for all actions at this state."""
        idx = self._discretize(state)
        return self.Q[idx]

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_values(state)))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Q-learning update (Bellman backup)."""
        idx = self._discretize(state)
        q_sa = self.Q[idx][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_values(next_state))
        
        td_error = target - q_sa
        self.Q[idx][action] += self.alpha * td_error
        self.visit_counts[idx] += 1


def train_discretized_q(
    env: ContinuousWarehouse,
    bins_per_dim: Sequence[int],
    selected_dims: Sequence[int],
    cfg: DiscretizedConfig | None = None,
    seed: int = 42,
) -> tuple[DiscretizedQAgent, list[float]]:
    """Train discretized Q-learning agent."""
    cfg = cfg if cfg is not None else DiscretizedConfig()
    agent = DiscretizedQAgent(
        bins_per_dim=bins_per_dim,
        selected_dims=selected_dims,
        n_actions=env.n_actions,
        cfg=cfg,
        seed=seed,
    )
    
    epsilon = cfg.epsilon_start
    rewards: list[float] = []
    
    for _ in range(cfg.episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)
        rewards.append(total_reward)
    
    return agent, rewards
