"""Tile-coded linear semi-gradient Q-learning for the continuous warehouse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuous_warehouse import ContinuousWarehouse


@dataclass
class TileQConfig:
    n_tilings: int = 8
    tiles_per_dim: int = 4
    alpha: float = 0.2
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.997
    epsilon_min: float = 0.05
    episodes: int = 2000


class TileCoder:
    """2D tile coder with offset tilings and one extra tile per dimension."""

    def __init__(self, n_tilings: int, tiles_per_dim: int, low: float = 0.0, high: float = 4.0) -> None:
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim
        self.ext = tiles_per_dim + 1
        self.low = low
        self.high = high
        self.span = high - low
        self.tile_width = self.span / tiles_per_dim
        self.delta = self.tile_width / n_tilings
        self.tiles_per_tiling = self.ext * self.ext
        self.n_features = n_tilings * self.tiles_per_tiling

    def encode(self, x: float, y: float) -> np.ndarray:
        phi = np.zeros(self.n_features, dtype=np.float32)

        for k in range(self.n_tilings):
            ox = k * self.delta
            oy = k * self.delta
            bx = int(np.floor((x - self.low + ox) / self.tile_width))
            by = int(np.floor((y - self.low + oy) / self.tile_width))
            bx = int(np.clip(bx, 0, self.ext - 1))
            by = int(np.clip(by, 0, self.ext - 1))
            idx = k * self.tiles_per_tiling + bx * self.ext + by
            phi[idx] = 1.0

        return phi


class TileCodedQAgent:
    """Linear Q approximation over tile-coded features of (x, y)."""

    def __init__(self, n_actions: int = 4, cfg: TileQConfig | None = None, seed: int = 42) -> None:
        self.cfg = cfg if cfg is not None else TileQConfig()
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

        self.tile_coder = TileCoder(self.cfg.n_tilings, self.cfg.tiles_per_dim)
        self.n_features = self.tile_coder.n_features

        # Semi-gradient Q-learning (Eq. 7.18 style): one linear head per action.
        self.W = np.zeros((n_actions, self.n_features), dtype=np.float32)
        self.alpha = self.cfg.alpha / self.cfg.n_tilings

    @property
    def n_parameters(self) -> int:
        return int(self.W.size)

    def features(self, state: np.ndarray) -> np.ndarray:
        return self.tile_coder.encode(float(state[0]), float(state[1]))

    def q_values(self, state: np.ndarray) -> np.ndarray:
        phi = self.features(state)
        return self.W @ phi

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_values(state)))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        phi = self.features(state)
        q_sa = float(self.W[action] @ phi)
        if done:
            target = reward
        else:
            target = reward + self.cfg.gamma * float(np.max(self.q_values(next_state)))
        td_error = target - q_sa
        self.W[action] += self.alpha * td_error * phi


def train_tile_coded_q(
    env: ContinuousWarehouse,
    cfg: TileQConfig | None = None,
    seed: int = 42,
) -> tuple[TileCodedQAgent, list[float]]:
    cfg = cfg if cfg is not None else TileQConfig()
    agent = TileCodedQAgent(n_actions=env.n_actions, cfg=cfg, seed=seed)

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
