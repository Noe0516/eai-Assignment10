"""DQN agent for the continuous warehouse environment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback for missing dependency
    torch = None  # type: ignore[assignment]

    class _NNStub:
        Module = object

    nn = _NNStub()  # type: ignore[assignment]
    F = Any  # type: ignore[assignment]
    optim = Any  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc

from continuous_warehouse import ContinuousWarehouse


@dataclass
class DQNConfig:
    episodes: int = 2000
    gamma: float = 0.99
    lr: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.997
    epsilon_min: float = 0.05
    replay_size: int = 10_000
    batch_size: int = 64
    target_update_steps: int = 100
    warmup_transitions: int = 500


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, d: bool) -> None:
        self.buffer.append((s, a, r, ns, d))

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[int(i)] for i in idx]
        s, a, r, ns, d = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.asarray(ns, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        )


class QNet(nn.Module):
    def __init__(self, input_dim: int, n_actions: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        cfg: DQNConfig | None = None,
        use_replay: bool = True,
        use_target: bool = True,
        seed: int = 42,
    ) -> None:
        if TORCH_IMPORT_ERROR is not None:
            raise ImportError(
                "PyTorch is required for DQN. Install with: python3 -m pip install torch"
            ) from TORCH_IMPORT_ERROR

        self.cfg = cfg if cfg is not None else DQNConfig()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.use_replay = use_replay
        self.use_target = use_target
        self.rng = np.random.default_rng(seed)

        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNet(state_dim, n_actions).to(self.device)
        self.target_net = QNet(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.replay_size, seed=seed)
        self.train_steps = 0

    @property
    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.q_net.parameters()))

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(s)
            return int(torch.argmax(q, dim=1).item())

    def observe(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, d: bool) -> None:
        self.buffer.push(s, a, r, ns, d)

    def learn_one_step(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, d: bool) -> float | None:
        if self.use_replay:
            if len(self.buffer) < max(self.cfg.warmup_transitions, self.cfg.batch_size):
                return None
            batch = self.buffer.sample(self.cfg.batch_size)
            states, actions, rewards, next_states, dones = batch
        else:
            states = np.asarray([s], dtype=np.float32)
            actions = np.asarray([a], dtype=np.int64)
            rewards = np.asarray([r], dtype=np.float32)
            next_states = np.asarray([ns], dtype=np.float32)
            dones = np.asarray([float(d)], dtype=np.float32)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            target_model = self.target_net if self.use_target else self.q_net
            next_q = target_model(next_states_t).max(dim=1, keepdim=True).values
            targets = rewards_t + (1.0 - dones_t) * self.cfg.gamma * next_q

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.use_target and self.train_steps % self.cfg.target_update_steps == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


def train_dqn(
    env: ContinuousWarehouse,
    cfg: DQNConfig | None = None,
    use_replay: bool = True,
    use_target: bool = True,
    seed: int = 42,
) -> tuple[DQNAgent, list[float]]:
    cfg = cfg if cfg is not None else DQNConfig()
    agent = DQNAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        cfg=cfg,
        use_replay=use_replay,
        use_target=use_target,
        seed=seed,
    )

    # Fill replay memory with random experience before training updates.
    if use_replay:
        state = env.normalize_state(env.reset())
        while len(agent.buffer) < cfg.warmup_transitions:
            action = int(agent.rng.integers(0, env.n_actions))
            next_state_raw, reward, done, _ = env.step(action)
            next_state = env.normalize_state(next_state_raw)
            agent.observe(state, action, reward, next_state, done)
            state = env.normalize_state(env.reset()) if done else next_state

    epsilon = cfg.epsilon_start
    rewards: list[float] = []

    for _ in range(cfg.episodes):
        state = env.normalize_state(env.reset())
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state, epsilon)
            next_state_raw, reward, done, _ = env.step(action)
            next_state = env.normalize_state(next_state_raw)
            agent.observe(state, action, reward, next_state, done)
            agent.learn_one_step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)
        rewards.append(total_reward)

    return agent, rewards
