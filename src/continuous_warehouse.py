"""Continuous warehouse environment used in function approximation experiments.

State (4D mode): (x, y, theta, v)
State (6D mode): (x, y, theta, v, load, battery)
Actions: 0=N, 1=S, 2=E, 3=W
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class EnvConfig:
    floor_size: float = 4.0
    goal_center: Tuple[float, float] = (3.5, 3.5)
    hazard_center: Tuple[float, float] = (2.0, 2.0)
    goal_radius: float = 0.5
    hazard_radius: float = 0.5
    transition_noise_xy: float = 0.03
    transition_noise_theta: float = 0.05
    transition_noise_v: float = 0.02
    dt: float = 0.25
    max_steps: int = 200


class ContinuousWarehouse:
    """Continuous warehouse floor with noisy transitions and circular regions."""

    ACTION_TO_HEADING = {
        0: math.pi / 2.0,   # North
        1: 3.0 * math.pi / 2.0,  # South
        2: 0.0,             # East
        3: math.pi,         # West
    }

    def __init__(
        self,
        use_6d: bool = False,
        seed: int | None = None,
        config: EnvConfig | None = None,
    ) -> None:
        self.cfg = config if config is not None else EnvConfig()
        self.use_6d = use_6d
        self.rng = np.random.default_rng(seed)
        self.v_max = 1.0
        self.state_dim = 6 if use_6d else 4
        self.n_actions = 4
        self._state = np.zeros(self.state_dim, dtype=np.float32)
        self._step_count = 0

    def reset(self) -> np.ndarray:
        self._step_count = 0
        x = 0.5 + self.rng.normal(0.0, 0.05)
        y = 0.5 + self.rng.normal(0.0, 0.05)
        theta = float(self.rng.uniform(0.0, 2.0 * math.pi))
        v = float(self.rng.uniform(0.0, 0.1))

        if self.use_6d:
            load = float(self.rng.uniform(0.1, 1.0))
            battery = 1.0
            self._state = np.array([x, y, theta, v, load, battery], dtype=np.float32)
        else:
            self._state = np.array([x, y, theta, v], dtype=np.float32)

        self._clamp_state()
        return self._state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self._step_count += 1
        action = int(np.clip(action, 0, self.n_actions - 1))

        x, y, theta, v = [float(z) for z in self._state[:4]]
        desired_heading = self.ACTION_TO_HEADING[action]

        # Turn towards selected cardinal action with noise.
        turn_gain = 0.25
        theta = self._wrap_angle(
            theta + turn_gain * self._angle_diff(desired_heading, theta)
            + self.rng.normal(0.0, self.cfg.transition_noise_theta)
        )

        speed_limit = self.v_max
        if self.use_6d:
            load = float(self._state[4])
            battery = float(self._state[5])
            speed_limit = self.v_max * (1.0 - 0.35 * load) * (0.4 + 0.6 * battery)

        v = v + 0.2 * (speed_limit - v) + self.rng.normal(0.0, self.cfg.transition_noise_v)
        v = float(np.clip(v, 0.0, max(0.05, speed_limit)))

        dx = v * math.cos(theta) * self.cfg.dt + self.rng.normal(0.0, self.cfg.transition_noise_xy)
        dy = v * math.sin(theta) * self.cfg.dt + self.rng.normal(0.0, self.cfg.transition_noise_xy)
        x += dx
        y += dy

        if self.use_6d:
            battery = float(np.clip(self._state[5] - (0.004 + 0.003 * v), 0.0, 1.0))
            self._state = np.array([x, y, theta, v, self._state[4], battery], dtype=np.float32)
        else:
            self._state = np.array([x, y, theta, v], dtype=np.float32)

        self._clamp_state()

        reward = -0.01
        done = False
        info = {}

        if self._in_circle(self._state[0], self._state[1], self.cfg.hazard_center, self.cfg.hazard_radius):
            reward = -1.0
            done = True
            info["terminal_reason"] = "hazard"

        if self._in_circle(self._state[0], self._state[1], self.cfg.goal_center, self.cfg.goal_radius):
            goal_reward = 1.0
            if self.use_6d:
                # Heavier payloads are worth more at delivery.
                goal_reward += 0.5 * float(self._state[4])
            reward = goal_reward
            done = True
            info["terminal_reason"] = "goal"

        if self._step_count >= self.cfg.max_steps:
            done = True
            info["terminal_reason"] = "timeout"

        return self._state.copy(), float(reward), done, info

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state features to [0, 1] for neural-network training."""
        norm = np.asarray(state, dtype=np.float32).copy()
        norm[0] = np.clip(norm[0] / self.cfg.floor_size, 0.0, 1.0)
        norm[1] = np.clip(norm[1] / self.cfg.floor_size, 0.0, 1.0)
        norm[2] = np.clip(norm[2] / (2.0 * math.pi), 0.0, 1.0)
        norm[3] = np.clip(norm[3] / max(1e-6, self.v_max), 0.0, 1.0)
        if self.use_6d:
            norm[4] = np.clip(norm[4], 0.0, 1.0)
            norm[5] = np.clip(norm[5], 0.0, 1.0)
        return norm

    def _clamp_state(self) -> None:
        self._state[0] = np.clip(self._state[0], 0.0, self.cfg.floor_size)
        self._state[1] = np.clip(self._state[1], 0.0, self.cfg.floor_size)
        self._state[2] = self._wrap_angle(float(self._state[2]))
        self._state[3] = np.clip(self._state[3], 0.0, self.v_max)
        if self.use_6d:
            self._state[4] = np.clip(self._state[4], 0.0, 1.0)
            self._state[5] = np.clip(self._state[5], 0.0, 1.0)

    @staticmethod
    def _in_circle(x: float, y: float, center: tuple[float, float], radius: float) -> bool:
        return ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius ** 2

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        return float(theta % (2.0 * math.pi))

    @staticmethod
    def _angle_diff(target: float, current: float) -> float:
        diff = (target - current + math.pi) % (2.0 * math.pi) - math.pi
        return float(diff)