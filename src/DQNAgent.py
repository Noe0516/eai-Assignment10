"""Compatibility wrapper for historical imports.

Use the lower-case module `dqn_agent.py` as the canonical implementation.
This file re-exports the public API so existing imports of `DQNAgent` keep working.
"""

from dqn_agent import DQNAgent, DQNConfig, QNet, ReplayBuffer, train_dqn

__all__ = [
    "DQNAgent",
    "DQNConfig",
    "QNet",
    "ReplayBuffer",
    "train_dqn",
]
