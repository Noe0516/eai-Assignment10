"""Quick test to verify all agents work correctly."""

from continuous_warehouse import ContinuousWarehouse
from discretized_q import DiscretizedConfig, train_discretized_q
from dqn_agent import DQNConfig, train_dqn
from tile_coded_q import TileQConfig, train_tile_coded_q


def test_discretized():
    print("Testing Discretized Q-learning...")
    env = ContinuousWarehouse(use_6d=False, seed=42)
    cfg = DiscretizedConfig(episodes=10)
    agent, rewards = train_discretized_q(
        env,
        bins_per_dim=[20, 20],
        selected_dims=[0, 1],
        cfg=cfg,
        seed=42,
    )
    print(f"  ✓ Trained for {len(rewards)} episodes")
    print(f"  ✓ Final reward: {rewards[-1]:.3f}")
    print(f"  ✓ Parameters: {agent.n_parameters}")


def test_tile_coded():
    print("Testing Tile-Coded Q-learning...")
    env = ContinuousWarehouse(use_6d=False, seed=42)
    cfg = TileQConfig(episodes=10)
    agent, rewards = train_tile_coded_q(env, cfg=cfg, seed=42)
    print(f"  ✓ Trained for {len(rewards)} episodes")
    print(f"  ✓ Final reward: {rewards[-1]:.3f}")
    print(f"  ✓ Parameters: {agent.n_parameters}")


def test_dqn():
    print("Testing DQN...")
    env = ContinuousWarehouse(use_6d=False, seed=42)
    cfg = DQNConfig(episodes=10)
    agent, rewards = train_dqn(env, cfg=cfg, use_replay=True, use_target=True, seed=42)
    print(f"  ✓ Trained for {len(rewards)} episodes")
    print(f"  ✓ Final reward: {rewards[-1]:.3f}")
    print(f"  ✓ Parameters: {agent.n_parameters}")


def test_dqn_6d():
    print("Testing DQN 6D...")
    env = ContinuousWarehouse(use_6d=True, seed=42)
    cfg = DQNConfig(episodes=10)
    agent, rewards = train_dqn(env, cfg=cfg, use_replay=True, use_target=True, seed=42)
    print(f"  ✓ Trained for {len(rewards)} episodes")
    print(f"  ✓ Final reward: {rewards[-1]:.3f}")
    print(f"  ✓ Parameters: {agent.n_parameters}")


def test_discretized_6d():
    print("Testing Discretized Q-learning 6D...")
    env = ContinuousWarehouse(use_6d=True, seed=42)
    cfg = DiscretizedConfig(episodes=10)
    agent, rewards = train_discretized_q(
        env,
        bins_per_dim=[10, 10, 10, 10, 10, 10],
        selected_dims=[0, 1, 2, 3, 4, 5],
        cfg=cfg,
        seed=42,
    )
    print(f"  ✓ Trained for {len(rewards)} episodes")
    print(f"  ✓ Final reward: {rewards[-1]:.3f}")
    print(f"  ✓ Parameters: {agent.n_parameters}")
    print(f"  ✓ Visit counts shape: {agent.visit_counts.shape}")


if __name__ == "__main__":
    print("Running agent tests...\n")
    test_discretized()
    test_tile_coded()
    test_dqn()
    test_dqn_6d()
    test_discretized_6d()
    print("\n✓ All tests passed!")
