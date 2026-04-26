"""Minimal test of run_experiments with just a few episodes."""

from pathlib import Path
import numpy as np

from continuous_warehouse import ContinuousWarehouse
from discretized_q import DiscretizedConfig, train_discretized_q
from dqn_agent import DQNConfig, train_dqn
from tile_coded_q import TileQConfig, train_tile_coded_q


OUT_DIR = Path(__file__).resolve().parent / "artifacts"
OUT_DIR.mkdir(exist_ok=True)


def rolling_average(values: list[float], window: int = 100) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size < window:
        return arr
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(arr, kernel, mode="valid")


def main():
    seed = 42
    
    print("Training 4D agents...")
    env_disc = ContinuousWarehouse(use_6d=False, seed=seed)
    env_tile = ContinuousWarehouse(use_6d=False, seed=seed)
    env_dqn = ContinuousWarehouse(use_6d=False, seed=seed)
    
    disc_cfg = DiscretizedConfig(episodes=20)
    tile_cfg = TileQConfig(episodes=20)
    dqn_cfg = DQNConfig(episodes=20)
    
    print("  - Discretized Q-learning...")
    disc_agent, disc_rewards = train_discretized_q(
        env_disc,
        bins_per_dim=[20, 20],
        selected_dims=[0, 1],
        cfg=disc_cfg,
        seed=seed,
    )
    print(f"    ✓ {len(disc_rewards)} episodes, params: {disc_agent.n_parameters}")
    
    print("  - Tile-Coded Q-learning...")
    tile_agent, tile_rewards = train_tile_coded_q(env_tile, cfg=tile_cfg, seed=seed)
    print(f"    ✓ {len(tile_rewards)} episodes, params: {tile_agent.n_parameters}")
    
    print("  - DQN...")
    dqn_agent, dqn_rewards = train_dqn(env_dqn, cfg=dqn_cfg, use_replay=True, use_target=True, seed=seed)
    print(f"    ✓ {len(dqn_rewards)} episodes, params: {dqn_agent.n_parameters}")
    
    print("\nTraining DQN ablations...")
    print("  - Full DQN...")
    env_ab_full = ContinuousWarehouse(use_6d=False, seed=seed)
    _, dqn_full_rewards = train_dqn(env_ab_full, cfg=dqn_cfg, use_replay=True, use_target=True, seed=seed)
    print(f"    ✓ {len(dqn_full_rewards)} episodes")
    
    print("  - No Replay...")
    env_ab_no_replay = ContinuousWarehouse(use_6d=False, seed=seed)
    _, dqn_no_replay_rewards = train_dqn(env_ab_no_replay, cfg=dqn_cfg, use_replay=False, use_target=True, seed=seed)
    print(f"    ✓ {len(dqn_no_replay_rewards)} episodes")
    
    print("  - No Target Network...")
    env_ab_no_target = ContinuousWarehouse(use_6d=False, seed=seed)
    _, dqn_no_target_rewards = train_dqn(env_ab_no_target, cfg=dqn_cfg, use_replay=True, use_target=False, seed=seed)
    print(f"    ✓ {len(dqn_no_target_rewards)} episodes")
    
    print("\nTraining 6D agents (bonus)...")
    print("  - Discretized Q-learning 6D...")
    env_disc_6d = ContinuousWarehouse(use_6d=True, seed=seed)
    disc6_agent, disc6_rewards = train_discretized_q(
        env_disc_6d,
        bins_per_dim=[10, 10, 10, 10, 10, 10],
        selected_dims=[0, 1, 2, 3, 4, 5],
        cfg=DiscretizedConfig(episodes=20, epsilon_decay=0.999, epsilon_min=0.1),
        seed=seed,
    )
    print(f"    ✓ {len(disc6_rewards)} episodes, params: {disc6_agent.n_parameters}")
    
    print("  - DQN 6D...")
    env_dqn_6d = ContinuousWarehouse(use_6d=True, seed=seed)
    dqn6_agent, dqn6_rewards = train_dqn(
        env_dqn_6d,
        cfg=DQNConfig(episodes=20, epsilon_decay=0.998),
        use_replay=True,
        use_target=True,
        seed=seed,
    )
    print(f"    ✓ {len(dqn6_rewards)} episodes, params: {dqn6_agent.n_parameters}")
    
    print("\n✓ All experiments completed successfully!")
    print(f"Artifacts directory: {OUT_DIR}")
    
    # Summary of key metrics
    print("\n" + "="*60)
    print("SUMMARY METRICS")
    print("="*60)
    print(f"\n4D Agents (after {disc_cfg.episodes} episodes):")
    print(f"  Discretized Q: {disc_agent.n_parameters:,} parameters, final reward: {disc_rewards[-1]:.3f}")
    print(f"  Tile-Coded Q:  {tile_agent.n_parameters:,} parameters, final reward: {tile_rewards[-1]:.3f}")
    print(f"  DQN:           {dqn_agent.n_parameters:,} parameters, final reward: {dqn_rewards[-1]:.3f}")
    
    print(f"\nDQN Ablation (after {dqn_cfg.episodes} episodes):")
    print(f"  Full DQN:           final reward: {dqn_full_rewards[-1]:.3f}")
    print(f"  No Replay:          final reward: {dqn_no_replay_rewards[-1]:.3f}")
    print(f"  No Target Network:  final reward: {dqn_no_target_rewards[-1]:.3f}")
    
    visited_bins = int(np.count_nonzero(disc6_agent.visit_counts))
    total_bins = int(np.prod(disc6_agent.visit_counts.shape))
    visited_pct = 100.0 * visited_bins / total_bins
    
    print(f"\n6D Agents (after {20} episodes):")
    print(f"  Discretized Q: {disc6_agent.n_parameters:,} parameters")
    print(f"    Visit coverage: {visited_bins}/{total_bins} bins ({visited_pct:.2f}%)")
    print(f"    Final reward: {disc6_rewards[-1]:.3f}")
    print(f"  DQN:           {dqn6_agent.n_parameters:,} parameters")
    print(f"    Final reward: {dqn6_rewards[-1]:.3f}")


if __name__ == "__main__":
    main()
