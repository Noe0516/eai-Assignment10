# Exercise 2 Report: Continuous Warehouse RL

## 1. Setup and Methods

Implemented files:
- continuous_warehouse.py
- discretized_q.py
- tile_coded_q.py
- dqn_agent.py
- run_experiments.py (runner for plots + metrics + PDF)

Environment:
- State (4D): (x, y, theta, v)
- Actions: North, South, East, West
- Goal region: circle center (3.5, 3.5), radius 0.5
- Hazard region: circle center (2.0, 2.0), radius 0.5
- Gaussian transition noise in position, heading, and speed updates

DQN normalization to [0,1] before network input:
- x/4, y/4, theta/(2*pi), v/v_max

Approaches:
- Discretized Q-learning: 20x20 bins over (x,y), ignores theta and v
- Tile-coded linear Q-learning: 8 offset tilings of 4x4, each with one extra tile per dimension => 8*(4+1)^2 = 200 binary features
- DQN (PyTorch): MLP 4(or 6)->64->64->4, Adam optimizer, replay buffer 10,000, batch 64, target update every 100 steps, replay warmup >= 500 transitions

## 2. Expected Findings

- In 4D, all three methods should achieve comparable policy quality.
- Tile-coded Q-learning often converges fastest due to smooth generalization and low overhead.
- DQN may learn slightly slower initially but uses full state.
- DQN ablation: removing replay is usually the most destabilizing change (deadly triad context), while removing target network tends to increase variance.

## 3. Bonus 6D

Extended state: (x, y, theta, v, load, battery)
- Load affects speed and goal reward
- Battery depletes and reduces speed over time

Discretization at 10 bins/dimension:
- Q-table size = 10^6 * 4 = 4,000,000 entries
- Demonstrates curse of dimensionality versus DQN function approximation.

## 4. Run Instructions

From workspace root folder:

```bash
cd /Users/noeortega/eai-Assignment10/src
python3 -m pip install torch
python3 run_experiments.py
```

Generated outputs:
- artifacts/learning_curves_comparison.png
- artifacts/dqn_ablation_curves.png
- artifacts/bonus_6d_comparison.png
- artifacts/policy_discretized.png
- artifacts/policy_tile.png
- artifacts/policy_dqn.png
- artifacts/metrics_summary.txt
- exercise_2_report.pdf

## 5. Status in This Session

Code implementation is complete and syntax-checked. Full experiment execution and final numeric tables were not run here because PyTorch installation/execution was skipped in this environment.
