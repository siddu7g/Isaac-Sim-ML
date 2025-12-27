# Isaac-Sim-RL

NVIDIA Isaac Sim warehouse.usd environment setup  using Pegasus drones extensions

![Isaac](resources/isaacsimss.png)


## Stage 1: Integrating PPO-Driven RL Policy using PX4 Offboard Control

Unlike high-level planners, the proposed system directly learns micro-waypoint navigation policies that output continuous position setpoints in the NED frame. The learned policy demonstrates stable convergence to goal locations, obstacle avoidance, and generalizable behavior under deterministic inference.

| Component | Description                       |
| --------- | --------------------------------- |
| ΔN        | Normalized North distance to goal |
| ΔE        | Normalized East distance to goal  |
| Roll      | UAV roll angle                    |
| Pitch     | UAV pitch angle                   |

The action space includes:

| Action | Description          |
| ------ | -------------------- |
| aₙ     | Northward micro-step |
| aₑ     | Eastward micro-step  |

Rewards Function is designed to be:

r = 3 · (dₜ₋₁ − dₜ)
    − 0.005 · ||a||²
    − 0.001
    + 10.0   if goal reached


![PPO Trajactories](resources/trajectories.png)
