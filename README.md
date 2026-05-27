# Dual-Track MPC Planner

Robust dual-track MPC trajectory planner with obstacle avoidance and path tracking based on a 2-DOF vehicle model and quadratic programming (OSQP).

## Features

- Dual-track 2-DOF vehicle dynamics
- Model Predictive Control (MPC)
- Obstacle avoidance
- Reference path tracking
- OSQP-based quadratic optimization
- Dynamic / kinematic low-speed blending
- Steering and comfort constraints

## Project Structure

```text
dualtrack_mpc_planner.py
docs/
└── DualTrack_MPC_Planner_User_Guide.docx
```

## Requirements

```bash
pip install numpy scipy osqp
```

## Example

```python
planner = RobustDualTrackMPCPlanner()
result = planner.plan(state, reference, vx, delta_prev)
```

## Documentation

See:

`docs/DualTrack_MPC_Planner_User_Guide.docx`
