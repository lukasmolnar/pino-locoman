# Loco-manipulation with Pinocchio

![alt text](image.png)

## Setup

Create Conda environment, install Pinocchio and MeshCat:

```bash
conda create -n pino python=3.12
conda activate pino
conda install pinocchio meshcat-python -c conda-forge
```

## Examples

Adapted examples from Pinocchio GitHub for B2G description (Unitree B2 quadruped + Z1 arm). Visualize with MeshCat (no physics simulator).
- COM: Move center of mass up and down, while arm end effector position remains constant.
- IK: Given desired arm end effector position and rotation, compute whole-body joint angles.
- Casadi: Formulate optimization problem in joint space (states: joint positions, inputs: joint velocities). Track desired goal state.

New example:
- Dynamics: Uses centroidal dynamics through Pinocchio. Cannot yet interact with ground.


## TODO

- Interact with ground. Get forces from robot model (?)
- Add forces to optimization problem so the centroidal dynamics consider them.
- Add motion commands: Gait pattern, stepping locations / bezier curves, end effector position / rotation. Add to optimization through constraints & objective.