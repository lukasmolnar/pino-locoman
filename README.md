# Loco-manipulation with Pinocchio

![alt text](b2g_description/b2g_image.png)

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

Centroidal Dynamics:
- Old: Uses Casadi opti stack. Doesn't consider ground reaction forces.
- New: Uses Casadi SX formulation. This is necessary because `pinocchio.casadi` provides this (eg. for Jacobians). Considers ground reaction forces and end-effector constraints. Solution is not yet satisfactory.


## TODO

- Debug new centroidal dynamics formulation.
- Add motion commands: Gait pattern, stepping locations / bezier curves, end effector position / rotation. Add to optimization through constraints & objective.