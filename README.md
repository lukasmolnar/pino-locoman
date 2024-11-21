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
- Uses `casadi.opti` with MX expressions. SX are converted to MX using a `casadi.Function`. The OCP revieves a target centroidal momentum, as well as a fixed gait sequence. The centroidal dynamics are formulated as follows:
    - State: Centroidal momentum, generalized coordinates
    - Input: Ground reaction forces, joint velocities (base velocity is calculated through the centroidal momentum matrix)

# TODO

- Add end-effector task.