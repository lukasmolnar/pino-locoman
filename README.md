# Loco-manipulation with Pinocchio

![alt text](b2g_description/b2g_image.png)

## Setup

Create Conda environment, install Pinocchio and MeshCat:

```bash
conda create -n pino python=3.12
conda activate pino
conda install pinocchio meshcat-python -c conda-forge
```

If using OSQP solver:

```bash
conda install osqp -c conda-forge
```

## Examples

Run OCP and MPC examples with:

```bash
python run_mpc_centroidal.py
```

In the example files define:
- Desired COM momentum (linear + angular)
- Desired arm end-effector force + linear velocity
- Desired gait type and period. The gait sequence is then parametrized, swing legs track bezier curves in z-direction and optimize in x/y-directions.


## Optimal Control Problem

The OCP is formulated with the casadi Opti stack. This uses MX expressions, whereas Pinocchio uses SX expressions. For this reason all relevant Pinocchio expressions used in the constraints are converted to MX using casadi Functions.

There are two options for what dynamics to use in the OCP:

### Centroidal Dynamics

- States: Centroidal momentum, generalized coordinates
- Inputs: Ground reaction forces, joint velocities (base velocity is calculated through the centroidal momentum matrix)

### RNEA Dynamics

- States: Generalized coordinates and velocities
- Inputs: Ground reaction forces, joint torques


## Solvers

### Fatrop

Uses auto-structure detection. This currently only works on the centroidal dynamics model, because the RNEA dynamics constraints violate the diagonal structure assumed by Fatrop.

Code generation: The Fatrop solver can be exported to a C file and compiled to a shared library (see codegen folder).

### OSQP

The Opti formulation is converted to a Sequential Quadruatic Program (SQP). Each SQP iteration is solved with OSQP, and the solution is updated using the Armijo line-search method. This works on both centroidal and RNEA dynamics models.
