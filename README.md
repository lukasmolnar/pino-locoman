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
- Inputs: Ground reaction forces, joint torques. For the Fatrop solver the generalized velocities `v_next` also need to be included, to preserve the diagnoal structure. 


## Solvers

### Fatrop

Directly solves the constrained nonlinear optimization problem. Uses auto-structure detection, which significantly reduces the solve time compared to Ipopt. The solver is warm-started with the MPC solution from the previous step. 

Code generation: The Fatrop solver can be exported to a C file and compiled to a shared library (see codegen folder). For hardware deployment the solver can be loaded with `casadi::external`.

### OSQP

The Opti formulation is converted to a Sequential Quadruatic Program (SQP). Each SQP iteration is solved with OSQP, and the solution is updated using the Armijo line-search method.

Code generation: The SQP matrices and vectors can be compiled to a shared library (see codegen folder). For hardware deployment they can be loaded with `casadi::external`. The OSQP setup and solve needs to be formulated in C++ like it is done here.
