import numpy as np
import casadi

from helpers import *
from centroidal_dynamics import CentroidalDynamics


class OptimalControlProblem:
    def __init__(
            self,
            robot_class,
            com_goal,
            mu=0.7,
        ):
        self.opti = casadi.Opti()
        self.model = robot_class.model
        self.data = robot_class.data
        self.x_nom = robot_class.x_nom

        self.gait_sequence = robot_class.gait_sequence
        self.nodes = self.gait_sequence.nodes
        self.dt = self.gait_sequence.dt
        self.Q = robot_class.Q
        self.R = robot_class.R

        mass = self.data.mass[0]
        self.dyn = CentroidalDynamics(self.model, mass)

        n_contact_feet = self.gait_sequence.n_contacts
        nf = robot_class.nf
        nj = robot_class.nj
        nv = robot_class.nv

        # State and input dimensions
        ndx = 6 + nv  # COM + generalized velocities
        nu = nf + nj  # forces + joint velocities

        # Desired state and input
        dx_des = np.zeros(ndx)
        dx_des[:6] = com_goal
        f_des = np.tile([0, 0, 9.81 * mass / n_contact_feet], n_contact_feet)
        if robot_class.arm_ee:
            f_des = np.concatenate((f_des, robot_class.arm_f_des))
        u_des = np.concatenate((f_des, np.zeros(nj)))

        # Decision variables
        self.DX = []
        self.U = []
        for i in range(self.nodes):
            self.DX.append(self.opti.variable(ndx))
            self.U.append(self.opti.variable(nu))
        self.DX.append(self.opti.variable(ndx))

        # Objective function
        obj = 0

        # Tracking and regularization
        for i in range(self.nodes):
            dx = self.DX[i]
            u = self.U[i]
            err_dx = dx - dx_des
            err_u = u - u_des
            obj += 0.5 * err_dx.T @ self.Q @ err_dx
            obj += 0.5 * err_u.T @ self.R @ err_u
        
        # Final state
        dx = self.DX[self.nodes]
        err_dx = dx - dx_des
        obj += 0.5 * dx.T @ self.Q @ dx

        self.opti.minimize(obj)

        # Initial state
        self.opti.subject_to(self.DX[0] == [0] * ndx)

        for i in range(self.nodes):
            # Get end-effector frames
            contact_ee_ids = [self.model.getFrameId(f) for f in self.gait_sequence.contact_list[i]]
            swing_ee_ids = [self.model.getFrameId(f) for f in self.gait_sequence.swing_list[i]]
            if robot_class.arm_ee:
                arm_ee_id = self.model.getFrameId(robot_class.arm_ee)
            else:
                arm_ee_id = None

            # Gather all state and input info
            dx = self.DX[i]
            x = self.dyn.state_integrate()(self.x_nom, dx)
            u = self.U[i]
            h = x[:6]
            q = x[6:]
            forces = u[:nf]
            dq_j = u[nf:]
            dq_b = self.dyn.get_base_velocity()(h, q, dq_j)
            dq = casadi.vertcat(dq_b, dq_j)

            # Dynamics constraint
            dx_next = self.DX[i+1]
            dx_dyn = self.dyn.centroidal_dynamics(contact_ee_ids, arm_ee_id)(x, u, dq_b)
            self.opti.subject_to(dx_next == dx + dx_dyn * self.dt)

            # Contact constraints
            for idx, frame_id in enumerate(contact_ee_ids):
                # Friction cone
                f_e = forces[idx * 3 : (idx + 1) * 3]
                f_normal = f_e[2]
                f_tangent_square = f_e[0]**2 + f_e[1]**2

                self.opti.subject_to(f_normal >= 0)
                self.opti.subject_to(mu**2 * f_normal**2 >= f_tangent_square)

                # Zero end-effector velocity (linear)
                vel = self.dyn.get_frame_velocity(frame_id)(q, dq)
                vel_lin = vel[:3]
                self.opti.subject_to(vel_lin == [0] * 3)

            # Swing foot trajectory
            for frame_id in swing_ee_ids:
                # Track bezier velocity (only in z)
                vel_z = self.dyn.get_frame_velocity(frame_id)(q, dq)[2]
                vel_z_des = self.gait_sequence.get_bezier_vel_z(0, i, h=0.1)
                self.opti.subject_to(vel_z == vel_z_des)

            # Arm task
            if arm_ee_id:
                # Zero end-effector velocity (linear and angular)
                vel = self.dyn.get_frame_velocity(arm_ee_id)(q, dq)
                self.opti.subject_to(vel == [0] * 6)

                # Force at end-effector
                f_e = forces[3 * n_contact_feet :]
                self.opti.subject_to(f_e == robot_class.arm_f_des)

            # State and input constraints
            # self.opti.subject_to(self.opti.bounded(-10, h, 10))  # COM
            # self.opti.subject_to(self.opti.bounded(-1, q, 1))  # q
            # self.opti.subject_to(self.opti.bounded(-500, forces, 500))  # forces
            # self.opti.subject_to(self.opti.bounded(-1, dq_j, 1))  # dq_j

            # Warm start
            self.opti.set_initial(self.DX[i], dx_des)
            self.opti.set_initial(self.U[i], u_des)

        # Final state
        # x_final = self.x_nom + self.DX[self.nodes]
        # self.opti.subject_to(self.opti.bounded(-10, x_final[:6], 10))  # COM
        # self.opti.subject_to(self.opti.bounded(-1, x_final[6:], 1))  # q

        # Warm start
        self.opti.set_initial(self.DX[self.nodes], dx_des)


    def solve(self, solver="fatrop", approx_hessian=True):
        if solver == "ipopt":
            opts = {"verbose": False}
            opts["ipopt"] = {
                "max_iter": 1000,
                "linear_solver": "mumps",
                "tol": 1e-4,
                "mu_strategy": "adaptive",
                "nlp_scaling_method": "gradient-based",
            }

            if approx_hessian:
                opts["ipopt"]["hessian_approximation"] = "limited-memory"

        elif solver == "fatrop":
            opts = {
                "expand": True,
                "structure_detection": "auto",
                "debug": True,
            }
            opts["fatrop"] = {
                "mu_init": 0.1
            }

        else:
            raise ValueError(f"Solver {solver} not supported")

        # Solver initialization
        self.opti.solver(solver, opts)

        try:
            self.sol = self.opti.solve()
        except:  # noqa: E722
            self.sol = self.opti.debug

        # if self.sol.stats()["return_status"] == "Solve_Succeeded":
        self._retract_trajectory()


    def _retract_trajectory(self):
        self.hs = []
        self.qs = []
        self.us = []
        self.gaps = []

        for i in range(self.nodes):
            dx_sol = self.sol.value(self.DX[i])
            x_sol = self.dyn.state_integrate()(self.x_nom, dx_sol)
            u_sol = self.sol.value(self.U[i])
            self.hs.append(np.array(x_sol[:6]))
            self.qs.append(np.array(x_sol[6:]))
            self.us.append(np.array(u_sol))

        dx_last = self.sol.value(self.DX[self.nodes])
        x_last = self.dyn.state_integrate()(self.x_nom, dx_last)
        self.hs.append(np.array(x_last[:6]))
        self.qs.append(np.array(x_last[6:]))

    def _compute_gaps(self):
        # TODO
        return

    def _simulate_step(self, x, u):
        # TODO
        return

