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
        Q = robot_class.Q
        R = robot_class.R

        mass = self.data.mass[0]
        dt = self.gait_sequence.dt
        self.dyn = CentroidalDynamics(self.model, mass, dt)

        n_contacts = self.gait_sequence.n_contacts
        n_forces = 3 * n_contacts
        n_joints = self.model.nv - 6

        ndx = len(self.x_nom)  # same size!
        nu = n_forces + n_joints

        # Desired state and input
        dx_des = np.zeros(ndx)
        dx_des[:6] = com_goal
        f_des = np.tile([0, 0, 9.81 * mass / n_contacts], n_contacts)
        u_des = np.concatenate((f_des, np.zeros(n_joints)))

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
            obj += 0.5 * err_dx.T @ Q @ err_dx
            obj += 0.5 * err_u.T @ R @ err_u
        
        # Final state
        dx = self.DX[self.nodes]
        err_dx = dx - dx_des
        obj += 0.5 * dx.T @ Q @ dx

        self.opti.minimize(obj)

        # Initial state
        self.opti.subject_to(self.DX[0] == [0] * ndx)

        for i in range(self.nodes):
            # Get end-effector frames
            contact_ee_ids = [self.model.getFrameId(f) for f in self.gait_sequence.contact_list[i]]
            swing_ee_ids = [self.model.getFrameId(f) for f in self.gait_sequence.swing_list[i]]

            # Gather all state and input info
            x = self.x_nom + self.DX[i]
            u = self.U[i]
            h = x[:6]
            q = x[6:]
            forces = u[:n_forces]
            dq_j = u[n_forces:]
            dq_b = self.dyn.get_base_velocity()(h, q, dq_j)
            dq = casadi.vertcat(dq_b, dq_j)

            # Dynamics constraint
            x_next = self.x_nom + self.DX[i+1]
            x_next_dyn = self.dyn.centroidal_dynamics(contact_ee_ids)(x, u, dq_b)
            self.opti.subject_to(x_next == x_next_dyn)

            # Contact constraints
            for idx, frame_id in enumerate(contact_ee_ids):
                # Friction cone
                f_e = forces[idx * 3 : (idx + 1) * 3]
                f_normal = f_e[2]
                f_tangent_square = f_e[0]**2 + f_e[1]**2

                self.opti.subject_to(f_normal >= 0)
                self.opti.subject_to(mu**2 * f_normal**2 >= f_tangent_square)

                # Zero end-effector velocity
                vel_lin = self.dyn.get_frame_velocity(frame_id)(q, dq)
                self.opti.subject_to(vel_lin == [0] * 3)

            # Swing foot trajectory
            for frame_id in swing_ee_ids:
                # Track bezier velocity (only in z)
                vel_z = self.dyn.get_frame_velocity(frame_id)(q, dq)[2]
                vel_z_des = self.gait_sequence.get_bezier_vel_z(0, i, h=0.1)
                self.opti.subject_to(vel_z == vel_z_des)

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

        h_nom = self.x_nom[:6]
        q_nom = self.x_nom[6:]

        for i in range(self.nodes):
            dx_sol = self.sol.value(self.DX[i])
            u_sol = self.sol.value(self.U[i])
            h = h_nom + dx_sol[:6]
            q = q_nom + dx_sol[6:]
            self.hs.append(h)
            self.qs.append(q)
            self.us.append(u_sol)

        dx_last = self.sol.value(self.DX[self.nodes])
        h_last = h_nom + dx_last[:6]
        q_last = q_nom + dx_last[6:]
        self.hs.append(h_last)
        self.qs.append(q_last)

    def _compute_gaps(self):
        # TODO
        return

    def _simulate_step(self, x, u):
        # TODO
        return

