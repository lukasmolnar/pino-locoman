import numpy as np
import pinocchio as pin
import casadi

from helpers import *
from centroidal_dynamics import CentroidalDynamics


class OptimalControlProblem:
    def __init__(
            self,
            model,
            data,
            gait_sequence,
            com_goal,
            x_nom,
            mu=0.5,
        ):
        self.opti = casadi.Opti()
        self.model = model
        self.data = data
        self.x_nom = x_nom
        self.nodes = gait_sequence.nodes

        ndx = len(x_nom) - 1  # exclude quaternion
        nf = 3 * gait_sequence.n_contacts  # contact forces
        nj = ndx - 12  # joints (excluding COM and base)
        nu = nf + nj  # total inputs

        self.dyn = CentroidalDynamics(
            model=model,
            mass=self.data.mass[0],
            dt=gait_sequence.dt
        )

        # COM selection matrix for tracking
        S_com = np.zeros((3, ndx))
        for i, com in enumerate([0, 1, 5]):  # x, y, yaw indices
            S_com[i, com] = 1

        # Decision variables
        self.c_dxs = self.opti.variable(ndx, self.nodes + 1)  # state trajectory
        self.c_us = self.opti.variable(nu, self.nodes)  # control trajectory

        # Objective function
        obj = 0

        # Tracking and regularization
        for i in range(self.nodes + 1):
            dx = self.c_dxs[:, i]
            err_joints = dx[12:]  # regularization
            err_com = S_com @ dx - com_goal  # tracking
            obj += (
                1e-3 * 0.5 * err_joints.T @ err_joints
                + 1 * 0.5 * err_com.T @ err_com
            )

        self.opti.minimize(obj)

        for i in range(self.nodes):
            # Get end-effector frames
            contact_ee_ids = [model.getFrameId(f) for f in gait_sequence.contact_list[i]]
            swing_ee_ids = [model.getFrameId(f) for f in gait_sequence.swing_list[i]]

            # Gather all state and input info
            x = self.dyn.state_integrate()(x_nom, self.c_dxs[:, i])
            u = self.c_us[:, i]
            h = x[:6]
            q = x[6:]
            dq_j = u[nf:]
            dq_b = self.dyn.get_base_velocity()(h, q, dq_j)
            dq = casadi.vertcat(dq_b, dq_j)

            # Dynamics constraint
            x_next = self.dyn.state_integrate()(x_nom, self.c_dxs[:, i + 1])
            x_next_dyn = self.dyn.centroidal_dynamics(contact_ee_ids)(x, u, dq_b)
            gap = self.dyn.state_difference()(x_next, x_next_dyn)
            self.opti.subject_to(gap == [0] * ndx)

            # Contact constraints
            for idx, frame_id in enumerate(contact_ee_ids):
                # Friction cone
                f_e = self.c_us[idx * 3 : (idx + 1) * 3, i]
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
                vel_z_des = gait_sequence.get_bezier_vel_z(0, i, h=0.1)
                self.opti.subject_to(vel_z == vel_z_des)

        # State and input constraints
        self.opti.subject_to(self.opti.bounded(-50, self.c_dxs[:6, :], 50))  # COM
        self.opti.subject_to(self.opti.bounded(-2, self.c_dxs[6:, :], 2))  # q
        self.opti.subject_to(self.opti.bounded(-500, self.c_us[:nf, :], 500))  # f_e
        self.opti.subject_to(self.opti.bounded(-1, self.c_us[nf:, :], 1))  # dq_j

        # Initial state: nominal state
        dx_0 = self.c_dxs[:, 0]
        self.opti.subject_to(dx_0 == [0] * ndx)

        # Warm start
        self.opti.set_initial(
            self.c_dxs, np.vstack([np.zeros(ndx) for _ in range(self.nodes + 1)]).T
        )
        self.opti.set_initial(
            self.c_us, np.vstack([np.zeros(nu) for _ in range(self.nodes)]).T
        )

    def solve(self, approx_hessian=True):
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

        # Solver initialization
        self.opti.solver("ipopt", opts)  # set numerical backend

        try:
            self.sol = self.opti.solve()
        except:  # noqa: E722
            self.sol = self.opti.debug

        if self.sol.stats()["return_status"] == "Solve_Succeeded":
            self._retract_trajectory()
            self._compute_gaps()

    def _retract_trajectory(self):
        self.hs = []
        self.qs = []
        self.us = []
        self.gaps = []

        h_nom = np.array(self.x_nom)[:6]
        q_nom = np.array(self.x_nom)[6:]

        for _, (dx_sol, u_sol) in enumerate(
            zip(self.sol.value(self.c_dxs).T, self.sol.value(self.c_us).T)
        ):
            dh = dx_sol[:6]
            dq = dx_sol[6:]
            h = h_nom + dh
            q = pin.integrate(self.model, q_nom, dq)
            self.hs.append(h)
            self.qs.append(q)
            self.us.append(u_sol)

        dq_last = self.sol.value(self.c_dxs).T[self.nodes, 6:]
        q_last = pin.integrate(self.model, q_nom, dq_last)
        self.qs.append(q_last)

    def _compute_gaps(self):
        # TODO
        return

    def _simulate_step(self, x, u):
        # TODO
        return

