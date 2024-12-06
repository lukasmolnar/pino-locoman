import numpy as np
import casadi

from helpers import *
from centroidal_dynamics import CentroidalDynamics


class OptimalControlProblem:
    def __init__(
            self,
            robot_class,
            nodes,
            com_goal,
            step_height=0.1,
            mu=0.7,
        ):
        self.opti = casadi.Opti()
        self.model = robot_class.model
        self.data = robot_class.data
        self.nodes = nodes
        self.com_goal = com_goal

        self.gait_sequence = robot_class.gait_sequence
        self.gait_N = self.gait_sequence.N
        self.dt = self.gait_sequence.dt
        self.Q = robot_class.Q
        self.R = robot_class.R

        mass = self.data.mass[0]
        ee_ids = [self.model.getFrameId(f) for f in self.gait_sequence.feet]
        self.dyn = CentroidalDynamics(self.model, mass, ee_ids)

        if robot_class.arm_ee:
            arm_ee_id = self.model.getFrameId(robot_class.arm_ee)
        else:
            arm_ee_id = None

        n_feet = len(ee_ids)
        n_contact_feet = self.gait_sequence.n_contacts
        nq = robot_class.nq
        nv = robot_class.nv
        nf = robot_class.nf
        nj = robot_class.nj

        # State and input dimensions
        nx = 6 + nq  # COM + generalized coordinates
        ndx = 6 + nv  # COM + generalized velocities
        nu = nf + nj  # forces + joint velocities

        # Decision variables
        self.DX = []
        self.U = []
        for i in range(self.nodes):
            self.DX.append(self.opti.variable(ndx))
            self.U.append(self.opti.variable(nu))
        self.DX.append(self.opti.variable(ndx))

        # Parameters
        self.x_init = self.opti.parameter(nx)  # initial state
        self.gait_idx = self.opti.parameter()  # where we are within the gait
        self.contact_schedule = self.opti.parameter(n_feet, self.nodes)  # gait schedule: contacts / bezier indices

        # Desired state and input
        x_des = robot_class.x_nom
        x_des[:6] = self.com_goal
        dx_des = self.dyn.state_difference()(self.x_init, x_des)
        f_des = np.tile([0, 0, 9.81 * mass / n_contact_feet], n_feet)
        if robot_class.arm_ee:
            f_des = np.concatenate((f_des, robot_class.arm_f_des))
        u_des = np.concatenate((f_des, np.zeros(nj)))

        # OBJECTIVE
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


        # CONSTRAINTS
        self.opti.subject_to(self.DX[0] == [0] * ndx)

        for i in range(self.nodes):
            # Gather all state and input info
            dx = self.DX[i]
            x = self.dyn.state_integrate()(self.x_init, dx)
            u = self.U[i]
            h = x[:6]
            q = x[6:]
            forces = u[:nf]
            dq_j = u[nf:]
            dq_b = self.dyn.get_base_velocity()(h, q, dq_j)
            dq = casadi.vertcat(dq_b, dq_j)

            # Dynamics constraint
            dx_next = self.DX[i+1]
            dx_dyn = self.dyn.centroidal_dynamics(arm_ee_id)(x, u, dq_b)
            self.opti.subject_to(dx_next == dx + dx_dyn * self.dt)

            # Contact and swing constraints
            for idx, frame_id in enumerate(ee_ids):
                f_e = forces[idx * 3 : (idx + 1) * 3]

                # Determine contact and bezier index from schedule
                schedule_idx = self.contact_schedule[idx, i]
                in_contact = casadi.if_else(schedule_idx == 0, 1, 0)
                bezier_idx = schedule_idx - 1

                # Friction cone
                f_normal = f_e[2]
                f_tangent_square = f_e[0]**2 + f_e[1]**2
                self.opti.subject_to(in_contact * f_normal >= 0)
                self.opti.subject_to(in_contact * mu**2 * f_normal**2 >= in_contact * f_tangent_square)

                # Zero end-effector velocity (linear)
                vel = self.dyn.get_frame_velocity(frame_id)(q, dq)
                vel_lin = vel[:3]
                self.opti.subject_to(in_contact * vel_lin == [0] * 3)

                # Zero end-effector force
                self.opti.subject_to((1 - in_contact) * f_e == [0] * 3)

                # Track bezier velocity (only in z)
                vel_z = vel_lin[2]
                vel_z_des = self.gait_sequence.get_bezier_vel_z(0, bezier_idx, h=step_height)
                vel_diff = vel_z - vel_z_des
                self.opti.subject_to((1 - in_contact) * vel_diff == 0)

                # Humanoid example: flat feet
                if n_feet == 2:
                    vel_ang = vel[3:]
                    self.opti.subject_to(vel_ang == [0] * 3)

            # Arm task
            if arm_ee_id:
                # Zero end-effector velocity (linear and angular)
                vel = self.dyn.get_frame_velocity(arm_ee_id)(q, dq)
                vel_lin = vel[:3]
                vel_diff = vel_lin - robot_class.arm_vel_des
                self.opti.subject_to(vel_diff == [0] * 3)
                # obj += 0.5 * vel_diff.T @ vel_diff

                # Force at end-effector (after all feet)
                f_e = forces[3*n_feet:]
                self.opti.subject_to(f_e == robot_class.arm_f_des)

            # Warm start
            self.opti.set_initial(self.DX[i], np.zeros(ndx))
            self.opti.set_initial(self.U[i], u_des)

        # Warm start
        self.opti.set_initial(self.DX[self.nodes], np.zeros(ndx))

        # OBJECTIVE
        self.opti.minimize(obj)

        # Store solutions
        self.hs = []
        self.qs = []
        self.us = []

    def update_initial_state(self, x_init):
        self.opti.set_value(self.x_init, x_init)

    def update_gait_sequence(self, shift_idx=0):
        contact_schedule = self.gait_sequence.shift_contact_schedule(shift_idx)
        self.opti.set_value(self.contact_schedule, contact_schedule[:, :self.nodes])
        self.opti.set_value(self.gait_idx, shift_idx)

    def warm_start(self, dx_prev=None, u_prev=None):
        # Shift previous solution
        if dx_prev is not None:
            for i in range(self.nodes):
                self.opti.set_initial(self.DX[i], dx_prev[i+1])

        if u_prev is not None:
            for i in range(self.nodes - 1):
                self.opti.set_initial(self.U[i], u_prev[i+1])

    def init_solver(self, solver="fatrop", approx_hessian=True):
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
                "print_level": 0,
                "tol": 1e-3,
                "mu_init": 1e-8,
            }

            # TODO: Check if code-generation is possible
            # opts["jit"] = True
            # opts["compiler"] = "shell"
            # opts["jit_options"] = {
            #     "flags": ["-O3"],
            #     "verbose": True,
            # }

        else:
            raise ValueError(f"Solver {solver} not supported")

        # Solver initialization
        self.opti.solver(solver, opts)

    def solve(self, retract_all=True):

        try:
            self.sol = self.opti.solve()
        except:  # noqa: E722
            self.sol = self.opti.debug

        # TODO: Check solution status
        self._retract_trajectory(retract_all)

        # Store previous solution
        self.dx_prev = [self.sol.value(dx) for dx in self.DX]
        self.u_prev = [self.sol.value(u) for u in self.U]

    def _retract_trajectory(self, retract_all=True):
        x_init = self.opti.value(self.x_init)

        for i in range(self.nodes):
            dx_sol = self.sol.value(self.DX[i])
            x_sol = self.dyn.state_integrate()(x_init, dx_sol)
            u_sol = self.sol.value(self.U[i])
            self.hs.append(np.array(x_sol[:6]))
            self.qs.append(np.array(x_sol[6:]))
            self.us.append(np.array(u_sol))

            if i == 0 and not retract_all:
                return

        dx_last = self.sol.value(self.DX[self.nodes])
        x_last = self.dyn.state_integrate()(x_init, dx_last)
        self.hs.append(np.array(x_last[:6]))
        self.qs.append(np.array(x_last[6:]))

    def _compute_gaps(self):
        # TODO
        return

    def _simulate_step(self, x, u):
        # TODO
        return

