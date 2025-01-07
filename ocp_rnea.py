import numpy as np
import casadi

from helpers import *
from dynamics_rnea import DynamicsRNEA


class OptimalControlRNEA:
    def __init__(
            self,
            robot,
            nodes,
            step_height=0.1,
            mu=0.7,
        ):
        self.opti = casadi.Opti()
        self.model = robot.model
        self.data = robot.data
        self.gait_sequence = robot.gait_sequence
        self.nq = robot.nq
        self.nv = robot.nv
        self.nf = robot.nf
        self.nj = robot.nj

        self.nodes = nodes
        self.nf = robot.nf

        mass = self.data.mass[0]
        dt = self.gait_sequence.dt
        ee_ids = robot.ee_ids
        arm_ee_id = robot.arm_ee_id
        n_feet = len(ee_ids)
        n_contact_feet = self.gait_sequence.n_contacts

        # Weights
        Q_diag = np.concatenate((
            [1000] * 3,  # base pos
            [1000] * 3,  # base rot
            [10] * self.nj  # joints
        ))
        Q_diag = np.concatenate((Q_diag, Q_diag))  # both pos and vel
        R_diag = np.concatenate((
            [1e-1] * self.nj,  # joint torques
            [1e-2] * self.nf  # forces
        ))
        Q = np.diag(Q_diag)
        R = np.diag(R_diag)

        # State and inputs to optimize
        nx = self.nq + self.nv  # positions + velocities
        ndx_opt = self.nv * 2  # position deltas + velocities
        nu_opt = self.nj + self.nf  # joint torques + forces

        # Dynamics
        self.dyn = DynamicsRNEA(self.model, mass, ee_ids)

        # Decision variables
        self.DX_opt = []
        self.U_opt = []
        for i in range(self.nodes):
            self.DX_opt.append(self.opti.variable(ndx_opt))
            self.U_opt.append(self.opti.variable(nu_opt))
        self.DX_opt.append(self.opti.variable(ndx_opt))

        # Parameters
        self.x_init = self.opti.parameter(nx)  # initial state
        self.gait_idx = self.opti.parameter()  # where we are within the gait
        self.contact_schedule = self.opti.parameter(n_feet, self.nodes)  # gait schedule: contacts / bezier indices
        self.com_goal = self.opti.parameter(6)  # linear + angular momentum
        self.arm_f_des = self.opti.parameter(3)  # force at end-effector
        self.arm_vel_des = self.opti.parameter(3)  # velocity at end-effector

        # Desired state and input
        v_des = casadi.vertcat(self.com_goal, [0] * self.nj)
        x_des = casadi.vertcat(robot.q0, v_des)
        dx_des = self.dyn.state_difference()(self.x_init, x_des)
        f_des = np.tile([0, 0, 9.81 * mass / n_contact_feet], n_feet)
        if robot.arm_ee_id:
            # TODO: Check if arm force = 0 is ok (it is constrained later)
            f_des = np.concatenate((f_des, np.zeros(3)))
        u_des = np.concatenate((np.zeros(robot.nj), f_des))  # zero torque

        # OBJECTIVE
        obj = 0

        # Tracking and regularization
        for i in range(self.nodes):
            dx = self.DX_opt[i]
            u = self.U_opt[i]
            err_dx = dx - dx_des
            err_u = u - u_des
            obj += 0.5 * err_dx.T @ Q @ err_dx
            obj += 0.5 * err_u.T @ R @ err_u
        
        # Final state
        dx = self.DX_opt[self.nodes]
        err_dx = dx - dx_des
        obj += 0.5 * dx.T @ Q @ dx

        # CONSTRAINTS
        self.opti.subject_to(self.DX_opt[0][:self.nv] == [0] * self.nv)  # initial pos (not vel)

        for i in range(self.nodes):
            # Gather all state and input info
            dx_opt = self.DX_opt[i]
            dq = dx_opt[:self.nv]  # NOTE: delta q, not velocity
            x = self.dyn.state_integrate()(self.x_init, dx_opt)
            u = self.U_opt[i]
            q = x[:self.nq]
            v = x[self.nq:]
            forces = u[self.nj:]

            # Dynamics constraint
            dx_next = self.DX_opt[i+1]
            dq_next = dx_next[:self.nv]
            x_next = self.dyn.state_integrate()(self.x_init, dx_next)
            v_next = x_next[self.nq:]
            self.opti.subject_to(dq_next == dq + v * dt)

            # RNEA constraint
            # a = (v_next - v) / dt  # finite difference
            a = np.zeros(self.nv)  # TODO: fix this
            dyn_gaps = self.dyn.rnea_constraint(arm_ee_id)(x, u, a)
            self.opti.subject_to(dyn_gaps[6:] == [0] * self.nj)  # TODO: fix base

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
                vel = self.dyn.get_frame_velocity(frame_id)(q, v)
                vel_lin = vel[:3]
                self.opti.subject_to(in_contact * vel_lin == [0] * 3)

                # Zero end-effector force
                self.opti.subject_to((1 - in_contact) * f_e == [0] * 3)

                # Track bezier velocity (only in z)
                vel_z = vel_lin[2]
                vel_z_des = self.gait_sequence.get_bezier_vel_z(0, bezier_idx, h=step_height)
                vel_diff = vel_z - vel_z_des
                self.opti.subject_to((1 - in_contact) * vel_diff == 0)

                # Humanoid
                if n_feet == 2:
                    # Flat feet
                    vel_ang = vel[3:]
                    self.opti.subject_to(vel_ang == [0] * 3)

            # Humanoid
            if n_feet == 2:
                # Minimum distance between feet
                min_dist = 0.095  # standing pose is 0.1
                foot_0 = self.dyn.get_frame_position(ee_ids[0])(q)
                foot_1 = self.dyn.get_frame_position(ee_ids[1])(q)
                foot_diff = foot_1 - foot_0
                self.opti.subject_to(foot_diff[0]**2 + foot_diff[1]**2 >= min_dist**2)

            # Arm task
            if arm_ee_id:
                # Zero end-effector velocity (linear and angular)
                vel = self.dyn.get_frame_velocity(arm_ee_id)(q, v)
                vel_lin = vel[:3]
                vel_diff = vel_lin - self.arm_vel_des
                self.opti.subject_to(vel_diff == [0] * 3)
                # obj += 0.5 * vel_diff.T @ vel_diff

                # Force at end-effector (after all feet)
                f_e = forces[3*n_feet:]
                self.opti.subject_to(f_e == self.arm_f_des)

            # Warm start
            self.opti.set_initial(self.DX_opt[i], np.zeros(ndx_opt))
            self.opti.set_initial(self.U_opt[i], u_des)

        # Warm start
        self.opti.set_initial(self.DX_opt[self.nodes], np.zeros(ndx_opt))

        # OBJECTIVE
        self.opti.minimize(obj)

        # TODO: Clean up how solutions are stored
        self.DX_prev = None
        self.U_prev = None
        self.lam_g = None
        self.qs = []
        self.vs = []
        self.taus = []
        self.fs = []

    def update_initial_state(self, x_init):
        self.opti.set_value(self.x_init, x_init)

    def update_gait_sequence(self, shift_idx=0):
        contact_schedule = self.gait_sequence.shift_contact_schedule(shift_idx)
        self.opti.set_value(self.contact_schedule, contact_schedule[:, :self.nodes])
        self.opti.set_value(self.gait_idx, shift_idx)

    def set_com_goal(self, com_goal):
        self.opti.set_value(self.com_goal, com_goal)

    def set_arm_task(self, arm_f_des, arm_vel_des):
        self.opti.set_value(self.arm_f_des, arm_f_des)
        self.opti.set_value(self.arm_vel_des, arm_vel_des)

    def warm_start(self):
        # Shift previous solution
        # NOTE: No warm-start for last node, copying the 2nd last node performs worse.
        if self.DX_prev is not None:
            DX_init = self.DX_prev[1]
            for i in range(self.nodes):
                DX_diff = self.DX_prev[i+1] - DX_init
                self.opti.set_initial(self.DX_opt[i], DX_diff)

        if self.U_prev is not None:
            for i in range(self.nodes - 1):
                self.opti.set_initial(self.U_opt[i], self.U_prev[i+1])

        if self.lam_g is not None:
            self.opti.set_initial(self.opti.lam_g, self.lam_g)

    def init_solver(self, solver="fatrop", compile_solver=False):
        if solver == "ipopt":
            opts = {"verbose": False}
            opts["ipopt"] = {
                "max_iter": 1000,
                "linear_solver": "mumps",
                "tol": 1e-4,
                "mu_strategy": "adaptive",
                "nlp_scaling_method": "gradient-based",
                "hessian_approximation": "limited-memory",
            }

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
                "warm_start_init_point": True,
                "warm_start_mult_bound_push": 1e-7,
                "bound_push": 1e-7,
            }

        else:
            raise ValueError(f"Solver {solver} not supported")

        # Solver initialization
        self.opti.solver(solver, opts)

        # Code generation
        if compile_solver:
            solver_function = self.opti.to_function(
                "compiled_solver",
                [self.x_init, self.contact_schedule, self.gait_idx, self.com_goal, self.arm_f_des, self.arm_vel_des],  # parameters
                [self.DX_opt[1], self.U_opt[0]]  # output
            )
            solver_function.generate("compiled_solver.c")

    def solve(self, retract_all=True):
        try:
            self.sol = self.opti.solve()
        except:  # noqa: E722
            self.sol = self.opti.debug

        # TODO: Check solution status
        self._retract_trajectory(retract_all)

        # Store previous solution
        self.DX_prev = [self.sol.value(dx) for dx in self.DX_opt]
        self.U_prev = [self.sol.value(u) for u in self.U_opt]

        # Store dual variables
        self.lam_g = self.sol.value(self.opti.lam_g)

    def _retract_trajectory(self, retract_all=True):
        x_init = self.opti.value(self.x_init)

        for i in range(self.nodes):
            dx_sol = self.sol.value(self.DX_opt[i])
            x_sol = self.dyn.state_integrate()(x_init, dx_sol)
            u_sol = self.sol.value(self.U_opt[i])
            self.qs.append(np.array(x_sol[:self.nq]))
            self.vs.append(np.array(x_sol[self.nq:]))
            self.taus.append(np.array(u_sol[:self.nj]))
            self.fs.append(np.array(u_sol[self.nj:]))

            if i == 0 and not retract_all:
                return

        dx_last = self.sol.value(self.DX_opt[self.nodes])
        x_last = self.dyn.state_integrate()(x_init, dx_last)
        self.qs.append(np.array(x_last[:self.nq]))
        self.vs.append(np.array(x_last[self.nq:]))
        self.taus.append(np.array(u_sol[:self.nj]))
        self.fs.append(np.array(u_sol[self.nj:]))
